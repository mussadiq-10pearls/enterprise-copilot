from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from openai import BadRequestError
from app.config import OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_DEPLOYMENT
from app.agent.state import AgentState
from app.agent.tools import search_docs, store_preference, refuse_request
from app.memory.long_term import load_memory
from app.rag.retriever import search_company_docs as search_company_docs_raw
from app.safety.injection import detect_prompt_injection
from langchain_core.messages import ToolMessage

# LLM initialization
llm = AzureChatOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_API_KEY,
    azure_deployment=OPENAI_DEPLOYMENT,
    model=OPENAI_DEPLOYMENT,
    api_version="2024-02-15-preview",
    temperature=0,
)

tools = [search_docs, store_preference, refuse_request]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    # --- Prompt injection check ---
    if state["messages"]:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "content") and detect_prompt_injection(last_msg.content):
            refusal = AIMessage(content="I cannot process this request due to safety policy.")
            return {
                "messages": [refusal],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }

    # Load long‑term memory preferences
    user_prefs = load_memory(state["user_id"])
    pref_text = ", ".join([f"{k}: {v}" for k, v in user_prefs.items()]) if user_prefs else "None"

    system_prompt = SystemMessage(content=f"""You are an enterprise copilot. 

**CRITICAL RULE**: For ANY question that asks for information that might be in company documents (e.g., lists, policies, internal data, hospital names, employee details, workation rules, credentials), you MUST use the `search_docs` tool. Do NOT answer from your own general knowledge.

ONLY answer directly without using a tool if the user is just greeting you or asking about your capabilities (e.g., "hello", "what can you do?").

User preferences: {pref_text}

Tools:
- `search_docs`: use for ALL factual questions about the company, documents, policies, internal data, or any list/information that could be in the documents. Input: a query string.
- `store_preference`: use when the user explicitly asks you to remember something. Input: user_id, key, value.
- `refuse_request`: use when the request is unsafe, off‑topic, or violates policy. Input: reason.

If `search_docs` returns no results, politely say you couldn't find the information.""")

    try:
        response = llm_with_tools.invoke([system_prompt] + state["messages"])
    except BadRequestError as e:
        error_msg = AIMessage(content="I cannot process that request due to safety policies. Please rephrase.")
        return {
            "messages": [error_msg],
            "next_action": "respond",
            "retrieved_chunks": state.get("retrieved_chunks", [])
        }

    if hasattr(response, "tool_calls") and response.tool_calls:
        return {
            "messages": [response],
            "next_action": "execute_tool",
            "retrieved_chunks": state.get("retrieved_chunks", [])
        }
    else:
        answer = response.content
        chunks = state.get("retrieved_chunks", [])
        if chunks:
            from app.safety.groundedness import groundedness_check
            if not groundedness_check(state["messages"][-1].content, chunks, answer):
                answer += "\n\n[Note: This answer may not be fully supported by the retrieved documents.]"
        return {
            "messages": [AIMessage(content=answer)],
            "next_action": "respond",
            "retrieved_chunks": chunks
        }


def tool_node(state: AgentState):
    from app.rag.retriever import search_company_docs as search_company_docs_raw
    from app.memory.long_term import save_memory
    from langchain_core.messages import ToolMessage  # ensure this import is at top
    
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    tool_messages = []
    all_chunks = []
    for tc in tool_calls:
        if tc["name"] == "search_docs":
            result_str, raw_chunks = search_company_docs_raw(tc["args"]["query"])
            tool_messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))
            all_chunks.extend(raw_chunks)
        elif tc["name"] == "store_preference":
            # Directly call save_memory from memory module
            save_memory(state["user_id"], tc["args"]["key"], tc["args"]["value"])
            result = f"Stored {tc['args']['key']} = {tc['args']['value']}"
            tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        elif tc["name"] == "refuse_request":
            result = f"REFUSED: {tc['args']['reason']}"
            tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            tool_messages.append(ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]))
    return {
        "messages": tool_messages,
        "next_action": "respond",
        "retrieved_chunks": all_chunks
    }