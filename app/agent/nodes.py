from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from openai import BadRequestError
from app.config import OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_DEPLOYMENT
from app.agent.state import AgentState
from app.agent.tools import search_docs, store_preference, refuse_request
from app.memory.long_term import load_memory
from app.rag.retriever import search_company_docs as search_company_docs_raw
from app.safety.injection import detect_prompt_injection, detect_sensitive_request
import uuid

# LLM initialization
llm = AzureChatOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_API_KEY,
    azure_deployment=OPENAI_DEPLOYMENT,
    model=OPENAI_DEPLOYMENT,
    api_version="2024-02-15-preview",
    temperature=0,
    max_tokens=500
)

tools = [search_docs, store_preference, refuse_request]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    last_msg = state["messages"][-1]
    
    # ========== 1. SAFETY & SENSITIVE CONTENT (HIGHEST PRIORITY) ==========
    if not isinstance(last_msg, ToolMessage) and hasattr(last_msg, "content"):
        # Check for sensitive requests (passwords, credentials, etc.)
        if hasattr(detect_sensitive_request, "__call__"):
            if detect_sensitive_request(last_msg.content):
                refusal = AIMessage(content="I cannot provide sensitive information such as passwords or credentials.")
                return {
                    "messages": [refusal],
                    "next_action": "respond",
                    "retrieved_chunks": state.get("retrieved_chunks", [])
                }
        
        # Check for prompt injection / jailbreak patterns
        if detect_prompt_injection(last_msg.content):
            refusal = AIMessage(content="I cannot process this request due to safety policy.")
            return {
                "messages": [refusal],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
    
    # ========== 2. FORCE TOOL CALL ONLY FOR FIRST SUBSTANTIVE QUERY ==========
    has_existing_tool_call = any(isinstance(m, ToolMessage) for m in state["messages"])
    if not isinstance(last_msg, ToolMessage) and hasattr(last_msg, "content") and not has_existing_tool_call:
        content_lower = last_msg.content.lower().strip()
        is_greeting = content_lower in ["hello", "hi", "hey", "good morning", "good afternoon", ""]
        is_pref = content_lower.startswith("remember") or "store preference" in content_lower
        if not is_greeting and not is_pref:
            tool_call_msg = AIMessage(content="", tool_calls=[{
                "name": "search_docs",
                "args": {"query": last_msg.content},
                "id": "forced_call_" + str(uuid.uuid4())
            }])
            return {
                "messages": [tool_call_msg],
                "next_action": "execute_tool",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
    
    # ========== 3. LOAD LONG‑TERM MEMORY PREFERENCES ==========
    user_prefs = load_memory(state["user_id"])
    pref_text = ", ".join([f"{k}: {v}" for k, v in user_prefs.items()]) if user_prefs else "None"

    system_prompt = SystemMessage(content=f"""You are a strict enterprise copilot. Your only source of information is the documents returned by the `search_docs` tool.

    **Rules for calling `search_docs`**:
    - Call `search_docs` **only once** at the beginning of the conversation, when the user asks the first substantive question.
    - Do not call `search_docs` again for follow‑up questions (e.g., "what are their numbers", "tell me more"). Instead, answer directly from the information already returned in the conversation history.

    **If you already have a ToolMessage containing document chunks in the conversation history**:
    - Use that information to answer all subsequent questions. Do not call `search_docs` again.

    **If the tool output says "No relevant documents found"**:
    - Respond exactly: "I have no knowledge for this question."

    **Never** generate code, explanations, or external information. Only state facts from the documents.

    **CRITICAL**:
    - Do not reorder, infer, or assume relationships that are not explicitly stated.
    - If the documents contain a numbered list or hierarchy, reproduce it exactly as written.
    - If you are unsure, say "I cannot find a clear hierarchy in the documents."
    User preferences: {pref_text}
    Tools available: `search_docs` (initial retrieval), `store_preference`, `refuse_request`.""")

    # ========== 4. INVOKE LLM WITH TOOLS ==========
    try:
        response = llm_with_tools.invoke([system_prompt] + state["messages"])
    except BadRequestError as e:
        error_msg = AIMessage(content="I cannot process that request due to safety policies. Please rephrase.")
        return {
            "messages": [error_msg],
            "next_action": "respond",
            "retrieved_chunks": state.get("retrieved_chunks", [])
        }

    # ========== 5. HANDLE TOOL CALLS OR DIRECT ANSWERS ==========
    if hasattr(response, "tool_calls") and response.tool_calls:
        return {
            "messages": [response],
            "next_action": "execute_tool",
            "retrieved_chunks": state.get("retrieved_chunks", [])
        }
    else:
        # Allow direct answer only if we already have a ToolMessage (i.e., retrieval already performed)
        if has_existing_tool_call:
            return {
                "messages": [response],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
        else:
            print("WARNING: No tool call and no prior tool output – blocking.")
            forced_answer = AIMessage(content="I have no knowledge for this question. Please ask about information contained in our documents.")
            return {
                "messages": [forced_answer],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }

def tool_node(state: AgentState):
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    tool_messages = []
    all_chunks = []
    for tc in tool_calls:
        if tc["name"] == "search_docs":
            result_str, raw_chunks = search_company_docs_raw(tc["args"]["query"])
            print(f"DEBUG: search returned {len(raw_chunks)} chunks, first 100 chars: {result_str[:100]}")
            tool_messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))
            all_chunks.extend(raw_chunks)
        elif tc["name"] == "store_preference":
            from app.memory.long_term import save_memory
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