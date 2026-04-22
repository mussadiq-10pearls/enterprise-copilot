from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from app.config import OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_DEPLOYMENT
from app.agent.state import AgentState
from app.agent.tools import search_docs, store_preference, refuse_request
from app.memory.long_term import load_memory
from app.rag.retriever import search_company_docs

# LLM initialization using direct variables
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
    user_prefs = load_memory(state["user_id"])
    pref_text = ", ".join([f"{k}: {v}" for k, v in user_prefs.items()]) if user_prefs else "None"
    system_prompt = SystemMessage(content=f"""You are an enterprise copilot. 
User preferences: {pref_text}

Tools:
- search_docs: use when the user asks about information in company documents.
- store_preference: use when the user explicitly asks you to remember something.
- refuse_request: use when the request is unsafe, off‑topic, or violates policy.

Decide which tool to call, or answer directly if no tool is needed.""")
    
    response = llm_with_tools.invoke([system_prompt] + state["messages"])
    if hasattr(response, "tool_calls") and response.tool_calls:
        return {"messages": [response], "next_action": "execute_tool"}
    else:
        return {"messages": [response], "next_action": "respond"}

def tool_node(state: AgentState):
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    results = []
    for tc in tool_calls:
        if tc["name"] == "search_docs":
            result = search_company_docs(tc["args"]["query"])
        elif tc["name"] == "store_preference":
            # store_preference expects user_id, key, value
            result = store_preference(state["user_id"], tc["args"]["key"], tc["args"]["value"])
        elif tc["name"] == "refuse_request":
            result = refuse_request(tc["args"]["reason"])
        else:
            result = f"Unknown tool: {tc['name']}"
        results.append(result)
    return {"messages": [AIMessage(content=str(results))], "next_action": "respond"}