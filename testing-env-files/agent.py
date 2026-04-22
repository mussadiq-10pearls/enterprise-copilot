import os
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI

from retriever_tool import search_company_docs_func

load_dotenv()

# === Long-term memory ===
MEMORY_FILE = "long_term_memory.json"

def load_memory(user_id: str) -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
    return data.get(user_id, {})

def save_memory(user_id: str, key: str, value: str):
    data = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
    if user_id not in data:
        data[user_id] = {}
    data[user_id][key] = value
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# === Tools ===
@tool
def store_preference(user_id: str, key: str, value: str) -> str:
    """Store a user preference."""
    save_memory(user_id, key, value)
    return f"Stored {key} = {value}"

@tool
def refuse_request(reason: str) -> str:
    """Refuse unsafe or off-topic requests."""
    return f"REFUSED: {reason}"

@tool
def search_docs(query: str) -> str:
    """Search internal documents for information."""
    return search_company_docs_func(query)

# === LLM (Grok via Azure serverless) ===
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2024-02-15-preview",
    temperature=0,
)

tools = [search_docs, store_preference, refuse_request]
llm_with_tools = llm.bind_tools(tools)

# === State ===
from typing import Annotated, List, TypedDict
import operator

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    next_action: str
    user_id: str

# === Nodes ===
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
            result = search_company_docs_func(tc["args"]["query"])
        elif tc["name"] == "store_preference":
            result = store_preference(state["user_id"], tc["args"]["key"], tc["args"]["value"])
        elif tc["name"] == "refuse_request":
            result = refuse_request(tc["args"]["reason"])
        else:
            result = f"Unknown tool: {tc['name']}"
        results.append(result)
    return {"messages": [AIMessage(content=str(results))], "next_action": "respond"}

# === Build graph ===
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    lambda s: s["next_action"],
    {"execute_tool": "tools", "respond": END}
)
builder.add_edge("tools", "agent")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# === Test ===
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-user-1"}}
    print("=== Agent Test ===")
    initial_state = {
        "messages": [HumanMessage(content="What do our documents say about the main topic?")],
        "next_action": "",
        "user_id": "user123"
    }
    for event in graph.stream(initial_state, config):
        print(event)
        print("----")