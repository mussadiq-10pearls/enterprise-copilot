from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agent.state import AgentState
from app.agent.nodes import agent_node, tool_node

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