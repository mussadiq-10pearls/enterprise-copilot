from fastapi import APIRouter, HTTPException
from app.api.models import ChatRequest, ChatResponse
from app.agent.graph import graph
from langchain_core.messages import HumanMessage
import uuid

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    state = {
        "messages": [HumanMessage(content=request.message)],
        "next_action": "",
        "user_id": request.user_id
    }
    
    final_state = None
    for event in graph.stream(state, config):
        final_state = event
    
    try:
        answer = final_state["agent"]["messages"][-1].content
    except:
        answer = "I couldn't process your request."
    
    return ChatResponse(response=answer, thread_id=thread_id)

@router.get("/health")
async def health():
    return {"status": "ok"}