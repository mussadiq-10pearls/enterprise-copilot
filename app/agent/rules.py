from langchain_core.messages import AIMessage, ToolMessage
import uuid

# ------------------------------------------------------------------
# Rule helpers (all logic moved here)
# ------------------------------------------------------------------
def handle_safety(state, last_msg, user_input):
    """Check sensitive keywords and prompt injection."""
    from app.safety.injection import detect_prompt_injection, detect_sensitive_request
    if not isinstance(last_msg, ToolMessage) and user_input:
        if detect_sensitive_request(user_input):
            return {
                "messages": [AIMessage(content="I cannot provide sensitive information such as passwords or credentials.")],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
        if detect_prompt_injection(user_input):
            return {
                "messages": [AIMessage(content="I cannot process this request due to safety policy.")],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
    return None

def handle_greeting(state, last_msg, user_input):
    """Respond to simple greetings."""
    if not isinstance(last_msg, ToolMessage) and user_input:
        clean = user_input.lower().strip().rstrip('?.,!')
        if clean in ["hi", "hello", "hey", "greetings", "howdy"]:
            return {
                "messages": [AIMessage(content="Hello. How can I help?")],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
    return None

def handle_why(state, last_msg, user_input):
    """Handle 'why?' after refusal, creative block, or greeting."""
    if not isinstance(last_msg, ToolMessage) and user_input:
        clean = user_input.lower().strip().rstrip('?.,!')
        if clean == "why":
            for m in reversed(state["messages"][:-1]):
                if isinstance(m, AIMessage):
                    content_lower = m.content.lower()
                    # Check for any refusal or "no knowledge" message
                    if ("cannot process" in content_lower or 
                        "cannot provide creative" in content_lower or
                        "no knowledge" in content_lower or
                        "cannot provide sensitive" in content_lower):
                        return {
                            "messages": [AIMessage(content="Because the request violates our policies or the information is not in the documents.")],
                            "next_action": "respond",
                            "retrieved_chunks": state.get("retrieved_chunks", [])
                        }
                    # If previous assistant message was a greeting
                    elif any(g in content_lower for g in ["hello", "how can i help"]):
                        return {
                            "messages": [AIMessage(content="I am designed to answer questions based on my knowledge only.")],
                            "next_action": "respond",
                            "retrieved_chunks": state.get("retrieved_chunks", [])
                        }
                    break
    return None

def handle_creative(state, last_msg, user_input):
    """Block creative requests (poems, songs, etc.)."""
    if not isinstance(last_msg, ToolMessage) and user_input:
        clean = user_input.lower().strip()
        creative_words = ["poem", "song", "joke", "rap", "rhyme", "creative", "story"]
        if any(word in clean for word in creative_words):
            return {
                "messages": [AIMessage(content="I cannot provide creative content. Please ask a factual question based on my knowledge")],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
    return None

def handle_short_ambiguous(state, last_msg, user_input):
    """Respond to very short or ambiguous inputs."""
    if not isinstance(last_msg, ToolMessage) and user_input:
        stripped = user_input.strip()
        if len(stripped) <= 2 or stripped in ["?", "??", "."]:
            return {
                "messages": [AIMessage(content="Please ask a complete question.")],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
    return None

def force_tool_call(state, last_msg, user_input):
    """Force a search_docs tool call for substantive document queries."""
    has_existing_tool_call = any(isinstance(m, ToolMessage) for m in state["messages"])
    if not isinstance(last_msg, ToolMessage) and user_input and not has_existing_tool_call:
        content_lower = user_input.lower()
        is_question = user_input.strip().endswith("?") or any(q in content_lower for q in ["what", "how", "why", "when", "where", "who", "list", "tell me", "explain"])
        has_doc_keyword = any(k in content_lower for k in ["policy", "document", "wfo", "hospital", "workation", "leave", "benefit", "hr"])
        is_pref = content_lower.startswith("remember") or "store preference" in content_lower
        if (is_question or has_doc_keyword) and not is_pref:
            tool_call_msg = AIMessage(content="", tool_calls=[{
                "name": "search_docs",
                "args": {"query": user_input},
                "id": "forced_call_" + str(uuid.uuid4())
            }])
            return {
                "messages": [tool_call_msg],
                "next_action": "execute_tool",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
    return None

def handle_casual_greeting_followup(state, last_msg, user_input):
    """Respond to 'how are you', 'how are you doing', 'what's up' after a greeting."""
    if not isinstance(last_msg, ToolMessage) and user_input:
        clean = user_input.lower().strip().rstrip('?.,!')
        casual_greetings = ["how are you", "how are you doing", "how's it going", "what's up", "how do you do"]
        if clean in casual_greetings:
            # Check if previous assistant message contained a greeting
            for m in reversed(state["messages"][:-1]):
                if isinstance(m, AIMessage) and any(g in m.content.lower() for g in ["hello", "how can i help"]):
                    return {
                        "messages": [AIMessage(content="I'm functioning well. How can I assist with company documents?")],
                        "next_action": "respond",
                        "retrieved_chunks": state.get("retrieved_chunks", [])
                    }
            # Fallback if no previous greeting
            return {
                "messages": [AIMessage(content="I'm functioning well. How can I assist with company documents?")],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
    return None

def handle_casual_followup(state, last_msg, user_input):
    """Respond to casual follow-ups like 'you tell me' after a greeting."""
    if not isinstance(last_msg, ToolMessage) and user_input:
        clean = user_input.lower().strip().rstrip('?.,!')
        casual_phrases = ["you tell me", "tell me", "what can you do", "what do you do", "tell me something", "so what"]
        if clean in casual_phrases:
            # Check if previous assistant message contained a greeting
            for m in reversed(state["messages"][:-1]):
                if isinstance(m, AIMessage) and any(g in m.content.lower() for g in ["hello", "how can i help"]):
                    return {
                        "messages": [AIMessage(content="I can answer questions about company documents. What would you like to know?")],
                        "next_action": "respond",
                        "retrieved_chunks": state.get("retrieved_chunks", [])
                    }
            # If no greeting found, still respond casually (fallback)
            return {
                "messages": [AIMessage(content="I can answer questions about company documents. What would you like to know?")],
                "next_action": "respond",
                "retrieved_chunks": state.get("retrieved_chunks", [])
            }
    return None