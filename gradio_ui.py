import gradio as gr
import requests
import uuid

API_URL = "http://localhost:8000"
sessions = {}

def chat_with_copilot(message, history):
    session_id = str(uuid.uuid4())
    thread_id = sessions.get(session_id)
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "user_id": "gradio_demo", "thread_id": thread_id},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            sessions[session_id] = data["thread_id"]
            return data["response"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Remove theme and examples if not supported
demo = gr.ChatInterface(fn=chat_with_copilot)
demo.launch(share=True)