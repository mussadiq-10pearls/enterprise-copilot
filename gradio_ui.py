import gradio as gr
import requests
import uuid

API_URL = "http://localhost:8000"
USER_ID = "gradio_demo"

# Store thread_id for the current session
thread_id = None

def chat_with_copilot(message, history):
    global thread_id
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "user_id": USER_ID, "thread_id": thread_id},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            thread_id = data["thread_id"]  # update for next turn
            return data["response"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.ChatInterface(fn=chat_with_copilot)
demo.launch(server_name="0.0.0.0", server_port=7860)