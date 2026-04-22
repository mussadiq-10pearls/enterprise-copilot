import json
import os
from app.config import MEMORY_FILE

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def load_memory(user_id: str) -> dict:
    ensure_dir(MEMORY_FILE)
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
    return data.get(user_id, {})

def save_memory(user_id: str, key: str, value: str):
    ensure_dir(MEMORY_FILE)
    data = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
    if user_id not in data:
        data[user_id] = {}
    data[user_id][key] = value
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)