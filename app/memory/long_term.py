import json
import os
from app.config import MEMORY_FILE   # now this exists

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