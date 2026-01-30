import os
import json

def get_todo_path(conversation_id: str) -> str:
    # Use the current directory or a specific data directory
    return f"{conversation_id}_todo.json"

def save_todo(conversation_id: str, plan: list[str]) -> None:
    path = get_todo_path(conversation_id)
    with open(path, "w") as f:
        json.dump({"plan": plan}, f, indent=2)

def load_todo(conversation_id: str) -> list[str]:
    path = get_todo_path(conversation_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return data.get("plan", [])
        except json.JSONDecodeError:
            return []
    return []
