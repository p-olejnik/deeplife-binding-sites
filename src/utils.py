import json

def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json(data: dict, file_path: str) -> None:
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
