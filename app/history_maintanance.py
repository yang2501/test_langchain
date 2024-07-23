import json
import plotly.io as pio
from plotly.graph_objs._figure import Figure
import pandas as pd

def save_messages_to_file(messages, filename="messages.json"):
    messages_to_save = []
    for message in messages:
        message_copy = message.copy()
        if message["role"] == "assistant":
            if isinstance(message["content"], Figure):
                message_copy["content"] = pio.to_json(message["content"])
            elif isinstance(message["content"], pd.DataFrame):
                message_copy["content"] = message["content"].to_json()
            elif isinstance(message["content"], list):
                new_content = []
                for item in message["content"]:
                    if isinstance(item, Figure):
                        new_content.append(pio.to_json(item))
                    elif isinstance(item, pd.DataFrame):
                        new_content.append(item.to_json())
                    else:
                        new_content.append(item)
                message_copy["content"] = new_content
        messages_to_save.append(message_copy)
    
    with open(filename, "w") as file:
        json.dump(messages_to_save, file, indent=4)

def load_messages_from_file(filename="messages.json"):
    try:
        with open(filename, "r") as file:
            messages = json.load(file)
            
        for message in messages:
            if message["role"] == "assistant":
                if isinstance(message["content"], str):
                    try:
                        message["content"] = pio.from_json(message["content"])
                    except ValueError:
                        try:
                            message["content"] = pd.read_json(message["content"])
                        except ValueError:
                            # If neither Figure nor DataFrame, keep as string
                            pass
                elif isinstance(message["content"], list):
                    new_content = []
                    for item in message["content"]:
                        try:
                            new_content.append(pio.from_json(item))
                        except ValueError:
                            try:
                                new_content.append(pd.read_json(item))
                            except ValueError:
                                new_content.append(item)
                    message["content"] = new_content
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading messages from file: {e}")
        messages = []

    return messages