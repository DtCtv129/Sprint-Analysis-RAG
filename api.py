from fastapi import FastAPI, UploadFile, File
from rag import ChatPDF
from typing import List
import tempfile
import os
import glob
import json
from fastapi import Body



class AssistantState:
    def __init__(self):
        self.assistant = ChatPDF()
        self.messages = []

assistant_state = AssistantState()

# Ingest files from the "docs" directory at the start of the app
docs_dir = "docs"
for file_path in glob.glob(os.path.join(docs_dir, '*')):
    assistant_state.assistant.ingest(file_path)
app = FastAPI()

@app.post("/generate/")
def send_message(sprint_data = Body(..., embed=True)):
    
    user_text = str(format_sprint_data(sprint_data))
    if user_text and len(user_text.strip()) > 0:
        user_text = user_text.strip()
        agent_text = assistant_state.assistant.ask(user_text)

        assistant_state.messages.append((user_text, True))
        assistant_state.messages.append((agent_text, False))
        agent_text=toJson(agent_text)
    return {"status": "message processed","report": agent_text}
def toJson(text):
    lines = text.split("\n")

    # Initialize an empty dictionary to hold the structured data
    data = {}

    # Initialize an empty string to hold the current key
    current_key = ""

    # Iterate over the lines
    for line in lines:
        # If the line starts with '**', it's a key
        if line.startswith("**"):
            current_key = line.strip("*: ")
            data[current_key] = {}
        # If the line starts with '-', it's a value
        elif line.startswith("-"):
            # Check if the line contains a colon
            if ": " in line:
                # Split the line into a key and a value
                key, value = line.strip("- ").split(": ")
                # Add the key-value pair to the current dictionary
                data[current_key][key] = value
            else:
                # If the line does not contain a colon, treat the whole line as a value
                if "Feedback" in data[current_key]:
                    data[current_key]["Feedback"] += " " + line.strip("- ")
                else:
                    data[current_key]["Feedback"] = line.strip("- ")

    # Convert the dictionary to a JSON object
    json_object = json.dumps(data, indent=4)
    return data
def format_sprint_data(sprint_data):
    # Extract values from the JSON object
    stride_length = sprint_data["metrics"]["stride_length"]
    stride_frequency = sprint_data["metrics"]["stride_frequency"]
    acceleration = sprint_data["metrics"]["acceleration"]
    velocity = sprint_data["metrics"]["velocity"]
    touchdown = sprint_data["metrics"]["touchdown"]
    gender = sprint_data["sprinter"]["gender"]
    phase = sprint_data["sprinter"]["phase"]

    # Capitalize the gender and phase correctly
    gender = gender.capitalize()
    phase = phase.capitalize()

    # Create the formatted string
    formatted_text = (
        f"Stride Length: {stride_length} | "
        f"Stride Frequency: {stride_frequency} | "
        f"Acceleration: {acceleration} | "
        f"Velocity: {velocity} | "
        f"Touchdown: {touchdown}, "
        f"The sprinter is a {gender} and he is in the {phase}"
    )

    return formatted_text