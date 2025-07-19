import mimetypes
from pathlib import Path 

"""Detect input type: Text / Image / Audio / Multimodal

Classify intent: Summarization, translation, classification, etc.

Handle edge cases: empty input, unsupported formats
"""

def detect_input_type(user_input):
    if isinstance(user_input, str):
        if user_input.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            return "image"
        elif user_input.lower().endswith((".mp3", ".wav", ".flac")):
            return "audio"
        elif user_input.lower().endswith(".txt"):
            return "text"
        else:
            return "text"
    elif isinstance(user_input, dict):
        if "image" in user_input:
            return "multimodal"
        elif "audio" in user_input:
            return "audio"
        else:
            return "text"
    else:
        return "text"  # fallback


def classify_intent(user_input):
    if isinstance(user_input, str):
        lower_input = user_input.lower()
        if "summarize" in lower_input or "summary" in lower_input:
            return "summarization"
        elif "translate" in lower_input:
            return "translation"
        elif "sentiment" in lower_input or "emotion" in lower_input:
            return "sentiment-analysis"
        elif ("story" in lower_input or "generate" in lower_input or 
              "write" in lower_input or "create" in lower_input):
            return "text-generation"
        elif "classify" in lower_input or "category" in lower_input:
            return "text-classification"
    elif isinstance(user_input, dict) and "text" in user_input:
        return classify_intent(user_input["text"])

    # Default to text-generation for general queries
    return "text-generation"


def route_task(input_data):
    """
    Main router: returns input type and predicted intent.
    """
    input_type = detect_input_type(input_data)
    intent = None
    
    if input_type == "text":
        intent = classify_intent(input_data)
    elif input_type == "multimodal":
        intent = "image-text-to-text"
    elif input_type == "image":
        intent = "image-classification"
    elif input_type == "audio":
        intent = "audio-classification"
    elif input_type == "text_file":
        intent = "summarization"
    else:
        intent = "unknown"
    
    return input_type, intent
