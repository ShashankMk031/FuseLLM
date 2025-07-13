import mimetypes
from pathlib import Path 

"""Detect input type: Text / Image / Audio / Multimodal

Classify intent: Summarization, translation, classification, etc.

Handle edge cases: empty input, unsupported formats
"""

def detect_input_type(input_data):
    """
    Determines input type: text, image, audio, or multimodal.
    """
    if isinstance(input_data, str):
        if Path(input_data).exists():
            mime, _ = mimetypes.guess_type(input_data)
            if mime:
                if "image" in mime:
                    return "image"
                elif "audio" in mime:
                    return "audio"
                elif "text" in mime or input_data.endswith(('.txt', '.md')):
                    return "text_file"
            return "unknown_file"
        else:
            return "text"
    
    elif isinstance(input_data, dict):
        if "text" in input_data and "image" in input_data:
            return "multimodal"
    
    return "unknown"

def classify_intent(text):
    """
    Very basic rule-based intent classifier.
    """
    text = text.lower()
    
    if any(keyword in text for keyword in ["summarize", "tl;dr", "make it short"]):
        return "summarization"
    elif any(keyword in text for keyword in ["translate", "in french", "to spanish"]):
        return "translation"
    elif any(keyword in text for keyword in ["classify", "label", "sentiment"]):
        return "text-classification"
    elif any(keyword in text for keyword in ["generate", "story", "complete", "write"]):
        return "text-generation"
    elif any(keyword in text for keyword in ["detect objects", "bounding box", "find objects"]):
        return "object-detection"
    elif any(keyword in text for keyword in ["speech to text", "what is he saying"]):
        return "automatic-speech-recognition"
    elif any(keyword in text for keyword in ["text to speech", "read this aloud"]):
        return "text-to-speech"
    elif any(keyword in text for keyword in ["explain", "meaning", "definition"]):
        return "zero-shot-classification"
    
    return "unknown"

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
