# core/response_formatter.py

def format_response(intent: str, output):
    """
    Formats the output based on the intent of the task.
    """
    if intent == "summarization":
        return f"Summary:\n{output.strip()}"
    
    elif intent == "translation":
        return f"Translated Text:\n{output.strip()}"
    
    elif intent == "text-generation":
        return f" Here's a story for you:\n{output.strip()}"
    
    elif intent == "sentiment-analysis":
        return f"Sentiment Analysis Result:\n{output}"
    
    elif intent == "image-classification":
        return f" Image Classification Labels:\n{output}"
    
    elif intent == "automatic-speech-recognition":
        return f" Transcribed Audio:\n{output.strip()}"
    
    elif intent == "question-answering":
        return f" Answer:\n{output.strip()}"
    
    elif intent == "multimodal":
        return f" Multimodal Output:\n{output.strip()}"
    
    else:
        return f" Result:\n{output.strip()}"
