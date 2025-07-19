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
        return f"{output.strip()}"
    
    elif intent == "sentiment-analysis":
        if isinstance(output, list):
            # Format sentiment analysis results
            return "\n".join([f"- {item['label'].title()}: {(item['score']*100):.1f}%" for item in output])
        return str(output)
    
    elif intent == "image-classification":
        if isinstance(output, list):
            # Format as a list of labels with confidence scores
            labels = [f"- {item['label']} ({(item['score']*100):.1f}%)" for item in output]
            return "\n".join(labels)
        return str(output)
    
    elif intent == "automatic-speech-recognition":
        return f" Transcribed Audio:\n{output.strip()}"
    
    elif intent == "question-answering":
        return f" Answer:\n{output.strip()}"
    
    elif intent == "multimodal":
        return f" Multimodal Output:\n{output.strip()}"
    
    else:
        return f" Result:\n{output.strip()}"
