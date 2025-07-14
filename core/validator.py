"""To ensure the output from any pipeline is valid, non-empty, consistent with the intent, and
optionally meets confidence thresholds before the response proceeds to ethics filtering or final return."""

def is_valid_output(output,intent : str, min_confidence : float = 0.5) -> bool:
    # Validate the pipeline output based on the type of the task/intent
    
    if not output:
        return False
    
    #Text based output
    if intent in ['text-generation','summarization', 'translation', 'image-to-text', 'automatic-speech-recognition']:
        if isinstance(output, list) and isinstance(output[0], dict):
            return bool(output[0].get("generated_text") or output[0].get("summary_text") or output[0].get("text"))
        elif isinstance(output, str):
            return bool(output.strip())
        return False
    
    #Classification output
    if intent in ['text-classification', 'audio-classification', 'image-classification', 'zero-shot-classification']:
        if isinstance(output, list) and len(output) > 0:
            return all("label" in o and "score" in o for o in output) and any(o["score"] >= min_confidence for o in output)
        return False
    
    #Object detection 
    if intent == "object-detection":
        return isinstance(output, list) and len(output) > 0 and "box" in output[0]
    
    #Multimodal detection
    if intent == "image-text-to-text":
        return isinstance(output,list) and "generated_text" in output[0]
    
    #Feature extraction and embeddings
    if intent == "feature-extraction":
        return isinstance(output, list) or isintance(output , np.ndarray)
    
    #Fall back
    return True
