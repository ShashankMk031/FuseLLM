# core/response_formatter.py

import re
from typing import Any, Dict, List, Union

def escape_special_chars(text: str) -> str:
    """Escape special regex characters in a string."""
    if not isinstance(text, str):
        return str(text)
    return re.sub(r'([\[\](){}*+?\\|^$])', r'\\\1', text)

def format_response(intent: str, output: Any) -> str:
    """
    Formats the output based on the intent of the task.
    
    Args:
        intent: The task intent (e.g., 'text-generation', 'summarization')
        output: The raw output from the pipeline
        
    Returns:
        str: Formatted response as a string
    """
    try:
        # Handle None or empty output
        if output is None:
            return "I couldn't generate a response. Please try again."
            
        # Get safe text representation first
        safe_output = _safe_get_text(output)
        
        # Handle different intents
        if intent == "summarization":
            return f"Here's a summary:\n\n{safe_output}"
        
        elif intent == "translation":
            return f"Translated Text:\n\n{safe_output}"
        
        elif intent == "text_generation":
            # Clean up the generated text
            text = safe_output.strip()
            # Remove any trailing incomplete sentences
            text = re.sub(r'[.!?][^.!?]*$', '', text)
            # Add a period if the text doesn't end with punctuation
            if text and text[-1] not in '.!?':
                text += '.'
            return text
        
        elif intent == "sentiment_analysis":
            if isinstance(output, list):
                return "Analysis Results:\n" + "\n".join([
                    f"- {escape_special_chars(_safe_get(item, 'label', 'Unknown')).title()}: "
                    f"{_safe_get(item, 'score', 0) * 100:.1f}%"
                    for item in output if isinstance(item, dict)
                ])
            return safe_output
        
        elif intent in ["text_classification", "zero_shot_classification"]:
            if isinstance(output, list):
                return "Classification Results:\n" + "\n".join([
                    f"- {escape_special_chars(_safe_get(item, 'label', 'Unknown'))}: "
                    f"{_safe_get(item, 'score', 0) * 100:.1f}%"
                    for item in output if isinstance(item, dict)
                ][:5])  # Limit to top 5 results
            return safe_output
        
        elif intent == "question_answering":
            if isinstance(output, dict):
                answer = _safe_get(output, 'answer', 'No answer found')
                score = _safe_get(output, 'score', 0)
                confidence = f" (Confidence: {score:.1%})" if score > 0 else ""
                return f"Answer: {answer}{confidence}"
            return safe_output
        
        elif intent == "image_classification":
            if isinstance(output, list):
                return "Image Analysis Results:\n" + "\n".join([
                    f"- {escape_special_chars(_safe_get(item, 'label', 'Unknown'))}: "
                    f"{_safe_get(item, 'score', 0) * 100:.1f}%"
                    for item in output if isinstance(item, dict)
                ][:3])  # Top 3 results
            return safe_output
            
        elif intent == "speech_recognition":
            return f"Transcribed Speech:\n\n{safe_output}"
        
        # Default case - return string representation with some cleaning
        if not safe_output.strip():
            return "I didn't receive a valid response. Could you try rephrasing your request?"
            
        return safe_output.strip()
        
    except Exception as e:
        # Log the error and return a user-friendly message
        import logging
        logging.error(f"Error formatting {intent} response: {str(e)}\nRaw output: {str(output)[:200]}...")
        return "I encountered an error while processing the response. Please try again."

def _safe_get_text(output: Any) -> str:
    """
    Safely extract text from various output formats.
    
    Handles:
    - Strings, numbers, booleans
    - Dictionaries with common text fields
    - Lists of strings or dictionaries
    - Nested structures
    """
    if output is None:
        return ""
        
    # Handle string types
    if isinstance(output, str):
        return output.strip()
        
    # Handle numeric and boolean types
    if isinstance(output, (int, float, bool)):
        return str(output)
    
    # Handle dictionaries
    if isinstance(output, dict):
        # Try common text fields first
        text_fields = [
            'text', 'generated_text', 'answer', 'translation', 
            'summary', 'content', 'output', 'result'
        ]
        
        for field in text_fields:
            if field in output and output[field] is not None:
                text = output[field]
                if isinstance(text, str) and text.strip():
                    return text.strip()
                
        # Check for common nested structures
        if 'choices' in output and isinstance(output['choices'], list):
            texts = []
            for choice in output['choices']:
                if isinstance(choice, dict):
                    text = _safe_get_text(choice.get('text') or choice.get('message'))
                    if text:
                        texts.append(text)
            if texts:
                return "\n\n".join(texts)
                
        # If no specific field found, try to stringify the whole dict
        try:
            # For simple dictionaries, create a clean string representation
            if all(isinstance(v, (str, int, float, bool)) for v in output.values()):
                items = [f"{k}: {v}" for k, v in output.items() if v is not None]
                if items:
                    return ", ".join(items)
            return str(output)
        except Exception:
            return ""
    
    # Handle lists and tuples
    if isinstance(output, (list, tuple)):
        # Empty list
        if not output:
            return ""
            
        # List of strings
        if all(isinstance(x, str) for x in output):
            return "\n".join(x.strip() for x in output if x.strip())
            
        # List of dictionaries
        if all(isinstance(x, dict) for x in output):
            texts = []
            for item in output:
                text = _safe_get_text(item)
                if text:
                    texts.append(text)
            return "\n\n".join(texts) if texts else ""
            
        # Mixed list, try to stringify each element
        try:
            return "\n".join(str(x) for x in output if x is not None)
        except Exception:
            pass
    
    # For any other type, try to convert to string
    try:
        text = str(output)
        return text if text.strip() else ""
    except Exception:
        return ""

def _safe_get(item: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary or return default."""
    try:
        if isinstance(item, dict):
            return item.get(key, default)
        return default
    except Exception:
        return default
