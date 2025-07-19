"""
Task Router Module

This module is responsible for:
1. Detecting the type of input (text, image, audio, etc.)
2. Classifying the intent of the input (summarization, translation, etc.)
3. Routing the input to the appropriate processing pipeline
"""
import mimetypes
import re
from pathlib import Path
from typing import Tuple, Union, Dict, Any, Optional

# Configure mimetypes
mimetypes.init()

# Common patterns for intent classification
SUMMARIZE_PATTERNS = [
    r'summar(?:y|ize|ise)',
    r'brief(?:ly)?',
    r'in (?:short|brief|summary)',
    r'tl;?dr',
]

TRANSLATE_PATTERNS = [
    r'translate(?: to)?',
    r'in (?:spanish|french|german|chinese|japanese|korean|hindi)',
]

SENTIMENT_PATTERNS = [
    r'sentiment',
    r'emotion',
    r'feel(?:ing)? (?:about|toward)',
    r'tone of',
]

GENERATION_PATTERNS = [
    r'write (?:me )?(?:a|an)',
    r'generate',
    r'create',
    r'compose',
]

CLASSIFICATION_PATTERNS = [
    r'classify',
    r'categor(?:y|ize|ise)',
    r'what (?:type|kind) of',
]


def detect_input_type(user_input: Union[str, bytes, Dict[str, Any], Path]) -> str:
    """
    Detect the type of input based on its content or file extension.
    
    Args:
        user_input: The input to analyze. Can be a string, bytes, dict, or Path.
        
    Returns:
        str: The detected input type ('text', 'image', 'audio', 'video', 'multimodal')
    """
    if isinstance(user_input, (str, Path)):
        input_str = str(user_input).lower()
        
        # Check for URLs
        if input_str.startswith(('http://', 'https://')):
            if any(ext in input_str for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                return 'image'
            elif any(ext in input_str for ext in ['.mp3', '.wav', '.ogg', '.flac']):
                return 'audio'
            elif any(ext in input_str for ext in ['.mp4', '.webm', '.mov', '.avi']):
                return 'video'
            return 'text'  # Default for URLs
            
        # Check file extensions
        if Path(input_str).suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            return 'image'
        elif Path(input_str).suffix.lower() in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
            return 'audio'
        elif Path(input_str).suffix.lower() in ['.mp4', '.webm', '.mov', '.avi', '.mkv']:
            return 'video'
        
        # Check MIME type for binary data
        if isinstance(user_input, bytes):
            import magic
            mime = magic.Magic(mime=True)
            file_type = mime.from_buffer(user_input[:1024])
            if file_type.startswith('image/'):
                return 'image'
            elif file_type.startswith('audio/'):
                return 'audio'
            elif file_type.startswith('video/'):
                return 'video'
    
    elif isinstance(user_input, dict):
        # Check for multimodal inputs
        if 'image' in user_input and 'text' in user_input:
            return 'multimodal'
        elif 'image' in user_input:
            return 'image'
        elif 'audio' in user_input:
            return 'audio'
    
    # Default to text
    return 'text'


def _matches_patterns(text: str, patterns: list) -> bool:
    """Check if the text matches any of the given patterns."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def classify_intent(user_input: Union[str, Dict[str, Any]]) -> str:
    """
    Classify the intent of the user input.
    
    Args:
        user_input: The input text or dict containing text to analyze.
        
    Returns:
        str: The detected intent (e.g., 'summarization', 'translation', etc.)
    """
    # Extract text from dict if needed
    if isinstance(user_input, dict) and 'text' in user_input:
        text = user_input['text']
    elif isinstance(user_input, str):
        text = user_input
    else:
        return 'text-generation'  # Default
    
    text = text.lower().strip()
    
    # Check for empty input
    if not text:
        return 'unknown'
    
    # Check for specific intents using patterns
    if _matches_patterns(text, SUMMARIZE_PATTERNS):
        return 'summarization'
    elif _matches_patterns(text, TRANSLATE_PATTERNS):
        return 'translation'
    elif _matches_patterns(text, SENTIMENT_PATTERNS):
        return 'sentiment-analysis'
    elif _matches_patterns(text, GENERATION_PATTERNS):
        return 'text-generation'
    elif _matches_patterns(text, CLASSIFICATION_PATTERNS):
        return 'text-classification'
    
    # Check for question-answering
    if any(text.startswith(q) for q in ['what', 'when', 'where', 'why', 'how', 'who', 'which']):
        if '?' in text or ' ' in text:  # Simple heuristic for questions
            return 'question-answering'
    
    # Default to text generation
    return 'text-generation'


def route_task(input_data: Any) -> Tuple[str, str]:
    """
    Route the input to the appropriate processing pipeline.
    
    This function:
    1. Detects the input type (text, image, audio, etc.)
    2. Classifies the intent (summarization, translation, etc.)
    3. Returns both the input type and intent
    
    Args:
        input_data: The input to analyze. Can be text, file path, binary data, or dict.
        
    Returns:
        tuple: (input_type, intent)
    """
    try:
        # Detect input type
        input_type = detect_input_type(input_data)
        
        # Classify intent
        intent = classify_intent(input_data)
        
        # Log the routing decision
        from ..utils.log_utils import log_event
        log_event(
            "task_router",
            f"Routed input: type={input_type}, intent={intent}",
            level="debug"
        )
        
        return input_type, intent
        
    except Exception as e:
        # Log the error and return defaults
        from ..utils.log_utils import log_event
        log_event(
            "task_router",
            f"Error in route_task: {str(e)}",
            level="error"
        )
        return 'text', 'text-generation'  # Default fallback
    else:
        intent = "unknown"
    
    return input_type, intent
