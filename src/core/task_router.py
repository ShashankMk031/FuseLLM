"""
Task Router Module

This module is responsible for:
1. Detecting the type of input (text, image, audio, etc.)
2. Classifying the intent of the input (summarization, translation, etc.)
3. Routing the input to the appropriate processing pipeline
"""
import mimetypes
import re
import logging
from pathlib import Path
from typing import Tuple, Union, Dict, Any, Optional

# Import configurations
from config import FILE_EXTENSIONS

# Configure logging
logger = logging.getLogger(__name__)

# Configure mimetypes
mimetypes.init()

# Common patterns for intent classification
INTENT_PATTERNS = {
    'summarization': [
        r'summar(?:y|ize|ise)',
        r'brief(?:ly)?',
        r'in (?:short|brief|summary)',
        r'tl;?dr',
    ],
    'translation': [
        r'translate(?: to)?',
        r'in (?:spanish|french|german|chinese|japanese|korean|hindi)',
    ],
    'sentiment_analysis': [
        r'sentiment',
        r'emotion',
        r'feel(?:ing)? (?:about|toward)',
        r'tone of',
    ],
    'text_generation': [
        r'write (?:me )?(?:a|an)',
        r'generate',
        r'create',
        r'compose',
    ],
    'text_classification': [
        r'classify',
        r'categor(?:y|ize|ise)',
        r'what (?:type|kind) of',
    ]
}


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


def _matches_patterns(text: str, intent_type: str) -> bool:
    """
    Check if the text matches any of the patterns for the given intent type.
    
    Args:
        text: The input text to check
        intent_type: The type of intent to check for
        
    Returns:
        bool: True if any pattern matches, False otherwise
    """
    if intent_type not in INTENT_PATTERNS:
        logger.warning(f"Unknown intent type: {intent_type}")
        return False
        
    patterns = INTENT_PATTERNS[intent_type]
    text = text.lower()
    return any(re.search(pattern, text) for pattern in patterns)


def classify_intent(user_input: Union[str, Dict[str, Any]]) -> str:
    """
    Classify the intent of the user input.
    
    Args:
        user_input: The input text or dict containing text to analyze.
        
    Returns:
        str: The detected intent (e.g., 'summarization', 'translation', etc.)
    """
    # Extract text if input is a dictionary
    if isinstance(user_input, dict):
        text = user_input.get('text', '')
    else:
        text = str(user_input)
    
    # Check for specific intents in order of specificity
    for intent_type in [
        'summarization',
        'translation',
        'sentiment_analysis',
        'text_classification',
        'text_generation'
    ]:
        if _matches_patterns(text, intent_type):
            # Convert to the format expected by the pipeline
            return intent_type.replace('_', '-')
    
    # Default to text generation if no specific intent is detected
    return 'text-generation'


def _detect_file_type(file_path: Union[str, Path]) -> str:
    """
    Detect the file type based on extension or content.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Detected file type ('text', 'image', 'audio', or 'unknown')
    """
    file_path = str(file_path).lower()
    
    # Check by file extension first
    for file_type, extensions in FILE_EXTENSIONS.items():
        if any(file_path.endswith(ext) for ext in extensions):
            return file_type
    
    # Fallback to mimetype detection
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type.startswith('text/'):
            return 'text'
        elif mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('audio/') or mime_type.startswith('video/'):
            return 'audio'
    
    return 'unknown'


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
        # Handle dictionary input
        if isinstance(input_data, dict):
            # If it's a file upload, detect type from file extension
            if 'file' in input_data:
                file_path = input_data['file']
                input_type = _detect_file_type(file_path)
                return input_type, classify_intent(input_data)
            # Otherwise, treat as text with metadata
            return 'text', classify_intent(input_data)
        
        # Handle file paths
        if isinstance(input_data, (str, Path)) and Path(input_data).exists():
            input_type = _detect_file_type(input_data)
            return input_type, classify_intent(str(input_data))
        
        # Handle binary data with type hint
        if isinstance(input_data, bytes):
            # Try to determine type from magic numbers
            if input_data.startswith(b'\x89PNG') or input_data.startswith(b'\xff\xd8\xff'):
                return 'image', 'image-classification'
            elif input_data.startswith(b'RIFF') and len(input_data) > 8 and input_data[8:12] == b'WAVE':
                return 'audio', 'automatic-speech-recognition'
        
        # Default to text processing
        return 'text', classify_intent(str(input_data))
        
    except Exception as e:
        logger.error(f"Error in route_task: {str(e)}")
        # Fallback to text processing
        return 'text', 'text-generation'
