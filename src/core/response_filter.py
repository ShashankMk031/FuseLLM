"""
Response filtering module to ensure appropriate and on-topic responses.
"""
import re
from typing import List, Optional

def is_response_appropriate(response: str, user_input: str) -> bool:
    """
    Check if a response is appropriate and on-topic.
    
    Args:
        response: The generated response to check
        user_input: The original user input for context
        
    Returns:
        bool: True if the response is appropriate, False otherwise
    """
    if not response or not isinstance(response, str):
        return False
        
    # Convert to lowercase for case-insensitive matching
    response_lower = response.lower().strip()
    user_input_lower = user_input.lower().strip()
    
    # Check for empty or very short responses
    if len(response_lower) < 3:
        return False
    
    # Check for common error messages
    error_indicators = [
        'error', 'sorry', 'apologize', 'cannot', 'unable', 'not sure', 
        'as an ai', 'language model', 'i am an ai'
    ]
    
    if any(indicator in response_lower for indicator in error_indicators):
        return False
    
    # Check for URLs or email addresses
    if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', response):
        return False
        
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', response):
        return False
    
    # For very short user inputs (like "hi"), be more lenient
    user_input_words = [w for w in user_input_lower.split() if len(w) > 2]  # Only consider words longer than 2 chars
    
    if len(user_input_words) <= 2:
        # Only check for obvious errors
        return len(response_lower.split()) > 1  # At least 2 words in response
    
    # List of phrases that indicate the response is off-topic or inappropriate
    inappropriate_phrases = [
        r"i(?:'m| am) a (?:huge|big|great) fan of",
        r"i(?:'m| am) (?:really|very) (?:excited|happy|interested) (?:about|in)",
        r"(?:i|you) (?:should|must) (?:watch|read|see|try)",
        r"(?:let(?:'s| us) (?:talk|discuss))",
        r"(?:i|you) (?:love|like|enjoy)",
        r"(?:would you like to|do you want to)",
        r"(?:i(?:'m| am) sorry,? (?:but|that))",
        r"as an (?:ai|artificial intelligence) (?:language )?model",
        r"i (?:can't|cannot) (?:answer|respond|help with that)",
        r"i don'?t (?:know|understand|have that information)",
        r"i(?:'m| am) not (?:able to|capable of|designed to)",
        r"my purpose is to",
        r"i was created to"
    ]
    
    # Check for inappropriate phrases
    for phrase in inappropriate_phrases:
        if re.search(phrase, response_lower, re.IGNORECASE):
            return False
    
    # Skip echo check for very short responses
    if len(response.split()) > 5:
        # Check if the response is just echoing the input
        input_words = set(word.lower() for word in user_input_words if len(word) > 3)  # Only check longer words
        response_words = set(word.lower() for word in response.split() if len(word) > 3)
        
        # If more than 2 significant words from input appear in response, it might be echoing
        common_words = input_words.intersection(response_words)
        if len(common_words) >= min(3, len(input_words)):
            return False
    
    return True

def filter_response(response: str, user_input: str, max_retries: int = 3) -> str:
    """
    Filter and clean up a response to ensure it's appropriate.
    
    Args:
        response: The generated response to filter
        user_input: The original user input for context
        max_retries: Maximum number of times to retry if response is inappropriate
        
    Returns:
        str: A filtered and cleaned response
    """
    if not response or not isinstance(response, str):
        return "I'm sorry, I couldn't generate a proper response."
    
    # Clean up the response
    response = response.strip()
    
    # Handle empty response
    if not response:
        return "I'm not sure how to respond to that. Could you try rephrasing?"
    
    # For very short user inputs (like "hi"), be less strict with the response
    user_input_words = user_input.strip().split()
    if len(user_input_words) <= 2:
        # Just ensure the response is not empty and doesn't contain errors
        if not response or "error" in response.lower():
            return "Hello! How can I assist you today?"
        return response
    
    # For longer inputs, apply the full filtering
    # Check if the response is appropriate
    if not is_response_appropriate(response, user_input):
        # Instead of returning an error, try to clean up the response
        # Remove any error messages or technical details
        response = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|<.*?>', '', response)
        response = response.strip()
        
        # If we still have a reasonable response, return it
        if len(response.split()) >= 2:
            return response
            
        return "I'm not sure how to respond to that. Could you provide more details?"
    
    # Remove any trailing incomplete sentences
    response = re.sub(r'[^.!?]+$', '', response)
    
    # Ensure the response ends with proper punctuation
    if not response.endswith(('.', '!', '?')):
        response = response.rstrip('.!?') + '.'
    
    return response
