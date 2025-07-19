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
    # Convert to lowercase for case-insensitive matching
    response_lower = response.lower()
    user_input_lower = user_input.lower()
    
    # List of phrases that indicate the response is off-topic or inappropriate
    inappropriate_phrases = [
        r"i(?:'m| am) a (?:huge|big|great) fan of",
        r"i(?:'m| am) (?:really|very) (?:excited|happy|interested) (?:about|in)",
        r"(?:i|you) (?:should|must) (?:watch|read|see|try)",
        r"(?:let(?:'s| us) (?:talk|discuss)",
        r"(?:i|you) (?:love|like|enjoy)",
        r"(?:would you like to|do you want to)",
        r"(?:i(?:'m| am) sorry,? (?:but|that)",
    ]
    
    # Check for inappropriate phrases
    for phrase in inappropriate_phrases:
        if re.search(phrase, response_lower):
            return False
    
    # Check if the response is just echoing the input
    if len(response.split()) > 5 and any(word in response_lower for word in user_input_lower.split()[:3]):
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
    
    # If the response is empty or too short, return a default message
    if len(response.split()) < 2:
        return "I'm not sure how to respond to that. Could you provide more details?"
    
    # Check if the response is appropriate
    if not is_response_appropriate(response, user_input):
        return "I'm sorry, I'm not sure how to respond to that. Could you try rephrasing your question?"
    
    # Remove any trailing incomplete sentences
    response = re.sub(r'[^.!?]+$', '', response)
    
    # Ensure the response ends with proper punctuation
    if not response.endswith(('.', '!', '?')):
        response = response.rstrip('.!?') + '.'
    
    return response
