import os
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import pipeline manager and NLP processor
from .pipeline_manager import run_pipeline
from .response_filter import filter_response
from ..nlp.processor import nlp_processor

# Import configuration
from config import MODEL_CONFIG, DEFAULT_PARAMS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_event(component: str, message: str, level: str = "info"):
    """Log an event with a consistent format."""
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(f"[{component.upper()}] {message}")

def format_response(intent: str, result: Any) -> str:
    """Format the response based on the intent."""
    if isinstance(result, str):
        return result.strip()
    elif isinstance(result, dict):
        if 'generated_text' in result:
            return result['generated_text'].strip()
        elif 'text' in result:
            return result['text'].strip()
    elif isinstance(result, list) and result:
        return format_response(intent, result[0])
    return str(result).strip()

def detect_intent(text: str) -> Dict[str, Any]:
    """
    Detect the intent of the input text using the NLP processor.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing intent information
    """
    if not text.strip():
        return {
            'intent': 'unknown',
            'score': 0.0,
            'description': 'No input provided'
        }
    
    # Get the primary intent
    intent = nlp_processor.get_primary_intent(text)
    
    if not intent:
        return {
            'intent': 'general',
            'score': 0.0,
            'description': 'General conversation'
        }
    
    return intent

def run_fuse_pipeline(
    user_input: Union[str, bytes], 
    input_type: str = "text",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main pipeline orchestrator that processes input through the entire workflow.
    
    This function coordinates the following steps:
    1. Input validation and preprocessing
    2. Intent detection using NLP
    3. Task routing based on intent
    4. Pipeline execution
    5. Result validation and formatting
    
    Args:
        user_input: The input text, file path, or binary data to process
        input_type: Type of input ('text', 'image', 'audio', etc.)
        metadata: Optional metadata about the input
        
    Returns:
        Processed output as a dictionary with response and metadata
    """
    try:
        # Initialize logging
        log_event("orchestrator", f"Processing {input_type} input")
        
        # Convert input to string if it's bytes
        if isinstance(user_input, bytes):
            try:
                user_input = user_input.decode('utf-8')
            except UnicodeDecodeError:
                # For binary data that's not text (images, audio)
                pass
        
        # Convert to string if not already
        input_text = str(user_input).strip() if user_input else ""
        
        # Determine the appropriate model and task based on input type and intent
        if input_type == "image":
            intent = "image-classification"
            model_name = MODEL_CONFIG.get('image-classification')
            task_params = DEFAULT_PARAMS.get('image-classification', {})
        elif input_type == "audio":
            intent = "automatic-speech-recognition"
            model_name = MODEL_CONFIG.get('automatic-speech-recognition')
            task_params = DEFAULT_PARAMS.get('automatic-speech-recognition', {})
        else:
            # For text, use the detected intent
            intent_info = detect_intent(input_text)
            intent = intent_info['intent']
            model_name = MODEL_CONFIG.get(intent, MODEL_CONFIG['text-generation'])
            task_params = DEFAULT_PARAMS.get(intent, DEFAULT_PARAMS['text-generation']).copy()
            
            # Handle special intents with more specific prompts
            if intent == 'greeting':
                return {"response": "Hello! How can I assist you today?", "intent": intent, "model": model_name, "confidence": intent_info.get('score', 1.0)}
            elif intent == 'joke':
                user_input = f"Tell me a funny, family-friendly joke about {input_text}" if input_text else "Tell me a funny, family-friendly joke"
            elif intent == 'weather':
                user_input = f"What's the weather like {input_text}?" if input_text else "What's the weather like today?"
            elif intent == 'definition':
                user_input = f"Provide a clear and concise definition of {input_text}"
            elif intent == 'science_question':
                user_input = f"Explain {input_text} in simple terms with accurate scientific information"
            elif intent == 'general_knowledge':
                user_input = f"Answer this question accurately and concisely: {input_text}"
        
        # Determine the task type based on input type
        if input_type in ["image", "audio"]:
            task_type = intent
        else:
            task_type = 'text-generation'  # Default task type for text
            
        # Get default parameters for the task
        task_params = DEFAULT_PARAMS.get(task_type, {}).copy()
        
        # For text generation, ensure we have all required parameters
        if task_type == 'text-generation':
            # Ensure we have all required parameters with defaults
            default_params = {
                'max_length': 100,
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.9,
                'repetition_penalty': 1.2,
                'do_sample': True,
                'num_return_sequences': 1
            }
            # Update with any default params not already set
            for k, v in default_params.items():
                if k not in task_params:
                    task_params[k] = v
        
        # Update with any metadata parameters
        if metadata and isinstance(metadata, dict):
            task_params.update(metadata)
        
        log_event("orchestrator", f"Using intent: {intent} with model: {model_name}")
        log_event("orchestrator", f"Task params: {task_params}")
        
        # Log the task type and parameters
        log_event("orchestrator", f"Running task: {task_type} with intent: {intent}")
        log_event("orchestrator", f"Using model: {model_name}")
        log_event("orchestrator", f"Input data: {user_input}")
        log_event("orchestrator", f"Task params: {task_params}")
        
        try:
            # Run the pipeline with the determined task type
            result = run_pipeline(
                task=task_type,
                model_name=model_name,
                input_data=user_input,
                **task_params
            )
            
            log_event("orchestrator", f"Pipeline result type: {type(result)}")
            
            # Check for error in the result
            if isinstance(result, dict) and 'error' in result:
                error_msg = f"Pipeline error: {result['error']}"
                log_event("orchestrator", error_msg, level="error")
                return {
                    "response": f"I encountered an error: {result['error']}",
                    "intent": intent,
                    "model": model_name,
                    "error": result['error'],
                    "success": False
                }
                
            # If we get here, the generation was successful
            return {
                "response": str(result) if not isinstance(result, str) else result,
                "intent": intent,
                "model": model_name,
                "success": True
            }
                
        except Exception as e:
            error_msg = f"Unexpected error in text generation: {str(e)}"
            log_event("orchestrator", error_msg, level="error")
            log_event("orchestrator", traceback.format_exc(), level="debug")
            return {
                "response": "I encountered an unexpected error while generating a response.",
                "intent": intent,
                "model": model_name,
                "error": str(e),
                "success": False
            }
        
        # This code is unreachable due to the return statements in the try/except blocks above
        
    except Exception as e:
        error_msg = f"Error in pipeline execution: {str(e)}"
        log_event("orchestrator", error_msg, level="error")
        return {
            "error": "I encountered an error while processing your request.",
            "details": str(e)
        }

# Ensure model cache directory exists
os.makedirs("model_cache", exist_ok=True)
