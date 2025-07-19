from typing import Any, Dict, Optional, Union
import logging
from pathlib import Path

from .task_router import route_task
from .pipeline_manager import run_pipeline
from .retriever import Retriever
from .fuser import Fuser
from .validator import is_valid_output
from .response_formatter import format_response
from .response_filter import filter_response
from ..utils.log_utils import log_event

logger = logging.getLogger(__name__)


def run_fuse_pipeline(
    user_input: Union[str, bytes], 
    input_type: str = "text", 
    metadata: Optional[Dict[str, Any]] = None
) -> Union[Dict[str, Any], str]:
    """
    Main pipeline orchestrator that processes input through the entire workflow.
    
    This function coordinates the following steps:
    1. Input validation and preprocessing
    2. Task routing and intent detection
    3. Content retrieval (if needed)
    4. Pipeline execution
    5. Result validation and formatting
    
    Args:
        user_input: The input text, file path, or binary data to process
        input_type: Type of input ('text', 'image', 'audio', etc.)
        metadata: Additional metadata for processing
        
    Returns:
        Processed output as a formatted string or error information
    """
    try:
        # Validate input
        if not user_input:
            raise ValueError("No input provided")
            
        log_event("orchestrator", f"Processing {input_type} input")
        
        # Step 1: Detect type and intent
        detected_type, intent = route_task(user_input)
        log_event("orchestrator", f"Detected type: {detected_type}, Intent: {intent}")

        if intent == "unknown":
            return "I couldn't determine what you're asking for. Could you please rephrase?"

        # Step 2: Retrieve content (if needed)
        retriever = Retriever()
        content = user_input
        
        if detected_type == 'text' and not str(user_input).startswith(('http://', 'https://')):
            # If it's a local file path, read its content
            if Path(user_input).is_file():
                with open(user_input, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                # Try to retrieve related content
                retrieved = retriever.retrieve_text(user_input)
                if retrieved:
                    content = retrieved[0]  # Use the first result for now
        
        # Ensure content is not empty
        if not content:
            return "I couldn't find any content to process. Please check your input and try again."

        # Step 3: Run appropriate pipeline
        log_event("orchestrator", f"Running pipeline for intent: {intent}")
        result = run_pipeline(task=intent, data=content)
        
        if not result:
            return "I couldn't generate a response. Please try again with a different input."

        # Step 4: Process the result
        fuser = Fuser()
        fused_result = fuser.fuse(task_type=intent, pipeline_output=result)
        
        if not fused_result:
            return {"error": "Failed to process the pipeline output."}

        # Step 5: Validate the output
        if not is_valid_output(fused_result, intent):
            log_event("orchestrator", "Output validation failed", level="warning")
            return "I'm having trouble generating a good response. Could you try rephrasing your request?"

        # Step 6: Format the response
        try:
            formatted_response = format_response(intent, fused_result)
            
            # Filter the response to ensure it's appropriate
            filtered_response = filter_response(
                response=formatted_response,
                user_input=user_input if isinstance(user_input, str) else str(user_input)
            )
            
            log_event("orchestrator", "Successfully generated and filtered response")
            return filtered_response
            
        except Exception as e:
            log_event("orchestrator", f"Error formatting response: {str(e)}", level="error")
            # Return a safe response even if formatting fails
            if isinstance(fused_result, str):
                return fused_result
            return str(fused_result)

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        log_event("orchestrator", error_msg, level="error")
        return "I encountered an error while processing your request. Please try again later."

