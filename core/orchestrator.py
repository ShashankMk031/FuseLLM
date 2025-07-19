from typing import Dict, Any, Union, List, Optional
from core.task_router import route_task
from core.pipeline_manager import run_pipeline
from core.retriever import Retriever
from core.fuser import Fuser 
from core.validator import is_valid_output 
from core.response_formatter import format_response


def run_fuse_pipeline(user_input: str, input_type: str = "text", metadata: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], str]:
    """
    Main pipeline orchestrator that processes input through the entire workflow.
    
    Args:
        user_input: The input text or file path to process
        input_type: Type of input ('text', 'image', 'audio', etc.)
        metadata: Additional metadata for processing
        
    Returns:
        Processed output as a dictionary or string, or error information
    """
    try:
        # Step 1: Detect type and intent
        detected_type, intent = route_task(user_input)
        print(f"Detected type: {detected_type} | Intent: {intent}")

        if intent == "unknown":
            return {"error": "Intent could not be identified."}

        # Step 2: Retrieve content (if needed)
        retriever = Retriever()
        if detected_type == 'text':
            content = retriever.retrieve_text(user_input) or user_input
        elif detected_type == 'image':
            content = retriever.retrieve_image(user_input) or user_input
        else:
            content = user_input

        # Ensure content is not empty
        if not content:
            return {"error": "No content to process."}

        # Step 3: Run appropriate pipeline
        print(f"\n[DEBUG] Running pipeline with task='{intent}', content type={type(content)}")
        result = run_pipeline(task=intent, data=content)
        print(f"[DEBUG] Pipeline result type: {type(result)}, content: {result[:200]}..." if isinstance(result, str) else result)

        # Step 4: Process the result
        fuser = Fuser()
        fused_result = fuser.fuse(task_type=intent, pipeline_output=result)
        
        if not fused_result:
            return {"error": "Failed to process the pipeline output."}

        # Step 5: Validate result before returning
        if not is_valid_output(fused_result, intent):
            return {
                "error": "Generated output is invalid or empty.",
                "raw_output": str(fused_result)[:500]  # Include first 500 chars of raw output
            }

        # Step 6: Format the final response
        formatted = format_response(intent, fused_result)
        return formatted
        
    except Exception as e:
        import traceback
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"[ERROR] Pipeline failed: {error_details}")
        return {"error": f"Processing failed: {str(e)}"}


