from core.task_router import classify_intent
from core.task_router import detect_input_type
from core.pipeline_manager import run_pipeline
from core.retriever import Retriever
from core.fuser import fuser 
from core.validator import is_valid_output 


def orchestrate(user_input: str, input_type: str = "text", metadata: dict = None):
    # Step 1: Detect type and intent
    detected_type, intent = detect_type_and_intent(user_input, input_type)
    print(f"Detected type: {detected_type} | Intent: {intent}")

    if intent == "unknown":
        return {"error": "Intent could not be identified."}

    # Step 2: Retrieve content (if needed)
    content = Retriever().retrieve(user_input, detected_type)

    # Step 3: Run appropriate pipeline
    result = run_pipeline(content, task=intent, input_type=detected_type)

    # Step 4: (Optional) Fuse if multimodal or hybrid logic is needed
    fused_result = fuser([result])  # You can keep it as is

    # Step 5: Validate result before returning
    if not is_valid_output(fused_result, intent):
        return {"error": "Generated output is invalid or empty."}

    # Step 6: (Later) Ethics check here
    # if not is_ethically_safe(fused_result):
    #     return {"error": "Generated content may violate ethical guidelines."}

    return fused_result
