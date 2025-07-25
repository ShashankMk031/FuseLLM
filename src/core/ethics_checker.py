import re

# Very simple list of blocked keywords for demonstration
BLOCKED_KEYWORDS = [
    "hate", "violence", "kill", "murder", "terrorist", "racist", "sex", "abuse",
    "drugs", "bomb", "attack", "nazi", "suicide"
]

def is_ethically_safe(output: any) -> bool:
    """
    Checks if the output contains any potentially unsafe or unethical content.
    Currently uses a basic keyword match. Can be upgraded to use a classification model.

    Args:
        output (Any): The model output (string or list of dicts depending on pipeline)

    Returns:
        bool: True if ethically safe, False otherwise
    """
    # Normalize string from different output types
    if isinstance(output, list) and len(output) > 0:
        text = output[0].get("generated_text") or output[0].get("summary_text") or output[0].get("text") or ""
    elif isinstance(output, dict):
        text = output.get("text", "")
    elif isinstance(output, str):
        text = output
    else:
        text = str(output)

    text = text.lower()

    # Rule-based keyword detection
    for keyword in BLOCKED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", text):
            return False

    return True
