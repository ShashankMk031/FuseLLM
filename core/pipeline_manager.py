from typing import Any
from transformers import pipeline
from functools import lru_cache

# Optional: override default model for specific tasks
DEFAULT_MODELS = {
    "text-generation": "gpt2",
    "summarization": "facebook/bart-large-cnn",
    "translation": "Helsinki-NLP/opus-mt-en-fr",
    "text-classification": "distilbert-base-uncased-finetuned-sst-2-english",
    "zero-shot-classification": "facebook/bart-large-mnli",
    "image-classification": "google/vit-base-patch16-224",
    "object-detection": "facebook/detr-resnet-50",
    "image-to-text": "Salesforce/blip-image-captioning-base",
    "audio-classification": "superb/hubert-base-superb-er",
    "automatic-speech-recognition": "openai/whisper-base",
    "text-to-speech": "espnet/kan-bayashi_ljspeech_vits",
    "image-text-to-text": "nlpconnect/vit-gpt2-image-captioning",
    "feature-extraction": "distilbert-base-uncased"
}

@lru_cache(maxsize=None)
def get_pipeline(task: str):
    """
    Load and cache the correct Hugging Face pipeline for the task.
    """
    if task not in DEFAULT_MODELS:
        raise ValueError(f"No supported pipeline for task: {task}")
    
    model = DEFAULT_MODELS[task]
    print(f"Loading pipeline for task: {task} with model: {model}")
    
    return pipeline(task, model=model)

from transformers import pipeline

def run_pipeline(task: str, data: Any, **kwargs) -> Any:
    """
    Run the appropriate Hugging Face pipeline for the given task and data.
    
    Args:
        task: The task to perform (e.g., 'text-generation', 'summarization')
        data: The input data to process
        **kwargs: Additional arguments to pass to the pipeline
        
    Returns:
        The processed output from the pipeline
    """
    try:
        # Try to get the pipeline for the specific task
        pipe = pipeline(task, **kwargs)
    except (KeyError, ValueError) as e:
        print(f"[WARN] Pipeline for '{task}' not found. Using fallback 'text-generation'")
        task = "text-generation"
        pipe = pipeline(task, **kwargs)
    
    try:
        # Handle different task types with appropriate parameters
        if task in ["image-classification", "object-detection"]:
            return pipe(data)
            
        elif task in ["automatic-speech-recognition", "audio-classification"]:
            return pipe(data)
            
        elif task in ["summarization", "translation"]:
            return pipe(data, max_length=130, min_length=30, do_sample=False)
            
        elif task == "text-generation":
            # For text generation, ensure we get a reasonable response
            if isinstance(data, str) and len(data) > 1024:
                data = data[:1024]  # Truncate very long inputs
            
            result = pipe(
                data,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256  # GPT-2's pad token
            )
            return result[0] if isinstance(result, list) and len(result) > 0 else str(result)
            
        elif task == "sentiment-analysis":
            return pipe(data, return_all_scores=True)
            
        else:
            # Generic fallback for other tasks
            return pipe(data)
            
    except Exception as e:
        print(f"[ERROR] Error running {task} pipeline: {str(e)}")
        return {"error": f"Failed to process {task} pipeline: {str(e)}"}

