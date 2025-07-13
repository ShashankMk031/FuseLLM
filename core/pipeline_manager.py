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

def run_pipeline(task: str, input_data, additional_args=None):
    """
    Run the pipeline for the given task and input.
    """
    pipe = get_pipeline(task)
    
    if additional_args is None:
        additional_args = {}
    
    # Simple example; can be extended per task
    try:
        if task == "zero-shot-classification":
            return pipe(input_data, candidate_labels=["science", "sports", "politics", "health", "tech"], **additional_args)
        elif task == "text-to-speech":
            return pipe(input_data, **additional_args)  # Returns waveform
        elif task == "image-text-to-text":
            return pipe(**input_data, **additional_args)
        else:
            return pipe(input_data, **additional_args)
    except Exception as e:
        return {"error": str(e), "input": input_data}
