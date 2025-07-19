"""
Configuration settings for FuseLLM.

This file contains all the configuration parameters used throughout the application.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.resolve()

# Data directories
DATA_DIR = BASE_DIR / 'data'
EMBEDDINGS_DIR = BASE_DIR / 'embeddings'

# Model configurations
MODEL_CONFIG = {
    # Text models
    'text_generation': 'microsoft/DialoGPT-small',  # More focused than medium
    'summarization': 'facebook/bart-large-cnn',
    'translation': 'Helsinki-NLP/opus-mt-en-fr',
    'text_classification': 'distilbert-base-uncased-finetuned-sst-2-english',
    'zero_shot': 'facebook/bart-large-mnli',
    
    # Image models
    'image_classification': 'google/vit-base-patch16-224',
    'image_to_text': 'nlpconnect/vit-gpt2-image-captioning',
    
    # Audio models
    'speech_recognition': 'openai/whisper-tiny',
    'audio_classification': 'superb/hubert-base-superb-er',
}

# Default parameters for different tasks
DEFAULT_PARAMS = {
    # Text generation parameters
    'max_length': 50,            # Shorter responses
    'min_length': 5,             # Minimum length
    'num_return_sequences': 1,   # Only one response
    'temperature': 0.7,          # Less random than before
    'top_k': 30,                 # More focused sampling
    'top_p': 0.85,               # Slightly more focused nucleus
    'repetition_penalty': 1.5,   # Higher penalty for repetition
    'length_penalty': 1.0,       # Neutral length penalty
    'no_repeat_ngram_size': 2,   # Prevent 2-gram repetition
    'do_sample': True,           # Enable sampling
    'early_stopping': True,      # Stop generation when appropriate
    'pad_token_id': 50256,       # For GPT-2 compatibility
}

# File type mappings
FILE_EXTENSIONS = {
    'text': ['.txt', '.md', '.csv', '.json', '.py'],
    'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
    'audio': ['.wav', '.mp3', '.flac', '.ogg'],
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'level': 'DEBUG',
            'filename': 'fusellm.log',
            'mode': 'a',
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        'fusellm': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# API Keys (load from environment variables)
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '') # Currently not needed

# Create necessary directories
for directory in [DATA_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(exist_ok=True)

# Validate configurations
assert all(model for model in MODEL_CONFIG.values()), "All model configurations must be non-empty"
assert all(isinstance(ext, str) for exts in FILE_EXTENSIONS.values() for ext in exts), \
    "All file extensions must be strings"