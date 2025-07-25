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
    # Text models - Using more reliable models
    'text-generation': 'gpt2',  # Using base GPT-2 for better compatibility
    'summarization': 'facebook/bart-large-cnn',
    'translation': 'Helsinki-NLP/opus-mt-en-fr',
    'text-classification': 'distilbert-base-uncased-finetuned-sst-2-english',
    'zero-shot-classification': 'facebook/bart-large-mnli',
    
    # Special models for specific intents - all use the same text generation model
    'joke': 'gpt2',
    'greeting': 'gpt2',
    'weather': 'gpt2',
    'definition': 'gpt2',
    'science': 'gpt2',
    'science_question': 'gpt2',
    'general_knowledge': 'gpt2',
    'general': 'gpt2'  # Add general intent
}

# Default parameters for different tasks
DEFAULT_PARAMS = {
    # Text generation parameters (for all text-based tasks)
    'text-generation': {
        'max_length': 100,           # Maximum length of the generated text
        'min_length': 10,            # Minimum length of the generated text
        'temperature': 0.7,          # Controls randomness (lower = more focused)
        'top_k': 50,                 # Keep only top k tokens with highest probability
        'top_p': 0.9,                # Nucleus sampling: keeps the top p% of probability mass
        'repetition_penalty': 1.2,   # Penalize repetition
        'length_penalty': 1.0,       # No length penalty
        'no_repeat_ngram_size': 3,   # Prevent n-gram repetition
        'do_sample': True,           # Enable sampling
        'early_stopping': True,      # Stop generation when appropriate
        'pad_token_id': 50256,       # For GPT-2 compatibility
        'eos_token_id': 50256,       # End of sequence token
        'num_return_sequences': 1,   # Only generate one sequence
    },
    
    # Default parameters for other tasks
    'summarization': {
        'max_length': 130,
        'min_length': 30,
        'do_sample': False
    },
    'translation': {
        'max_length': 128
    },
    'text-classification': {},
    'zero-shot-classification': {}
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