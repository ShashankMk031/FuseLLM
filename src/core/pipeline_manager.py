"""Pipeline Manager Module

This module handles the loading and execution of Hugging Face pipelines.
It provides a unified interface for running different types of ML models
and handles model loading, caching, and error handling.
"""
import os
import logging
from typing import Any, Dict, Optional, Union
from functools import lru_cache

from transformers import pipeline, Pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
import torch

# Import configurations
from config import MODEL_CONFIG, DEFAULT_PARAMS

# Configure logging
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU


def get_task_type(task: str) -> str:
    """Map specific task to its general category."""
    task = task.lower()
    
    if any(t in task for t in ["summar", "tldr"]):
        return "summarization"
    elif any(t in task for t in ["translat", "to french", "to english"]):
        return "translation"
    elif any(t in task for t in ["sentiment", "emotion", "tone"]):
        return "sentiment-analysis"
    elif any(t in task for t in ["classify", "categor"]):
        return "text-classification"
    elif any(t in task for t in ["image", "photo", "picture"]):
        if "caption" in task:
            return "image-to-text"
        return "image-classification"
    elif any(t in task for t in ["audio", "speech", "voice"]) and "recog" in task:
        return "automatic-speech-recognition"
    elif any(t in task for t in ["question", "answer"]):
        return "question-answering"
    
    return task  # Return as is if no mapping found


@lru_cache(maxsize=8)
def get_pipeline(task: str, model_name: str = None, **kwargs) -> Pipeline:
    """
    Load and cache the appropriate Hugging Face pipeline for the task.
    
    Args:
        task: The task to perform (e.g., 'text-generation', 'summarization')
        model_name: Optional model name to override the default
        **kwargs: Additional arguments to pass to the pipeline
        
    Returns:
        A Hugging Face pipeline for the specified task
        
    Raises:
        ValueError: If the task is not supported or model loading fails
    """
    # Normalize task name
    task = task.lower().strip()
    
    # Map to standard task name if needed
    task = get_task_type(task)
    
    # Get model name
    if not model_name:
        # Map task to model config key if needed
        model_key = task.replace('-', '_')
        if model_key not in MODEL_CONFIG:
            raise ValueError(
                f"No default model for task: {task}. "
                f"Available tasks: {', '.join(MODEL_CONFIG.keys())}"
            )
        model_name = MODEL_CONFIG[model_key]
    
    # Log model loading
    logger.info(f"Loading pipeline for task: '{task}' with model: {model_name}")
    
    try:
        # Get default parameters for the task
        task_params = DEFAULT_PARAMS.copy()
        task_params.update(kwargs)  # Override with any provided kwargs
        
        # Special handling for device placement
        device = task_params.pop("device", DEVICE)
        
        # Create the pipeline
        pipe = pipeline(
            task=task,
            model=model_name,
            device=device,
            **task_params
        )
        
        return pipe
        
    except Exception as e:
        logger.error(f"Failed to load pipeline for task '{task}': {str(e)}")
        raise ValueError(f"Failed to load model for task '{task}': {str(e)}")


def run_pipeline(task: str, data: Any, **kwargs) -> Any:
    """
    Run the appropriate Hugging Face pipeline for the given task and data.
    
    This function handles:
    - Input validation and preprocessing
    - Model loading and caching
    - Batch processing for large inputs
    - Error handling and logging
    
    Args:
        task: The task to perform (e.g., 'text-generation', 'summarization')
        data: The input data to process (str, list, dict, or file path)
        **kwargs: Additional arguments to pass to the pipeline
        
    Returns:
        The processed output from the pipeline
        
    Raises:
        ValueError: If the input is invalid or processing fails
    """
    logger.info(f"Running pipeline for task: {task}")
    
    try:
        # Get the appropriate pipeline
        pipe = get_pipeline(task, **kwargs)
    except (KeyError, ValueError) as e:
        logger.warning(f"Pipeline for '{task}' not found. Using fallback 'text-generation'")
        task = "text-generation"
        pipe = get_pipeline(task, **kwargs)
    
    try:
        # Handle different input types
        if isinstance(data, (str, bytes)) and os.path.isfile(data):
            # Handle file paths
            file_ext = os.path.splitext(data)[1].lower()
            
            if file_ext in ['.txt', '.md', '.csv', '.json']:
                # Read text files
                with open(data, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # For text generation, ensure the input is properly formatted
                if task == 'text-generation':
                    # Clean and prepare the input
                    content = content.strip()
                    # Add a prompt if it's a very short input
                    if len(content.split()) < 5:
                        content = f"User: {content}\nAI:"
                    # Generate response with constrained parameters
                    response = pipe(
                        content,
                        **{
                            **kwargs,
                            'return_full_text': False,
                            'max_length': min(kwargs.get('max_length', 50), 100),  # Cap max length
                            'temperature': min(kwargs.get('temperature', 0.7), 0.8),  # Cap temperature
                        }
                    )
                    # Ensure we return a string, not a list
                    if isinstance(response, list) and response:
                        return response[0].get('generated_text', '').strip()
                    return str(response).strip()
                
            elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                # Handle images
                from PIL import Image
                image = Image.open(data)
                return pipe(image, **kwargs)
                
            elif file_ext in ['.wav', '.mp3', '.flac', '.ogg']:
                # Handle audio files
                return pipe(data, **kwargs)
                
            else:
                # Try to process as text
                return pipe(str(data), **kwargs)
                
        elif isinstance(data, (list, tuple, Dataset)):
            # Process batch of inputs
            if len(data) > 10:  # For large batches, use dataset
                if not isinstance(data, Dataset):
                    data = Dataset.from_dict({"text": list(data)})
                return [out for out in pipe(KeyDataset(data, "text"), **kwargs)]
            else:
                return [pipe(item, **kwargs) for item in data]
                
        elif isinstance(data, dict):
            # Handle dictionary inputs (common for multimodal tasks)
            return pipe(**data, **kwargs)
            
        else:
            # Single input (text, image, audio, etc.)
            if task == 'text-generation' and isinstance(data, str):
                # Clean and prepare the input
                data = data.strip()
                # Format as a conversation if needed
                if not data.endswith(('?', '!', '.')):
                    data = f"User: {data}\nAI:"
                
                # Generate response with constrained parameters
                response = pipe(
                    data,
                    **{
                        **kwargs,
                        'return_full_text': False,
                        'max_length': min(kwargs.get('max_length', 50), 100),  # Cap max length
                        'temperature': min(kwargs.get('temperature', 0.7), 0.8),  # Cap temperature
                    }
                )
                
                # Ensure we return a string, not a list
                if isinstance(response, list) and response:
                    return response[0].get('generated_text', '').strip()
                return str(response).strip()
            return pipe(data, **kwargs)
            
    except Exception as e:
        logger.error(f"Error in run_pipeline for task '{task}': {str(e)}")
        
        # Provide more helpful error messages
        if "CUDA out of memory" in str(e):
            raise RuntimeError(
                "The model requires more GPU memory than is available. "
                "Try using a smaller model or running on CPU."
            ) from e
        elif "not found" in str(e).lower():
            raise ValueError(
                f"Model or configuration not found for task: {task}. "
                f"Available tasks: {', '.join(DEFAULT_MODELS.keys())}"
            ) from e
        elif "input" in str(e).lower() and "not supported" in str(e).lower():
            raise ValueError(
                f"Unsupported input type for task '{task}'. "
                f"Expected text, image, or audio, but got: {type(data)}"
            ) from e
            
        # Fallback to a more general pipeline if available
        if task not in ["text-generation", "summarization"]:
            try:
                logger.warning(f"Falling back to text-generation for task: {task}")
                pipe = get_pipeline("text-generation")
                return pipe(str(data), **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback pipeline also failed: {str(fallback_error)}")
                raise RuntimeError(
                    f"Failed to process with both {task} and fallback pipelines. "
                    f"Original error: {str(e)}"
                ) from fallback_error
        
        # Re-raise the original error if no fallback worked
        raise ValueError(f"Failed to process input: {str(e)}") from e
