"""Pipeline Manager Module

This module handles the loading and execution of Hugging Face pipelines.
It provides a unified interface for running different types of ML models
and handles model loading, caching, and error handling.
"""
import logging
import traceback
from typing import Any, Dict, Optional, Union

from transformers import pipeline, Pipeline
import torch

# Import configurations
from config import MODEL_CONFIG, DEFAULT_PARAMS

# Configure logging
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU

# Global pipeline cache
PIPELINE_CACHE = {}

def get_pipeline(task: str, model_name: str = None, **kwargs) -> Any:
    """
    Get a cached instance of a Hugging Face pipeline.
    
    Args:
        task: The task to create a pipeline for (e.g., 'text-generation')
        model_name: Optional model name to use instead of the default
        **kwargs: Additional arguments to pass to the pipeline
        
    Returns:
        A pipeline instance or error dict if creation fails
    """
    try:
        logger.info(f"Getting pipeline for task: {task}")
        
        # Get the default model for the task if none provided
        if not model_name:
            model_name = MODEL_CONFIG.get(task.replace('-', '_'))
            if not model_name:
                error_msg = f"No model configured for task: {task}"
                logger.error(error_msg)
                return {"error": error_msg}
        
        logger.info(f"Using model: {model_name} for task: {task}")
        
        # Create a cache key based on task and model name
        cache_key = f"{task}:{model_name}"
        
        # Check if we have a cached pipeline
        if cache_key in PIPELINE_CACHE:
            logger.info(f"Using cached pipeline for {cache_key}")
            return PIPELINE_CACHE[cache_key]
            
        logger.info(f"Loading new pipeline for task: {task} with model: {model_name}")
        
        try:
            # Common parameters for all pipelines
            pipeline_params = {
                'model': model_name,
                'device': DEVICE,
                'model_kwargs': {
                    'low_cpu_mem_usage': True
                }
            }
            
            # Special handling for text generation
            if task == "text-generation":
                logger.info("Configuring text-generation pipeline")
                # Ensure we have required parameters for text generation
                default_params = {
                    'max_length': 100,
                    'do_sample': True,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.2,
                    'pad_token_id': 50256,
                    'eos_token_id': 50256,
                }
                
                # Update with any provided kwargs (overriding defaults)
                default_params.update(kwargs)
                pipeline_params.update(default_params)
                
                logger.info(f"Creating text-generation pipeline with params: {pipeline_params}")
                
                # Try to load the pipeline with the specified parameters
                try:
                    pipe = pipeline(
                        task=task,
                        **pipeline_params
                    )
                except Exception as e:
                    logger.warning(f"Failed to load pipeline with default parameters: {str(e)}")
                    logger.info("Trying with trust_remote_code=True...")
                    pipeline_params['trust_remote_code'] = True
                    pipe = pipeline(
                        task=task,
                        **pipeline_params
                    )
                
                PIPELINE_CACHE[cache_key] = pipe
                logger.info(f"Successfully loaded text-generation pipeline")
                return pipe
                
            else:
                # For other tasks, use simpler pipeline creation
                logger.info(f"Creating pipeline for {task} with params: {pipeline_params}")
                pipe = pipeline(
                    task=task,
                    **pipeline_params
                )
                
                PIPELINE_CACHE[cache_key] = pipe
                logger.info(f"Successfully loaded pipeline for {task}")
                return pipe
                
        except Exception as e:
            error_msg = f"Error creating pipeline: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg}
                
    except Exception as e:
        error_msg = f"Unexpected error in get_pipeline: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}
        
        # Update with any provided kwargs (overriding defaults)
        task_params.update(kwargs)
        
        # Try to create the pipeline
        try:
            pipe = pipeline(
                task=task,
                model=model_name,
                device=DEVICE,
                **task_params
            )
            
            # Cache the pipeline
            PIPELINE_CACHE[cache_key] = pipe
            logger.info(f"Successfully loaded pipeline for {task}")
            return pipe
            
        except Exception as e:
            logger.error(f"Error creating pipeline: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # If model loading fails, try with trust_remote_code=True
            try:
                logger.info("Trying with trust_remote_code=True...")
                pipe = pipeline(
                    task=task,
                    model=model_name,
                    device=DEVICE,
                    trust_remote_code=True,
                    **task_params
                )
                PIPELINE_CACHE[cache_key] = pipe
                return pipe
            except Exception as e2:
                logger.error(f"Failed to load model with trust_remote_code: {str(e2)}")
                logger.debug(traceback.format_exc())
                return None
        
    except Exception as e:
        logger.error(f"Unexpected error in get_pipeline: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def run_pipeline(task: str, model_name: str, input_data: Any, **kwargs) -> Any:
    """
    Run a pipeline with the given input data.
    
    Args:
        task: The task to perform (e.g., 'text-generation')
        model_name: The model to use for the task
        input_data: The input data to process
        **kwargs: Additional arguments to pass to the pipeline
        
    Returns:
        The pipeline output or an error message
    """
    try:
        logger.info(f"Running pipeline for task: {task}")
        logger.info(f"Input data: {input_data}")
        
        # Filter out chat_history from generation parameters
        generation_params = kwargs.copy()
        generation_params.pop('chat_history', None)  # Remove chat_history if present
        
        # Get the pipeline with model_name as a separate argument
        pipe = get_pipeline(task, model_name=model_name, **generation_params)
        
        # Check if pipe is an error dict
        if isinstance(pipe, dict) and 'error' in pipe:
            logger.error(f"Pipeline creation failed: {pipe['error']}")
            return pipe
            
        if not pipe:
            error_msg = f"Failed to load pipeline for task: {task}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Run the pipeline
        if task == "text-generation":
            # Special handling for text generation
            if not isinstance(input_data, str) or not input_data.strip():
                error_msg = "Input text cannot be empty"
                logger.error(error_msg)
                return {"error": error_msg}
                
            # Prepare the prompt
            prompt = input_data.strip()
            if not any(prompt.endswith(p) for p in ('.', '?', '!')):
                prompt = prompt.rstrip('.!?') + '.'
            
            logger.info(f"Generating text with prompt: {prompt}")
            
            try:
                # Generate text with simplified parameters
                generation_params = {
                    'max_length': kwargs.get('max_length', 100),
                    'num_return_sequences': 1,
                    'temperature': kwargs.get('temperature', 0.7),
                    'top_k': kwargs.get('top_k', 50),
                    'top_p': kwargs.get('top_p', 0.9),
                    'repetition_penalty': kwargs.get('repetition_penalty', 1.2),
                    'do_sample': True,
                    'pad_token_id': 50256,
                    'eos_token_id': 50256
                }
                
                logger.info(f"Generation params: {generation_params}")
                
                # Generate the text
                output = pipe(prompt, **generation_params)
                
                logger.info(f"Generated output: {output}")
                
                # Format the output
                if output and isinstance(output, list) and len(output) > 0:
                    result = output[0]
                    if isinstance(result, dict) and 'generated_text' in result:
                        # Clean up the response
                        response = result['generated_text'].strip()
                        # Remove any prompt from the beginning if it's there
                        if response.startswith(prompt):
                            response = response[len(prompt):].strip()
                        return response
                    return str(result)
                
                return "I couldn't generate a response. Please try again."
                
            except Exception as e:
                error_msg = f"Error in text generation: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return {"error": error_msg}
            
        else:
            # For other tasks, just run the pipeline directly
            try:
                return pipe(input_data)
            except Exception as e:
                error_msg = f"Error running pipeline: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Unexpected error in run_pipeline: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}
        return {"error": f"Error processing request: {str(e)}"}

# Clean up resources when the module is unloaded
import atexit

@atexit.register
def cleanup():
    """Clean up any resources when the application exits."""
    global PIPELINE_CACHE
    try:
        PIPELINE_CACHE.clear()
        logger.info("Cleared pipeline cache")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
