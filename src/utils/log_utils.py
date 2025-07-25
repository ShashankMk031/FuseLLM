import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

# Configure root logger
logger = logging.getLogger("FuseLLM")
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create file handler which logs even debug messages
log_file = log_dir / f"fusellm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create console handler with a higher log level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
if not logger.handlers:  # Avoid adding handlers multiple times
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_event(module_name: str, message: str, level: str = "info", **kwargs):
    """
    Log an event with the specified level and additional context.
    
    Args:
        module_name: Name of the module where the event occurred
        message: The message to log
        level: Log level (debug, info, warning, error, critical)
        **kwargs: Additional context to include in the log
    """
    log_message = f"[{module_name}] {message}"
    if kwargs:
        log_message += f" | {kwargs}"
    
    level = level.lower()
    if level == "debug":
        logger.debug(log_message)
    elif level == "info":
        logger.info(log_message)
    elif level == "warning":
        logger.warning(log_message)
    elif level == "error":
        logger.error(log_message)
    elif level == "critical":
        logger.critical(log_message)
    else:
        logger.info(f"[Unknown Level:{level}] {log_message}")

def log_info(message: str, **kwargs):
    """Log an info level message."""
    log_event("INFO", message, "info", **kwargs)

def log_warning(message: str, **kwargs):
    """Log a warning level message."""
    log_event("WARNING", message, "warning", **kwargs)

def log_error(message: str, **kwargs):
    """Log an error level message."""
    log_event("ERROR", message, "error", **kwargs)

def log_debug(message: str, **kwargs):
    """Log a debug level message."""
    log_event("DEBUG", message, "debug", **kwargs)
