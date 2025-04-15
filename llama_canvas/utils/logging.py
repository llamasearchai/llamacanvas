"""
Logging utilities for LlamaCanvas.
"""

import logging
import sys
from typing import Optional

from llama_canvas.utils.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers have been added
    if not logger.handlers:
        # Set log level from settings
        log_level = getattr(logging, settings.get("LOG_LEVEL", "INFO"))
        logger.setLevel(log_level)

        # Create console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console)

        # Propagate to root logger
        logger.propagate = False

    return logger


def setup_logging(log_file: Optional[str] = None, log_level: str = "INFO") -> None:
    """
    Set up logging for the application.

    Args:
        log_file: Path to log file (optional)
        log_level: Logging level
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, log_level))
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Add file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress overly verbose loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
