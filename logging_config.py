import logging
import os
import sys
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configure logging for the application.

    Args:
        log_level: The logging level (default: INFO)
        log_file: Optional path to a log file
    """
    # Create logs directory if logging to file
    if log_file:
        log_dir = Path(os.path.dirname(log_file))
        log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress overly verbose logs from libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("ebooklib").setLevel(logging.WARNING)

    return root_logger
