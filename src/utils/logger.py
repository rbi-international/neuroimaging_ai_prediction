import logging
import os
from datetime import datetime

def get_logger(name: str, log_file: str = "logs/train.log", level: str = "INFO") -> logging.Logger:
    # Create logs/ folder if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file, encoding = "utf-8")

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

