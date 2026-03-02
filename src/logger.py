import logging
import os
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(f"logs/pipeline.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger