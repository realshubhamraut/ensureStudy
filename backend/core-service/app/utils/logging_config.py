"""
Centralized Logging Configuration for ensureStudy Core Service
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime

# Log directory
LOG_DIR = Path(__file__).parent.parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    service_name: str = "core-service",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """Set up logging for a service"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers = []
    
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
    
    if log_to_file:
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = LOG_DIR / f"{service_name}_{today}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # Error-only file
        error_file = LOG_DIR / f"{service_name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
    
    logger = logging.getLogger(service_name)
    logger.info(f"=== {service_name.upper()} STARTED ===")
    logger.info(f"Log file: {LOG_DIR / f'{service_name}_{datetime.now().strftime(\"%Y-%m-%d\")}.log'}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
