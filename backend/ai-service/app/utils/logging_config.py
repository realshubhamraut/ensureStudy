"""
Centralized Logging Configuration for ensureStudy
All services use this for consistent logging
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
    service_name: str = "ai-service",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging for a service
    
    Args:
        service_name: Name of the service (used in log filename)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to write to file
        log_to_console: Whether to write to console
    
    Returns:
        Configured logger
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Formatter
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
    
    # File handler (rotating, max 10MB, keep 5 backups)
    if log_to_file:
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = LOG_DIR / f"{service_name}_{today}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # Also log errors to separate file
        error_file = LOG_DIR / f"{service_name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8"
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
    
    # Log startup
    logger = logging.getLogger(service_name)
    logger.info(f"=== {service_name.upper()} STARTED ===")
    logger.info(f"Log level: {level}")
    if log_to_file:
        logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name)


# Exception logging decorator
def log_exceptions(logger: logging.Logger):
    """Decorator to log exceptions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator
