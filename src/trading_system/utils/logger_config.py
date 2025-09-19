# src/trading_system/utils/logger_config.py
import logging
import sys

def setup_logging(level="INFO"):
    """
    Configures the root logger for the application.

    Args:
        level (str): The logging level to set (e.g., "INFO", "DEBUG").
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Use a more detailed format for better debugging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Get the root logger and remove any existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # Configure the basicConfig with the new settings
    logging.basicConfig(
        level=log_level,
        format=log_format,
        stream=sys.stdout  # Ensure logs go to the console
    )
    
    # Create a specific logger for the trading system if needed,
    # or just use the root logger configured above.
    logger = logging.getLogger("trading_system")
    logger.info(f"Logging configured to level: {level}")