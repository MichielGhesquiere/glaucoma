import logging
import sys
import os
from datetime import datetime

def setup_logging(log_level: str = 'INFO', log_file: str = None, log_dir: str = 'logs', include_timestamp: bool = True) -> None:
    """
    Configures logging for the application.

    Sets up logging to both console and optionally to a file.

    Args:
        log_level (str): The minimum logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
                         Defaults to 'INFO'.
        log_file (str, optional): If provided, logs will also be written to this file.
                                  If None, a timestamped file is created in log_dir. Defaults to None.
        log_dir (str): Directory to store log files if log_file is None. Defaults to 'logs'.
        include_timestamp (bool): Whether to include timestamp in the default log filename. Defaults to True.
    """
    level = getattr(logging, log_level.upper(), logging.INFO) # Default to INFO if invalid level

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    handlers = [logging.StreamHandler(sys.stdout)] # Log to console by default

    if log_file is None:
        # Create a default log file name if none specified
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"app_log_{timestamp}.log" if include_timestamp else "app.log"
        log_file_path = os.path.join(log_dir, filename)
    else:
        # Use the specified log file path
        log_file_path = log_file
        # Ensure directory exists if a full path is given
        log_file_dir = os.path.dirname(log_file_path)
        if log_file_dir:
            os.makedirs(log_file_dir, exist_ok=True)

    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        handlers.append(file_handler)
        print(f"Logging to file: {log_file_path}") # Print path for user info

    # Remove existing handlers before configuring to avoid duplicate logs in interactive sessions
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure logging
    logging.basicConfig(level=level, format=log_format, datefmt=date_format, handlers=handlers)

    logging.info(f"Logging setup complete. Level: {log_level.upper()}")

# Example usage (optional, for testing)
if __name__ == '__main__':
    print("--- Default Logging (INFO to console and timestamped file in ./logs) ---")
    setup_logging()
    logging.debug("This is a debug message (should not appear).")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")

    print("\n--- DEBUG Logging to console only ---")
    # Need to reconfigure - basicConfig only works once unless force=True is used,
    # or by manipulating handlers directly. Simpler to just show config:
    # setup_logging(log_level='DEBUG', log_file=None) # This won't work as expected after first setup

    # Proper way to change level after setup (if needed):
    logging.getLogger().setLevel(logging.DEBUG)
    print("Changed root logger level to DEBUG")
    logging.debug("This debug message should now appear.")

    print("\n--- Logging to a specific file (INFO level) ---")
    specific_log_file = "test_log.log"
    setup_logging(log_level='INFO', log_file=specific_log_file)
    logging.info(f"This message should go to console and {specific_log_file}")

    # Clean up test log file
    if os.path.exists(specific_log_file):
         try:
             # Close handlers associated with the file before removing it
             for handler in logging.getLogger().handlers:
                 if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(specific_log_file):
                     handler.close()
                     logging.getLogger().removeHandler(handler)
             os.remove(specific_log_file)
             print(f"Cleaned up {specific_log_file}")
         except Exception as e:
             print(f"Error cleaning up {specific_log_file}: {e}")

    # Clean up logs directory if it was created and is empty
    if os.path.exists("logs") and not os.listdir("logs"):
        try:
            os.rmdir("logs")
            print("Cleaned up empty logs directory.")
        except Exception as e:
            print(f"Error cleaning up logs directory: {e}")