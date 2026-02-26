# src/utils/logger.py

import logging
import os
from datetime import datetime
import sys

def get_logger(name: str, log_to_file: bool = False, log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    """
    Create a logger with console output and optional timestamped log file.

    Args:
        name (str): Logger name (usually __name__).
        log_to_file (bool): If True, log messages to a file.
        log_dir (str): Log directory.
        level (int): Logger level (default: logging.INFO).

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

        # Console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Timestamped log file
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

            # Use script name if running as main
            if name == "__main__":
                script_path = sys.argv[0]
                script_name = os.path.splitext(os.path.basename(script_path))[0]
                log_filename = f"{script_name}_{timestamp}.log"
            else:
                log_filename = f"{name.replace('.', '_')}_{timestamp}.log"

            log_path = os.path.join(log_dir, log_filename)

            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f"Logger initialized with file: {log_path}")

    return logger