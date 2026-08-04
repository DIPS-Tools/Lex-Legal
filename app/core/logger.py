"""
Project logging utility.

Implemented with Loguru. Console and file output are controlled through
``.env`` settings, and daily log files are generated as
``logs/app_YYYYMMDD.log``.

Features:
1. Configuration-driven output and log levels through ``.env``.
2. Automatic file path: ``<project-root>/logs/app_YYYYMMDD.log``.
3. Automatic cleanup of expired logs according to the retention setting.
4. UTF-8 encoding for reliable multilingual log output.
5. Queue-based logging suitable for multithreaded and asynchronous execution.
6. Ready for use by importing ``logger`` from any project module.
7. Accurate caller locations by skipping Loguru and this utility's own frames.
"""

import sys
import inspect
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger


# Step 1: Load configuration from .env.
load_dotenv()

# Step 2: Read logging settings, using defaults when values are absent.
LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")

# Step 3: Define log paths by deriving the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

# Step 4: Define a colored, structured, readable log format.
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)

# Step 5: Initialize the logging configuration.
def init_logger():
    """
    Initialize and return the global Loguru logger.

    1. Remove Loguru's default console sink to prevent duplicate output.
    2. Enable or disable console output according to ``.env``.
    3. Enable or disable file output and create ``logs/`` when required.
    4. Configure formatting, levels, daily rotation, and retention.
    """
    # Remove Loguru's default console sink.
    logger.remove()

    # Add console output when enabled through .env.
    if LOG_CONSOLE_ENABLE:
        logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True
        )

    # Add daily file output when enabled through .env.
    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=LOG_FILE_PATH,
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

    return logger

# Step 6: Initialize the logger and patch caller-location metadata.
base_logger = init_logger()

def fix_log_position(record):
    """Set log metadata to the first external business-code stack frame."""
    for frame in inspect.stack():
        # Skip Loguru internals and this utility to locate the real caller.
        if ("_logger.py" in frame.filename or frame.function == "_log") or "logger.py" in frame.filename:
            continue
        # Replace the displayed source fields with the actual caller details.
        record.update(
            name=frame.filename.split("/")[-1].split("\\")[-1],
            function=frame.function,
            line=frame.lineno
        )
        break

# Apply caller correction and export the shared project logger.
logger = base_logger.patch(fix_log_position)

# Manual smoke test for caller-location correction.
if __name__ == '__main__':
    logger.info("【测试】logger.py内部调用（仅测试，业务模块调用会显示正确文件名）")
    logger.error("【测试】程序报错啦啦啦啦啦")
    print(f"日志文件输出路径：{LOG_FILE_PATH}")
