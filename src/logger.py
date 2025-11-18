"""
Centralized logging module for the entire project.
Uses FastAPI logger with custom formatting.
"""

import logging
import os
from fastapi.logger import logger as fastapi_logger

msg_fmt = "[ %(asctime)s ] %(levelname)s [ %(name)s ] - %(message)s"
time_fmt = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(msg_fmt, time_fmt)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

LOG_LEVEL = os.getenv('LOG_LEVEL', default='INFO')
fastapi_logger.addHandler(handler)
fastapi_logger.setLevel(LOG_LEVEL)

logger_name = os.getenv('API_NAME', default='tractian-ml-engineering-llm')
fastapi_logger.name = logger_name

logger = fastapi_logger
