from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import settings


def setup_logging() -> logging.Logger:
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("skill_rag_qdrant")
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    file_handler = RotatingFileHandler(settings.log_file, maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(settings.log_level.upper())

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(settings.log_level.upper())

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logging()
