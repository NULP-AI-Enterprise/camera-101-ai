"""Shared logging setup for stream.py and post_analyser.py."""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

LOGS_DIR = "logs"

_FMT = logging.Formatter(
    "%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str, filename: str) -> logging.Logger:
    """
    Return a logger that writes to console (INFO+) and a rotating file (DEBUG+).
    Safe to call multiple times — handlers are only added once.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    # Console — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_FMT)
    logger.addHandler(ch)

    # Rotating file — all levels, 5 MB × 3 backups
    fh = RotatingFileHandler(
        os.path.join(LOGS_DIR, filename),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FMT)
    logger.addHandler(fh)

    return logger
