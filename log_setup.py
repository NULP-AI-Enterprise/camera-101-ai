"""Shared logging setup: timezone-aware timestamps, rotating files."""
from __future__ import annotations

import datetime
import logging
import os
from logging.handlers import RotatingFileHandler

LOGS_DIR = "logs"

# Default UTC+3 (Ukraine/Kyiv). Override with LOG_TZ_OFFSET env var.
_TZ = datetime.timezone(datetime.timedelta(hours=int(os.environ.get("LOG_TZ_OFFSET", "3"))))


class _TZFormatter(logging.Formatter):
    """Formatter that always stamps records in the configured local timezone."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.datetime.fromtimestamp(record.created, tz=_TZ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


_FMT = _TZFormatter(
    "%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str, filename: str) -> logging.Logger:
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_FMT)
    logger.addHandler(ch)

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
