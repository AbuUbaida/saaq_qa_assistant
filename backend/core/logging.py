"""Logging helpers (tiny wrapper around stdlib logging)."""

from __future__ import annotations

import logging
from typing import Optional

from .config import get_settings

_configured = False


def setup_logging(level: Optional[str] = None) -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    log_level = (level or settings.log_level or "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


