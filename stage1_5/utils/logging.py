"""Logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional


def configure_logging(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
                      log_file: Optional[str | Path] = None) -> None:
    """Configure root logger with console + optional file handlers."""

    logging.basicConfig(level=getattr(logging, level.upper()),
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
