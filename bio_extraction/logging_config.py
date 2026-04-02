"""
logging_config.py
=================
Structured logging setup for the bio_extraction pipeline.

Every log record is written in two places:
1. ``./logs/pipeline.jsonl`` — one JSON object per line for machine consumption
   / post-hoc analysis.
2. ``stderr`` — human-readable format for live monitoring.

Both handlers include ``timestamp``, ``level``, ``phase_name``, ``doc_id``,
and ``message`` when those fields are present.

Usage
-----
    from bio_extraction.logging_config import setup_logging, get_phase_logger

    setup_logging(log_dir=Path("./logs"))

    logger = get_phase_logger("phase3_layout")
    logger.info("Processing slice", extra={"doc_id": "abc123"})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, MutableMapping

# ---------------------------------------------------------------------------
# JSON lines formatter
# ---------------------------------------------------------------------------


class _JsonLinesFormatter(logging.Formatter):
    """Emit one JSON object per log record, terminated by a newline."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "phase_name": getattr(record, "phase_name", None),
            "doc_id": getattr(record, "doc_id", None),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Human-readable stderr formatter
# ---------------------------------------------------------------------------


_STDERR_FORMAT = "%(asctime)s  %(levelname)-8s  " "[%(phase_name)s]  doc=%(doc_id)s  %(message)s"

_STDERR_DEFAULTS = {"phase_name": "-", "doc_id": "-"}


class _DefaultsFilter(logging.Filter):
    """Inject default values for ``phase_name`` and ``doc_id`` if not set."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, default in _STDERR_DEFAULTS.items():
            if not hasattr(record, key):
                setattr(record, key, default)
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_configured = False


def setup_logging(log_dir: Path | None = None, level: int = logging.DEBUG) -> None:
    """
    Configure the root logger with a JSON-lines file handler and a stderr handler.

    Safe to call multiple times — subsequent calls are no-ops unless
    ``_configured`` is reset.

    Parameters
    ----------
    log_dir:
        Directory for the ``pipeline.jsonl`` log file.
        Defaults to ``./logs`` relative to the current working directory.
    level:
        Minimum log level for both handlers.
    """
    global _configured
    if _configured:
        return

    log_dir = log_dir or Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("bio_extraction")
    root.setLevel(level)

    # --- JSON-lines file handler ---
    jsonl_handler = logging.FileHandler(log_dir / "pipeline.jsonl", encoding="utf-8")
    jsonl_handler.setLevel(level)
    jsonl_handler.setFormatter(_JsonLinesFormatter())
    jsonl_handler.addFilter(_DefaultsFilter())
    root.addHandler(jsonl_handler)

    # --- Human-readable stderr handler ---
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(logging.Formatter(_STDERR_FORMAT))
    stderr_handler.addFilter(_DefaultsFilter())
    root.addHandler(stderr_handler)

    root.propagate = False
    _configured = True


def get_phase_logger(phase_name: str) -> logging.Logger:
    """
    Return a child logger pre-configured with ``phase_name`` in its extra fields.

    Parameters
    ----------
    phase_name:
        The phase identifier string (e.g. ``"phase3_layout"``).

    Returns
    -------
    logging.Logger
        A logger whose records always carry ``phase_name`` in their ``extra``
        dict, ready for both the JSON-lines and stderr handlers.

    Example
    -------
        logger = get_phase_logger("phase5_extraction")
        logger.info("Entity found", extra={"doc_id": "abc123"})
    """
    logger = logging.getLogger(f"bio_extraction.{phase_name}")

    # Attach an adapter that injects phase_name automatically
    class _PhaseAdapter(logging.LoggerAdapter):
        def process(
            self, msg: str, kwargs: MutableMapping[str, Any]
        ) -> tuple[str, MutableMapping[str, Any]]:
            extra = kwargs.setdefault("extra", {})
            extra.setdefault("phase_name", phase_name)
            return msg, kwargs

    return _PhaseAdapter(logger, {"phase_name": phase_name})  # type: ignore[return-value]


# Alias for backward compatibility and runner.py import
configure_logging = setup_logging
