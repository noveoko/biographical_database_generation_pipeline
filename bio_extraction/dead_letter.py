"""
dead_letter.py
==============
DeadLetterQueue — records phase failures for later inspection or replay.

Each failure is written to::

    {dead_letter_dir}/{phase_name}/{doc_id}.json

The record contains:
- ``timestamp``   — UTC ISO-8601 string
- ``phase``       — phase_name string
- ``doc_id``      — pipeline primary key
- ``error_class`` — fully-qualified exception class name
- ``traceback``   — full traceback as a string
- ``input_data``  — JSON-serialised input model (if available), else null

A failure record is written even if input_data serialisation itself fails
(the field is set to ``null`` with a note in the traceback).
"""

from __future__ import annotations

import json
import logging
import traceback as tb
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("bio_extraction.dead_letter")


class DeadLetterQueue:
    """Records per-document phase failures with full diagnostic context."""

    def __init__(self, base_dir: Path | str = "./dead_letter") -> None:
        """
        Parameters
        ----------
        base_dir:
            Root directory under which per-phase sub-directories are created.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phase_dir(self, phase_name: str) -> Path:
        d = self.base_dir / phase_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _record_path(self, phase_name: str, doc_id: str) -> Path:
        return self._phase_dir(phase_name) / f"{doc_id}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        phase_name: str,
        doc_id: str,
        error: Exception,
        input_data: BaseModel | None = None,
    ) -> Path:
        """
        Write a failure record to the dead-letter directory.

        Parameters
        ----------
        phase_name:
            The phase that raised the exception.
        doc_id:
            Pipeline primary key of the failed document.
        error:
            The exception instance.
        input_data:
            The Pydantic input model that was passed to the phase (for replay).
            Pass ``None`` if not available.

        Returns
        -------
        Path
            Path to the written dead-letter file.
        """
        # Serialise input model — best effort
        input_json: Any = None
        if input_data is not None:
            try:
                input_json = json.loads(input_data.model_dump_json())
            except Exception as inner:
                input_json = f"<serialisation failed: {inner}>"

        payload: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "phase": phase_name,
            "doc_id": doc_id,
            "error_class": f"{type(error).__module__}.{type(error).__qualname__}",
            "traceback": "".join(tb.format_exception(type(error), error, error.__traceback__)),
            "input_data": input_json,
        }

        path = self._record_path(phase_name, doc_id)
        try:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            # Failing to write the dead-letter record should never crash the runner.
            logger.error(
                f"Could not write dead-letter record for {phase_name}/{doc_id}: {exc}"
            )

        logger.error(
            f"Dead-letter: {type(error).__name__} in {phase_name} for doc={doc_id}",
            extra={"phase_name": phase_name, "doc_id": doc_id},
        )
        return path

    def list_failures(self, phase_name: str | None = None) -> list[dict[str, Any]]:
        """
        Return all dead-letter records as a list of dicts.

        Parameters
        ----------
        phase_name:
            If given, return only failures from that phase.
            If ``None``, return failures from all phases.

        Returns
        -------
        list[dict]
            Parsed JSON payloads, sorted by timestamp ascending.
        """
        records: list[dict[str, Any]] = []

        dirs: list[Path]
        if phase_name is not None:
            phase_dir = self.base_dir / phase_name
            dirs = [phase_dir] if phase_dir.is_dir() else []
        else:
            dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]

        for d in dirs:
            for json_file in d.glob("*.json"):
                try:
                    records.append(json.loads(json_file.read_text(encoding="utf-8")))
                except Exception as exc:
                    logger.warning(f"Could not parse dead-letter file {json_file}: {exc}")

        records.sort(key=lambda r: r.get("timestamp", ""))
        return records
