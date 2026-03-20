"""
checkpoint.py
=============
CheckpointEngine — atomic serialise/deserialise of phase output models.

Each checkpoint is stored at::

    {checkpoint_dir}/{phase_name}/{doc_id}.json

Writes are atomic: data is written to a ``.tmp`` file first, then renamed
into place, so a crash mid-write never leaves a corrupt checkpoint.

``bytes`` fields inside Pydantic models are expected to survive the
``model_dump_json`` / ``model_validate_json`` round-trip via the base64
validators defined in ``contracts.py``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from bio_extraction.exceptions import CheckpointError

logger = logging.getLogger("bio_extraction.checkpoint")

T = TypeVar("T", bound=BaseModel)


class CheckpointEngine:
    """Manages serialisation, deserialisation, and discovery of phase checkpoints."""

    def __init__(self, base_dir: Path | str = "./checkpoints") -> None:
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

    def _checkpoint_path(self, phase_name: str, doc_id: str) -> Path:
        return self._phase_dir(phase_name) / f"{doc_id}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, phase_name: str, doc_id: str, data: BaseModel) -> Path:
        """
        Serialise *data* to disk atomically.

        Parameters
        ----------
        phase_name:
            Sub-directory key (e.g. ``"phase2_classification"``).
        doc_id:
            Pipeline primary key.
        data:
            The Pydantic model to serialise.

        Returns
        -------
        Path
            The path of the written checkpoint file.

        Raises
        ------
        CheckpointError
            If the write fails for any OS-level reason.
        """
        target = self._checkpoint_path(phase_name, doc_id)
        tmp = target.with_suffix(".tmp")
        try:
            # model_dump_json handles base64 for bytes fields via model overrides
            json_str = data.model_dump_json(indent=2)
            tmp.write_text(json_str, encoding="utf-8")
            os.replace(tmp, target)  # atomic on POSIX; best-effort on Windows
        except OSError as exc:
            raise CheckpointError(
                f"Failed to write checkpoint {phase_name}/{doc_id}: {exc}"
            ) from exc
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        logger.debug("Checkpoint saved", extra={"phase_name": phase_name, "doc_id": doc_id})
        return target

    def load(
        self, phase_name: str, doc_id: str, model_class: type[T]
    ) -> T | None:
        """
        Deserialise a checkpoint back into a Pydantic model.

        Parameters
        ----------
        phase_name:
            Sub-directory key.
        doc_id:
            Pipeline primary key.
        model_class:
            The Pydantic model class to validate against.

        Returns
        -------
        T | None
            The deserialised model, or ``None`` if no checkpoint exists.

        Raises
        ------
        CheckpointError
            If the file exists but cannot be read or fails validation.
        """
        path = self._checkpoint_path(phase_name, doc_id)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            return model_class.model_validate_json(raw)
        except Exception as exc:
            raise CheckpointError(
                f"Failed to load checkpoint {phase_name}/{doc_id}: {exc}"
            ) from exc

    def exists(self, phase_name: str, doc_id: str) -> bool:
        """Return True if a checkpoint file already exists for this (phase, doc_id) pair."""
        return self._checkpoint_path(phase_name, doc_id).exists()

    def list_completed(self, phase_name: str) -> set[str]:
        """
        Return the set of doc_ids for which a checkpoint exists under *phase_name*.

        Parameters
        ----------
        phase_name:
            Sub-directory key to scan.

        Returns
        -------
        set[str]
            doc_ids (stems of the ``.json`` files found).
        """
        phase_dir = self.base_dir / phase_name
        if not phase_dir.is_dir():
            return set()
        return {p.stem for p in phase_dir.glob("*.json")}

    def clear(self, phase_name: str | None = None) -> None:
        """
        Delete checkpoint files.

        Parameters
        ----------
        phase_name:
            If provided, clear only that phase's checkpoints.
            If ``None``, clear ALL phase checkpoints under ``base_dir``.
        """
        import shutil

        if phase_name is not None:
            phase_dir = self.base_dir / phase_name
            if phase_dir.is_dir():
                shutil.rmtree(phase_dir)
                logger.info(f"Cleared checkpoints for phase '{phase_name}'")
        else:
            for child in self.base_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
            logger.info("Cleared ALL checkpoints")
