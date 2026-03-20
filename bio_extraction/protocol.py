"""
protocol.py
===========
Abstract base class that every pipeline phase must implement.

Design notes
------------
- ``PhaseProtocol`` is Generic over its input and output Pydantic model types,
  so static type checkers can verify that phases are wired up correctly in
  the runner without any phase knowing about its neighbours.
- The runner calls only ``phase_name`` and ``run()``.  Everything else is
  the phase's own business.
- ``run()`` returning ``None`` is the sanctioned way for a phase to signal
  "discard this document silently" (e.g. lang_score below threshold in Phase 1).
  The runner logs the discard and does NOT enqueue the document downstream.
- Any exception raised by ``run()`` is caught by the runner, written to the
  dead-letter queue, logged at ERROR level, and processing continues.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class PhaseProtocol(ABC, Generic[InputT, OutputT]):
    """Abstract base for all pipeline phases."""

    @property
    @abstractmethod
    def phase_name(self) -> str:
        """
        Unique snake_case identifier for this phase.

        Used as:
        - The sub-directory name under ``./checkpoints/``
        - The sub-directory name under ``./dead_letter/``
        - The ``phase_name`` field injected into every log record by this phase's logger.

        Examples: ``"phase1_acquisition"``, ``"phase5_extraction"``
        """
        ...

    @abstractmethod
    def run(self, input_data: InputT) -> OutputT | None:
        """
        Process a single document through this phase.

        Parameters
        ----------
        input_data:
            The Pydantic model produced by the previous phase (or PhaseOneInput
            for Phase 1).

        Returns
        -------
        OutputT
            The phase's output model on success.
        None
            If the document should be silently discarded (runner skips it,
            no dead-letter entry).

        Raises
        ------
        Any exception
            Caught by the runner and routed to the dead-letter queue.
        """
        ...
