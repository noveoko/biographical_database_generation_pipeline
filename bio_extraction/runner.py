"""
runner.py
=========
PipelineRunner — orchestrates the 6-phase pipeline.

Architecture overview
---------------------
::

    PhaseOneInput(s)
         │
         ▼
    [Queue 0]  ─►  Phase 1  ─►  [Queue 1]  ─►  Phase 2  ─►  ...  ─►  Phase 6
                     │                │
               checkpoint          checkpoint
               dead-letter         dead-letter

Key design decisions
--------------------
1. **Queue-mediated**: phases never call each other.  The runner feeds items
   into a phase's input queue and drains the output queue before moving on.
2. **Checkpoint-first**: before calling ``phase.run()``, the runner checks
   whether a checkpoint already exists.  If so, it deserialises the saved
   result and enqueues it directly, skipping the phase entirely.
3. **Error isolation**: every ``phase.run()`` call is wrapped in try/except.
   Exceptions are routed to the dead-letter queue; the document is dropped
   from the pipeline; remaining documents are unaffected.
4. **None == discard**: if ``phase.run()`` returns ``None``, the document is
   logged as discarded and not enqueued downstream.
5. **Sequential by default** (``run_all``).  A ``run_parallel`` stub is
   provided for a future ThreadPoolExecutor implementation.
"""

from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bio_extraction.checkpoint import CheckpointEngine
from bio_extraction.config import Settings, get_settings
from bio_extraction.contracts import (
    AcquisitionResult,
    ClassificationResult,
    ExtractionResult,
    LayoutResult,
    OCRResult,
    PhaseOneInput,
    ResolutionResult,
)
from bio_extraction.dead_letter import DeadLetterQueue
from bio_extraction.logging_config import get_phase_logger, setup_logging
from bio_extraction.protocol import PhaseProtocol

# Map phase index (0-based) → the expected OUTPUT model class.
# This lets the runner deserialise checkpoints without phase knowledge.
_PHASE_OUTPUT_MODELS: dict[int, type[BaseModel]] = {
    0: AcquisitionResult,
    1: ClassificationResult,
    2: LayoutResult,
    3: OCRResult,
    4: ExtractionResult,
    5: ResolutionResult,
}

logger = logging.getLogger("bio_extraction.runner")


class PipelineRunner:
    """
    Orchestrates an ordered list of PhaseProtocol instances through the full
    document corpus.

    Parameters
    ----------
    phases:
        Ordered list of phase implementations, from Phase 1 (Acquisition) to
        Phase 6 (Resolution).  Must be exactly 6 elements.
    settings:
        Typed settings object.  If omitted, ``get_settings()`` is called.
    checkpoint_dir:
        Override the checkpoint root directory (handy in tests).
    dead_letter_dir:
        Override the dead-letter root directory (handy in tests).
    """

    def __init__(
        self,
        phases: list[PhaseProtocol],  # type: ignore[type-arg]
        settings: Settings | None = None,
        checkpoint_dir: Path | None = None,
        dead_letter_dir: Path | None = None,
    ) -> None:
        self.phases = phases
        self.settings = settings or get_settings()

        setup_logging(log_dir=Path(self.settings.log_dir))

        cp_dir = checkpoint_dir or Path(self.settings.checkpoint_dir)
        dl_dir = dead_letter_dir or Path(self.settings.dead_letter_dir)

        self.checkpoint = CheckpointEngine(base_dir=cp_dir)
        self.dead_letter = DeadLetterQueue(base_dir=dl_dir)

        # One queue between each pair of consecutive phases.
        # queues[i] feeds INTO phases[i] (i.e. it contains the OUTPUT of phases[i-1]).
        # queues[0] is populated by the runner with PhaseOneInput seeds.
        self._queues: list[queue.Queue[BaseModel]] = [
            queue.Queue() for _ in range(len(phases) + 1)
        ]

    # ------------------------------------------------------------------
    # Input seeding
    # ------------------------------------------------------------------

    def _seed_phase1_inputs(self) -> int:
        """
        Enumerate documents and push PhaseOneInput objects into queues[0].

        Returns the number of items enqueued.
        """
        source = self.settings.source
        count = 0

        if source == "local":
            input_dir = Path(self.settings.input_dir)
            if not input_dir.is_dir():
                logger.warning(f"input_dir does not exist or is not a directory: {input_dir}")
                return 0
            for pdf_path in sorted(input_dir.glob("*.pdf")):
                self._queues[0].put(
                    PhaseOneInput(source="local", local_path=pdf_path)
                )
                count += 1
            logger.info(f"Seeded {count} local PDF(s) from {input_dir}")

        elif source == "commoncrawl":
            # Phase 1 implementation handles CC enumeration internally.
            # The runner injects a single sentinel that tells Phase 1 to start
            # its own CC index walk.  Phase 1 is responsible for yielding
            # multiple AcquisitionResult objects.
            # For now we push one PhaseOneInput with cc_warc_record=None
            # as a trigger; the stub will raise NotImplementedError.
            self._queues[0].put(PhaseOneInput(source="commoncrawl", cc_warc_record={}))
            count = 1
            logger.info("Seeded CC acquisition trigger")

        else:
            raise ValueError(f"Unknown source mode: {source!r}")

        return count

    # ------------------------------------------------------------------
    # Core processing loop
    # ------------------------------------------------------------------

    def _process_phase(
        self,
        phase_index: int,
        input_data: BaseModel,
    ) -> BaseModel | None:
        """
        Run one phase for one document, with checkpoint and dead-letter handling.

        Parameters
        ----------
        phase_index:
            0-based index into ``self.phases``.
        input_data:
            The item dequeued from the phase's input queue.

        Returns
        -------
        BaseModel | None
            The phase output, or None if the document was discarded / failed.
        """
        phase = self.phases[phase_index]
        phase_logger = get_phase_logger(phase.phase_name)

        # Extract doc_id for checkpoint lookup.
        # Phase 1 input (PhaseOneInput) has no doc_id — checkpoint check is done
        # AFTER phase 1 produces its output (using the hash-derived doc_id).
        doc_id: str | None = getattr(input_data, "doc_id", None)

        # ------------------------------------------------------------------
        # Checkpoint fast-path (phases 2-6 only — Phase 1 has no input doc_id)
        # ------------------------------------------------------------------
        output_model_class = _PHASE_OUTPUT_MODELS.get(phase_index)
        if doc_id is not None and output_model_class is not None:
            if self.checkpoint.exists(phase.phase_name, doc_id):
                try:
                    cached = self.checkpoint.load(phase.phase_name, doc_id, output_model_class)
                    phase_logger.info(
                        "Checkpoint hit — skipping phase",
                        extra={"doc_id": doc_id},
                    )
                    return cached
                except Exception as exc:
                    phase_logger.warning(
                        f"Checkpoint load failed ({exc}); re-running phase",
                        extra={"doc_id": doc_id},
                    )

        # ------------------------------------------------------------------
        # Run the phase
        # ------------------------------------------------------------------
        try:
            phase_logger.info("Phase start", extra={"doc_id": doc_id or "?"})
            result = phase.run(input_data)  # type: ignore[arg-type]
        except Exception as exc:
            phase_logger.error(
                f"Phase raised {type(exc).__name__}: {exc}",
                extra={"doc_id": doc_id or "?"},
                exc_info=True,
            )
            self.dead_letter.record(
                phase_name=phase.phase_name,
                doc_id=doc_id or "unknown",
                error=exc,
                input_data=input_data,
            )
            return None

        # ------------------------------------------------------------------
        # Discard path
        # ------------------------------------------------------------------
        if result is None:
            phase_logger.info("Document discarded by phase", extra={"doc_id": doc_id or "?"})
            return None

        # ------------------------------------------------------------------
        # Checkpoint the output (phases 1-5 only; phase 6 writes to DB)
        # ------------------------------------------------------------------
        result_doc_id: str | None = getattr(result, "doc_id", None)
        if result_doc_id is not None and phase_index < 5:
            try:
                self.checkpoint.save(phase.phase_name, result_doc_id, result)
            except Exception as exc:
                phase_logger.warning(
                    f"Failed to save checkpoint: {exc}",
                    extra={"doc_id": result_doc_id},
                )

        phase_logger.info(
            "Phase complete",
            extra={"doc_id": result_doc_id or doc_id or "?"},
        )
        return result

    # ------------------------------------------------------------------
    # Public run methods
    # ------------------------------------------------------------------

    def run_all(self) -> dict[str, Any]:
        """
        Process all documents sequentially through all phases.

        The method:
        1. Seeds Phase 1's input queue.
        2. For each phase, drains its input queue, processes each item,
           and enqueues successful results into the next phase's input queue.
        3. Returns a summary dict with counts per phase.

        Returns
        -------
        dict[str, Any]
            ``{"phase_name": {"processed": N, "skipped": N, "discarded": N, "failed": N}, ...}``
        """
        summary: dict[str, dict[str, int]] = {}

        total_seeded = self._seed_phase1_inputs()
        logger.info(f"Pipeline starting — {total_seeded} document(s) seeded")

        for phase_index, phase in enumerate(self.phases):
            in_q = self._queues[phase_index]
            out_q = self._queues[phase_index + 1]
            counts = {"processed": 0, "skipped": 0, "discarded": 0, "failed": 0}

            logger.info(f"--- Running {phase.phase_name} ---")

            while not in_q.empty():
                item: BaseModel = in_q.get_nowait()
                doc_id: str | None = getattr(item, "doc_id", None)

                # Checkpoint fast-path check for summary accounting
                output_model_class = _PHASE_OUTPUT_MODELS.get(phase_index)
                if (
                    doc_id is not None
                    and output_model_class is not None
                    and self.checkpoint.exists(phase.phase_name, doc_id)
                ):
                    counts["skipped"] += 1
                    try:
                        cached = self.checkpoint.load(phase.phase_name, doc_id, output_model_class)
                        if cached is not None:
                            out_q.put(cached)
                    except Exception:
                        pass  # _process_phase will handle re-run on the next call
                    continue

                result = self._process_phase(phase_index, item)

                if result is None:
                    # Distinguish discard vs failure by checking dead-letter
                    _dlid = doc_id or "unknown"
                    if self.dead_letter._record_path(phase.phase_name, _dlid).exists():
                        counts["failed"] += 1
                    else:
                        counts["discarded"] += 1
                else:
                    counts["processed"] += 1
                    out_q.put(result)

            summary[phase.phase_name] = counts
            logger.info(f"{phase.phase_name} done: {counts}")

        logger.info("Pipeline complete")
        return summary

    def run_parallel(self) -> dict[str, Any]:
        """
        Future parallel execution using ``concurrent.futures.ThreadPoolExecutor``.

        TODO: Implement per-phase thread pools where each phase runs as a
              producer-consumer worker pulling from its input queue and pushing
              to its output queue.  Phases that are I/O-bound (Phase 1, Phase 4)
              benefit most; CPU-bound phases (Phase 5) should use ProcessPoolExecutor.

        For now, delegates to ``run_all()``.
        """
        # TODO: implement true parallelism
        return self.run_all()
