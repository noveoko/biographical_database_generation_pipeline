"""
exceptions.py
=============
Custom exception hierarchy for the bio_extraction pipeline.

All pipeline exceptions inherit from PipelineError so callers can
catch the entire family with a single ``except PipelineError`` clause,
or target a specific phase / subsystem with a narrower catch.
"""


class PipelineError(Exception):
    """Base for all pipeline errors."""


class PhaseError(PipelineError):
    """Error raised within a specific pipeline phase."""

    def __init__(self, phase_name: str, doc_id: str, message: str) -> None:
        self.phase_name = phase_name
        self.doc_id = doc_id
        super().__init__(f"[{phase_name}] doc={doc_id}: {message}")


class CheckpointError(PipelineError):
    """Raised when a checkpoint read or write operation fails."""


class ConfigError(PipelineError):
    """Raised when the configuration file is missing, malformed, or contains invalid values."""


class AcquisitionError(PhaseError):
    """Phase 1 — WARC fetch failure, corrupt PDF bytes, language-score below threshold, etc."""


class OCRError(PhaseError):
    """Phase 4 — Tesseract process failure, empty output, or below-threshold confidence."""


class ExtractionError(PhaseError):
    """Phase 5 — regex compile failure, LLM API error, or unparseable model output."""


class ResolutionError(PhaseError):
    """Phase 6 — SQLite write failure, fuzzy-match engine error, or integrity constraint violation."""
