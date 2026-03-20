"""
contracts.py
============
All Pydantic models shared across pipeline phases.

Design rules:
- Every model carries ``doc_id`` as the pipeline primary key.
- ``bytes`` fields are transparently round-tripped through base64 so that
  ``model.model_dump_json()`` produces valid JSON without any manual conversion.
- No model imports from any phase module — contracts are the *only* shared
  dependency between phases.
- All fields carry a ``description`` for self-documenting serialised checkpoints.
"""

from __future__ import annotations

import base64
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocumentSource(str, Enum):
    LOCAL = "local"
    COMMONCRAWL = "commoncrawl"


class DocumentType(str, Enum):
    DIRECTORY = "directory"
    NEWSPAPER = "newspaper"
    CIVIL_RECORD = "civil_record"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    REGEX_CACHED = "regex_cached"
    LLM_OLLAMA = "llm_ollama"
    LLM_API = "llm_api"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Helpers — transparent base64 serialisation for bytes fields
# ---------------------------------------------------------------------------


class _Base64Bytes(bytes):
    """
    A ``bytes`` subclass that Pydantic serialises to/from a base64 string.

    Use this as the field type anywhere a Pydantic model must store raw bytes
    in a JSON-serialisable checkpoint.  Validation accepts both ``bytes``
    (pass-through) and ``str`` (decoded from base64), so round-tripping
    through ``model_dump_json`` / ``model_validate_json`` works transparently.
    """

    @classmethod
    def __get_validators__(cls):  # pydantic v1 compat shim — v2 uses __get_pydantic_core_schema__
        yield cls._validate

    @classmethod
    def _validate(cls, v: Any) -> "_Base64Bytes":
        if isinstance(v, bytes):
            return cls(v)
        if isinstance(v, str):
            return cls(base64.b64decode(v))
        raise ValueError(f"Cannot coerce {type(v)} to bytes")

    def __repr__(self) -> str:  # keep repr tidy in logs
        return f"<{len(self)} bytes>"


def _bytes_to_b64(value: bytes | None) -> str | None:
    """Encode bytes to a base64 string for JSON serialisation."""
    if value is None:
        return None
    return base64.b64encode(value).decode("ascii")


def _b64_to_bytes(value: str | bytes | None) -> bytes | None:
    """Decode a base64 string back to bytes during deserialisation."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    return base64.b64decode(value)


# ---------------------------------------------------------------------------
# Phase 1 input — feeds the pipeline
# ---------------------------------------------------------------------------


class PhaseOneInput(BaseModel):
    """
    Seed object consumed by Phase 1 (Acquisition).

    The runner constructs one of these per candidate document and pushes it
    into Phase 1's input queue.  Exactly one of ``local_path`` or
    ``cc_warc_record`` must be populated, matching ``source``.
    """

    source: DocumentSource = Field(
        description="Whether the document comes from local disk or Common Crawl."
    )
    local_path: Path | None = Field(
        default=None,
        description="Absolute or relative path to the PDF file on local disk. Populated when source=LOCAL.",
    )
    cc_warc_record: dict[str, Any] | None = Field(
        default=None,
        description="Raw Common Crawl index record dict (url, filename, offset, length, …). Populated when source=COMMONCRAWL.",
    )


# ---------------------------------------------------------------------------
# CP1 — AcquisitionResult
# ---------------------------------------------------------------------------


class AcquisitionResult(BaseModel):
    """Output of Phase 1 (Acquisition). Contains the raw PDF bytes and provenance metadata."""

    model_config = {"arbitrary_types_allowed": True}

    doc_id: str = Field(
        description="Pipeline primary key — SHA-256 hex digest of the raw PDF bytes."
    )
    pdf_bytes: bytes = Field(
        description="Raw PDF bytes. Serialised as a base64 string in JSON checkpoints."
    )
    source: DocumentSource = Field(
        description="Origin of this document (local disk or Common Crawl)."
    )
    source_url: str | None = Field(
        default=None,
        description="Original URL of the document. None for local-mode documents.",
    )
    warc_id: str | None = Field(
        default=None,
        description="Common Crawl WARC record ID (e.g. '<urn:uuid:…>'). None for local-mode documents.",
    )
    lang_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability estimate (0–1) that the document is written in Polish.",
    )
    filename: str = Field(description="Original filename or a slug derived from the CC URL.")
    acquired_at: datetime = Field(
        description="UTC timestamp when Phase 1 completed for this document."
    )

    @field_validator("pdf_bytes", mode="before")
    @classmethod
    def _decode_pdf_bytes(cls, v: Any) -> bytes:
        """Accept either raw bytes (in-memory) or a base64 string (from JSON checkpoint)."""
        if isinstance(v, bytes):
            return v
        if isinstance(v, str):
            return base64.b64decode(v)
        raise ValueError(f"pdf_bytes must be bytes or base64 str, got {type(v)}")

    def model_dump_json(self, **kwargs: Any) -> str:  # type: ignore[override]
        """Override so that pdf_bytes is serialised as base64."""
        import json

        d = self.model_dump()
        d["pdf_bytes"] = base64.b64encode(self.pdf_bytes).decode("ascii")
        d["source"] = self.source.value
        d["acquired_at"] = self.acquired_at.isoformat()
        return json.dumps(d, **kwargs)


# ---------------------------------------------------------------------------
# CP2 — ClassificationResult
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """Output of Phase 2 (Classification). Records the detected document type and confidence."""

    doc_id: str = Field(description="Pipeline primary key matching the AcquisitionResult.")
    doc_type: DocumentType = Field(description="Detected document type.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classifier confidence score (0–1) for the assigned doc_type.",
    )
    sample_page_indices: list[int] = Field(
        description="Indices (0-based) of the pages that were sampled to determine document type.",
    )
    classified_at: datetime = Field(
        description="UTC timestamp when Phase 2 completed for this document."
    )


# ---------------------------------------------------------------------------
# CP3 — LayoutResult (with nested ContentSlice)
# ---------------------------------------------------------------------------


class ContentSlice(BaseModel):
    """
    A single rectangular region of interest within a document page.

    Represents one logical entry (e.g. a person record in a directory, a birth
    entry in a civil register) detected by the layout analysis phase.
    """

    model_config = {"arbitrary_types_allowed": True}

    slice_id: str = Field(
        description="Unique identifier within the document. Format: '{doc_id}_p{page_num}_e{entry_index}'.",
    )
    page_num: int = Field(description="0-based page number within the PDF.")
    entry_index: int = Field(description="0-based index of this entry on the given page.")
    bbox: tuple[int, int, int, int] = Field(
        description="Bounding box in pixels: (x1, y1, x2, y2) where (x1, y1) is top-left.",
    )
    image_bytes: bytes | None = Field(
        default=None,
        description="PNG-encoded crop of this slice. Serialised as base64 in JSON. None if not extracted.",
    )

    @field_validator("image_bytes", mode="before")
    @classmethod
    def _decode_image_bytes(cls, v: Any) -> bytes | None:
        if v is None:
            return None
        if isinstance(v, bytes):
            return v
        if isinstance(v, str):
            return base64.b64decode(v)
        raise ValueError(f"image_bytes must be bytes, base64 str, or None, got {type(v)}")

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        d = super().model_dump(**kwargs)
        if self.image_bytes is not None:
            d["image_bytes"] = base64.b64encode(self.image_bytes).decode("ascii")
        return d


class LayoutResult(BaseModel):
    """Output of Phase 3 (Layout Analysis). Contains all detected content slices."""

    doc_id: str = Field(description="Pipeline primary key.")
    doc_type: DocumentType = Field(description="Document type forwarded from ClassificationResult.")
    slices: list[ContentSlice] = Field(
        description="Ordered list of content slices detected across all pages.",
    )
    analyzed_at: datetime = Field(
        description="UTC timestamp when Phase 3 completed for this document."
    )


# ---------------------------------------------------------------------------
# CP4 — OCRResult (with nested OCREntry)
# ---------------------------------------------------------------------------


class OCREntry(BaseModel):
    """OCR output for a single ContentSlice."""

    slice_id: str = Field(description="Matches ContentSlice.slice_id — provenance link.")
    text: str = Field(description="Full plain-text output from Tesseract for this slice.")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean word-level confidence reported by Tesseract, normalised to 0–1.",
    )
    bboxes: list[tuple[int, int, int, int]] = Field(
        description="Word-level bounding boxes (x1, y1, x2, y2) in slice-local pixel coordinates.",
    )
    needs_review: bool = Field(
        description="True if confidence_score is below the configured threshold and manual review is recommended.",
    )


class OCRResult(BaseModel):
    """Output of Phase 4 (OCR). Contains OCR text for every slice in the document."""

    doc_id: str = Field(description="Pipeline primary key.")
    ocr_entries: list[OCREntry] = Field(
        description="One OCREntry per ContentSlice, in the same order as LayoutResult.slices.",
    )
    processed_at: datetime = Field(
        description="UTC timestamp when Phase 4 completed for this document."
    )


# ---------------------------------------------------------------------------
# CP5 — ExtractionResult (with nested PersonEntity)
# ---------------------------------------------------------------------------


class PersonEntity(BaseModel):
    """
    A single biographical entity extracted from one OCR slice.

    Represents one person as parsed from the raw OCR text, before cross-document
    deduplication takes place in Phase 6.
    """

    entity_id: str = Field(
        description="UUID v4 string uniquely identifying this extraction instance.",
    )
    slice_id: str = Field(
        description="Matches OCREntry.slice_id — provenance link back to the source text region.",
    )
    surname: str = Field(description="Primary surname (normalised to NFC Unicode, title-cased).")
    given_names: list[str] = Field(
        description="List of given names in the order they appear in the source."
    )
    birth_date: str | None = Field(
        default=None,
        description="Birth date as ISO-8601 string or partial year string (e.g. '1892'). None if unknown.",
    )
    death_date: str | None = Field(
        default=None,
        description="Death date as ISO-8601 string or partial year string. None if unknown.",
    )
    locations: list[str] = Field(
        description="Place names associated with this person (birth place, residence, etc.).",
    )
    roles: list[str] = Field(
        description="Professional or social roles extracted from the text (e.g. 'notariusz', 'lekarz').",
    )
    raw_text: str = Field(
        description="The original OCR text segment from which this entity was extracted (for auditability).",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence: 1.0 for a cache-hit regex match, lower for LLM-generated patterns.",
    )
    extraction_method: ExtractionMethod = Field(
        description="Which extraction strategy produced this entity.",
    )


class ExtractionResult(BaseModel):
    """Output of Phase 5 (Extraction). Contains all PersonEntity objects found in the document."""

    doc_id: str = Field(description="Pipeline primary key.")
    entities: list[PersonEntity] = Field(
        description="All biographical entities extracted from this document.",
    )
    extracted_at: datetime = Field(
        description="UTC timestamp when Phase 5 completed for this document."
    )


# ---------------------------------------------------------------------------
# Phase 6 Output — ResolutionResult (not checkpointed — goes to DB)
# ---------------------------------------------------------------------------


class ResolvedPerson(BaseModel):
    """
    The outcome of attempting to merge one PersonEntity into the master database.

    Produced by Phase 6 for each entity after fuzzy deduplication.
    """

    person_db_id: int = Field(
        description="SQLite rowid of the canonical person record (new or existing).",
    )
    entity_id: str = Field(
        description="Back-reference to PersonEntity.entity_id that was resolved.",
    )
    is_new: bool = Field(
        description="True if a new DB row was inserted; False if merged into an existing record.",
    )
    merge_confidence: float | None = Field(
        default=None,
        description="Fuzzy-match score (0–1) used for the merge decision. None when is_new=True.",
    )


class ResolutionResult(BaseModel):
    """
    Output of Phase 6 (Resolution). Records DB write outcomes for all entities in a document.

    This model is NOT checkpointed to disk — it is the terminal output written
    directly to the SQLite database.
    """

    doc_id: str = Field(description="Pipeline primary key.")
    resolved_persons: list[ResolvedPerson] = Field(
        description="One ResolvedPerson per input PersonEntity.",
    )
    resolved_at: datetime = Field(
        description="UTC timestamp when Phase 6 completed for this document."
    )
