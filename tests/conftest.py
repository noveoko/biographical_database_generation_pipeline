"""
conftest.py
===========
Shared pytest fixtures for the bio_extraction test suite.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bio_extraction.checkpoint import CheckpointEngine
from bio_extraction.contracts import (
    AcquisitionResult,
    ClassificationResult,
    ContentSlice,
    DocumentSource,
    DocumentType,
    ExtractionMethod,
    ExtractionResult,
    LayoutResult,
    OCREntry,
    OCRResult,
    PersonEntity,
    PhaseOneInput,
)
from bio_extraction.dead_letter import DeadLetterQueue


# ---------------------------------------------------------------------------
# Minimal dummy PDF bytes (not a valid PDF, but sufficient for unit tests)
# ---------------------------------------------------------------------------

DUMMY_PDF_BYTES: bytes = b"%PDF-1.4 dummy content for testing"
DUMMY_DOC_ID: str = hashlib.sha256(DUMMY_PDF_BYTES).hexdigest()


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture()
def tmp_dead_letter_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dead_letter"
    d.mkdir()
    return d


@pytest.fixture()
def checkpoint_engine(tmp_checkpoint_dir: Path) -> CheckpointEngine:
    return CheckpointEngine(base_dir=tmp_checkpoint_dir)


@pytest.fixture()
def dead_letter_queue(tmp_dead_letter_dir: Path) -> DeadLetterQueue:
    return DeadLetterQueue(base_dir=tmp_dead_letter_dir)


# ---------------------------------------------------------------------------
# Contract fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def phase_one_input_local(tmp_path: Path) -> PhaseOneInput:
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(DUMMY_PDF_BYTES)
    return PhaseOneInput(source=DocumentSource.LOCAL, local_path=pdf)


@pytest.fixture()
def acquisition_result() -> AcquisitionResult:
    return AcquisitionResult(
        doc_id=DUMMY_DOC_ID,
        pdf_bytes=DUMMY_PDF_BYTES,
        source=DocumentSource.LOCAL,
        source_url=None,
        warc_id=None,
        lang_score=0.95,
        filename="test.pdf",
        acquired_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture()
def classification_result() -> ClassificationResult:
    return ClassificationResult(
        doc_id=DUMMY_DOC_ID,
        doc_type=DocumentType.DIRECTORY,
        confidence=0.92,
        sample_page_indices=[0, 1, 2],
        classified_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture()
def layout_result() -> LayoutResult:
    slices = [
        ContentSlice(
            slice_id=f"{DUMMY_DOC_ID}_p0_e{i}",
            page_num=0,
            entry_index=i,
            bbox=(0, i * 50, 200, (i + 1) * 50),
            image_bytes=None,
        )
        for i in range(3)
    ]
    return LayoutResult(
        doc_id=DUMMY_DOC_ID,
        doc_type=DocumentType.DIRECTORY,
        slices=slices,
        analyzed_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture()
def ocr_result(layout_result: LayoutResult) -> OCRResult:
    entries = [
        OCREntry(
            slice_id=s.slice_id,
            text="Kowalski Jan ur. 1893 w Warszawie",
            confidence_score=0.88,
            bboxes=[(0, 0, 100, 20)],
            needs_review=False,
        )
        for s in layout_result.slices
    ]
    return OCRResult(
        doc_id=DUMMY_DOC_ID,
        ocr_entries=entries,
        processed_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture()
def extraction_result() -> ExtractionResult:
    entity = PersonEntity(
        entity_id="00000000-0000-0000-0000-000000000001",
        slice_id=f"{DUMMY_DOC_ID}_p0_e0",
        surname="Kowalski",
        given_names=["Jan"],
        birth_date="1893",
        death_date=None,
        locations=["Warszawa"],
        roles=[],
        raw_text="Kowalski Jan ur. 1893 w Warszawie",
        confidence=0.9,
        extraction_method=ExtractionMethod.REGEX_CACHED,
    )
    return ExtractionResult(
        doc_id=DUMMY_DOC_ID,
        entities=[entity],
        extracted_at=datetime.now(tz=timezone.utc),
    )


@pytest.fixture()
def tmp_dirs(tmp_path: Path) -> dict:
    dirs = {
        "checkpoints": tmp_path / "checkpoints",
        "dead_letter": tmp_path / "dead_letter",
        "logs": tmp_path / "logs",
        "input_pdfs": tmp_path / "input_pdfs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture()
def person_entity() -> "PersonEntity":
    from bio_extraction.contracts import PersonEntity, ExtractionMethod

    return PersonEntity(
        entity_id="11111111-1111-1111-1111-111111111111",
        slice_id="deadbeef01234567_p0_e0",
        surname="Kowalski",
        given_names=["Jan"],
        birth_date="1889",
        death_date=None,
        locations=["Grodno"],
        roles=[],
        raw_text="KOWALSKI Jan ur. 1889 Grodno",
        confidence=0.78,
        extraction_method=ExtractionMethod.REGEX_CACHED,
    )
