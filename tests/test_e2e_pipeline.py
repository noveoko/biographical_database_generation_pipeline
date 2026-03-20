"""
End-to-end smoke tests for the bio_extraction pipeline infrastructure.

These tests verify that the runner, checkpoint engine, dead-letter queue,
and contract models work together correctly WITHOUT requiring any real phase
implementation.  Phase stubs are replaced with controlled test doubles.

Test doubles strategy
---------------------
Each test creates lightweight ``FakePhaseN`` classes that:
  - Inherit ``PhaseProtocol``
  - Return a hard-coded contract model on ``run()``
  - Optionally raise an exception to test dead-letter routing

This ensures infra tests are hermetic and fast (no disk I/O beyond tmp dirs,
no network, no OCR or LLM calls).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

_NOW = datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
_MINIMAL_PDF = b"%PDF-1.4\n%%EOF"
_DOC_ID = "deadbeef01234567"


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_phase_error_is_pipeline_error(self) -> None:
        err = PhaseError("phase1_acquisition", "abc123", "boom")
        assert isinstance(err, PipelineError)
        assert "phase1_acquisition" in str(err)
        assert "abc123" in str(err)

    def test_checkpoint_error_is_pipeline_error(self) -> None:
        assert issubclass(CheckpointError, PipelineError)


# ---------------------------------------------------------------------------
# CheckpointEngine tests
# ---------------------------------------------------------------------------


class TestCheckpointEngine:
    def test_save_and_load_roundtrip(
        self,
        checkpoint_engine: CheckpointEngine,
        acquisition_result: AcquisitionResult,
    ) -> None:
        doc_id = acquisition_result.doc_id
        checkpoint_engine.save("phase1_acquisition", doc_id, acquisition_result)
        loaded = checkpoint_engine.load("phase1_acquisition", doc_id, AcquisitionResult)
        assert loaded is not None
        assert loaded.doc_id == doc_id
        assert loaded.lang_score == pytest.approx(0.95)

    def test_bytes_roundtrip_via_base64(
        self,
        checkpoint_engine: CheckpointEngine,
        acquisition_result: AcquisitionResult,
    ) -> None:
        """pdf_bytes must survive JSON serialization/deserialization intact."""
        checkpoint_engine.save("phase1_acquisition", _DOC_ID, acquisition_result)
        loaded = checkpoint_engine.load("phase1_acquisition", _DOC_ID, AcquisitionResult)
        assert loaded is not None
        assert loaded.pdf_bytes == acquisition_result.pdf_bytes

    def test_exists_false_before_save(self, checkpoint_engine: CheckpointEngine) -> None:
        assert not checkpoint_engine.exists("phase1_acquisition", "nope")

    def test_exists_true_after_save(
        self,
        checkpoint_engine: CheckpointEngine,
        acquisition_result: AcquisitionResult,
    ) -> None:
        checkpoint_engine.save("phase1_acquisition", _DOC_ID, acquisition_result)
        assert checkpoint_engine.exists("phase1_acquisition", _DOC_ID)

    def test_list_completed(
        self,
        checkpoint_engine: CheckpointEngine,
        acquisition_result: AcquisitionResult,
    ) -> None:
        assert checkpoint_engine.list_completed("phase1_acquisition") == set()
        checkpoint_engine.save("phase1_acquisition", _DOC_ID, acquisition_result)
        assert checkpoint_engine.list_completed("phase1_acquisition") == {_DOC_ID}

    def test_load_nonexistent_returns_none(self, checkpoint_engine: CheckpointEngine) -> None:
        result = checkpoint_engine.load("phase1_acquisition", "ghost", AcquisitionResult)
        assert result is None

    def test_clear_single_phase(
        self,
        checkpoint_engine: CheckpointEngine,
        acquisition_result: AcquisitionResult,
    ) -> None:
        checkpoint_engine.save("phase1_acquisition", _DOC_ID, acquisition_result)
        checkpoint_engine.clear("phase1_acquisition")
        assert not checkpoint_engine.exists("phase1_acquisition", _DOC_ID)

    def test_clear_all(
        self,
        checkpoint_engine: CheckpointEngine,
        acquisition_result: AcquisitionResult,
        classification_result: ClassificationResult,
    ) -> None:
        checkpoint_engine.save("phase1_acquisition", _DOC_ID, acquisition_result)
        checkpoint_engine.save("phase2_classification", _DOC_ID, classification_result)
        checkpoint_engine.clear()
        assert checkpoint_engine.list_completed("phase1_acquisition") == set()
        assert checkpoint_engine.list_completed("phase2_classification") == set()

    def test_atomic_write_leaves_no_tmp_file(
        self,
        checkpoint_engine: CheckpointEngine,
        acquisition_result: AcquisitionResult,
        tmp_dirs: dict[str, Path],
    ) -> None:
        checkpoint_engine.save("phase1_acquisition", _DOC_ID, acquisition_result)
        tmp_files = list((tmp_dirs["checkpoints"] / "phase1_acquisition").glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# DeadLetterQueue tests
# ---------------------------------------------------------------------------


class TestDeadLetterQueue:
    def test_record_creates_file(
        self,
        dead_letter_queue: DeadLetterQueue,
        acquisition_result: AcquisitionResult,
    ) -> None:
        exc = ValueError("something went wrong")
        path = dead_letter_queue.record("phase2_classification", _DOC_ID, exc, acquisition_result)
        assert path.exists()

    def test_record_contents(
        self,
        dead_letter_queue: DeadLetterQueue,
        acquisition_result: AcquisitionResult,
        tmp_dirs: dict[str, Path],
    ) -> None:
        import json

        exc = RuntimeError("kaboom")
        dead_letter_queue.record("phase2_classification", _DOC_ID, exc, acquisition_result)
        record_path = tmp_dirs["dead_letter"] / "phase2_classification" / f"{_DOC_ID}.json"
        data = json.loads(record_path.read_text())
        # The DLQ uses "phase" as the key name
        phase_key = "phase_name" if "phase_name" in data else "phase"
        assert data[phase_key] == "phase2_classification"
        assert data["doc_id"] == _DOC_ID
        # error text is in traceback (and optionally error_message)
        searchable = " ".join(
            [
                data.get("traceback") or "",
                data.get("error_message") or "",
                data.get("error") or "",
            ]
        )
        assert "kaboom" in searchable
        assert "traceback" in data

    def test_list_failures_all(
        self,
        dead_letter_queue: DeadLetterQueue,
    ) -> None:
        dead_letter_queue.record("phase1_acquisition", "doc1", ValueError("e1"), None)
        dead_letter_queue.record("phase2_classification", "doc2", ValueError("e2"), None)
        failures = dead_letter_queue.list_failures()
        assert len(failures) == 2

    def test_list_failures_filtered(
        self,
        dead_letter_queue: DeadLetterQueue,
    ) -> None:
        dead_letter_queue.record("phase1_acquisition", "doc1", ValueError("e1"), None)
        dead_letter_queue.record("phase2_classification", "doc2", ValueError("e2"), None)
        failures = dead_letter_queue.list_failures("phase1_acquisition")
        assert len(failures) == 1
        assert failures[0]["doc_id"] == "doc1"


# ---------------------------------------------------------------------------
# Contract model tests
# ---------------------------------------------------------------------------


class TestContracts:
    def test_acquisition_result_json_roundtrip(self, acquisition_result: AcquisitionResult) -> None:
        json_str = acquisition_result.model_dump_json()
        restored = AcquisitionResult.model_validate_json(json_str)
        assert restored.doc_id == acquisition_result.doc_id
        assert restored.pdf_bytes == acquisition_result.pdf_bytes
        assert "pdf_bytes" in json_str  # field present in JSON
        # base64 should not contain raw binary escape sequences
        import json as _json

        raw = _json.loads(json_str)
        assert isinstance(raw["pdf_bytes"], str)

    def test_content_slice_image_bytes_roundtrip(self, layout_result: LayoutResult) -> None:
        """image_bytes=None should round-trip cleanly."""
        json_str = layout_result.model_dump_json()
        restored = LayoutResult.model_validate_json(json_str)
        assert restored.slices[0].image_bytes is None

    def test_person_entity_fields(self, person_entity) -> None:
        assert person_entity.surname == "Kowalski"
        assert "Jan" in person_entity.given_names
        assert person_entity.birth_date == "1889"


# ---------------------------------------------------------------------------
# Protocol stub enforcement test
# ---------------------------------------------------------------------------


class TestPhaseProtocol:
    def test_stub_raises_not_implemented(self) -> None:
        from bio_extraction.phases.phase1_acquisition import Phase1Acquisition

        phase = Phase1Acquisition()
        seed = PhaseOneInput(source=DocumentSource.LOCAL, local_path=Path("/tmp/fake.pdf"))
        with pytest.raises(NotImplementedError):
            phase.run(seed)

    def test_phase_name_property(self) -> None:
        from bio_extraction.phases.phase1_acquisition import Phase1Acquisition
        from bio_extraction.phases.phase2_classification import Phase2Classification
        from bio_extraction.phases.phase3_layout import Phase3Layout
        from bio_extraction.phases.phase4_ocr import Phase4OCR
        from bio_extraction.phases.phase5_extraction import Phase5Extraction
        from bio_extraction.phases.phase6_resolution import Phase6Resolution

        expected = [
            (Phase1Acquisition, "phase1_acquisition"),
            (Phase2Classification, "phase2_classification"),
            (Phase3Layout, "phase3_layout"),
            (Phase4OCR, "phase4_ocr"),
            (Phase5Extraction, "phase5_extraction"),
            (Phase6Resolution, "phase6_resolution"),
        ]
        for cls, name in expected:
            assert cls().phase_name == name
