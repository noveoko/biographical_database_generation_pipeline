import pytest
import fitz
from datetime import datetime
from unittest.mock import patch, MagicMock

from bio_extraction.contracts import (
    AcquisitionResult,
    ClassificationResult,  # <-- added
    LayoutResult,
    DocumentType,
)
from bio_extraction.exceptions import PhaseError


@patch("bio_extraction.checkpoint.CheckpointEngine.load")
def test_layout_phase_integration_success(mock_checkpoint_load, layout_phase):
    # 1. Setup a dummy PDF in memory
    pdf_doc = fitz.open()
    page = pdf_doc.new_page()
    page.insert_text((72, 72), "Sample Entry Text")
    pdf_bytes = pdf_doc.write()

    # 2. Setup Mocks
    doc_input = ClassificationResult(
        doc_id="test_123",
        doc_type=DocumentType.DIRECTORY,
        confidence=0.95,
        sample_page_indices=[0],
        classified_at=datetime.utcnow(),
    )

    mock_checkpoint_load.return_value = AcquisitionResult(
        doc_id="test_123",
        pdf_bytes=pdf_bytes,
        acquired_at=datetime.utcnow(),
        source_url="internal://test",
    )

    # 3. Run Phase
    result = layout_phase.run(doc_input)

    # 4. Assertions
    assert isinstance(result, LayoutResult)
    assert len(result.slices) > 0
    assert result.doc_id == "test_123"
    assert "test_123_p0_e0" in result.slices[0].slice_id


@patch("bio_extraction.checkpoint.CheckpointEngine.load")
def test_layout_phase_raises_error_on_missing_checkpoint(mock_checkpoint_load, layout_phase):
    mock_checkpoint_load.side_effect = Exception("Not found")

    doc_input = MagicMock(spec=ClassificationResult)
    doc_input.doc_type = DocumentType.DIRECTORY
    doc_input.doc_id = "fail_id"

    with pytest.raises(PhaseError) as excinfo:
        layout_phase.run(doc_input)

    assert "Checkpoint load failed" in str(excinfo.value)
