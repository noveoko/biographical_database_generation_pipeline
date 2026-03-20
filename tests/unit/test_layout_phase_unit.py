import pytest
import numpy as np
from unittest.mock import MagicMock
from bio_extraction.contracts import DocumentType, ClassificationResult
from bio_extraction.phases.phase2_classification import ClassificationResult


@pytest.fixture
def layout_phase():
    return LayoutPhase()


def test_detect_columns_single_column(layout_phase):
    # Create a "blank" image (high values represent white/empty space in grayscale)
    img = np.full((100, 100), 255, dtype=np.uint8)
    # The current logic returns the whole width if no valleys are found
    columns = layout_phase._detect_columns(img)
    assert columns == [(0, 100)]


def test_segment_entries_basic(layout_phase):
    # Create an image with two horizontal black bars (entries)
    # 0 = black (ink), 255 = white (background)
    img = np.full((100, 10), 255, dtype=np.uint8)
    img[10:20, :] = 0  # Entry 1
    img[40:60, :] = 0  # Entry 2

    entries = layout_phase._segment_entries(img)
    # Expecting ranges roughly covering the "ink" blocks
    assert len(entries) >= 2
    assert entries[0][0] == 0
    assert entries[1][0] > 20


def test_run_skips_non_directory(layout_phase):
    doc = MagicMock(spec=ClassificationResult)
    doc.doc_type = DocumentType.OTHER  # Not a DIRECTORY
    result = layout_phase.run(doc)
    assert result is None
