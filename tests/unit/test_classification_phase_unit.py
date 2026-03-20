import pytest
from unittest.mock import patch
from bio_extraction.contracts import AcquisitionResult, DocumentType
from bio_extraction.exceptions import PhaseError
from bio_extraction.phases.phase2_classification import (
    ClassificationPhase,
)  # Replace with actual module name


@pytest.fixture
def phase():
    return ClassificationPhase()


class TestClassificationPhaseUnit:
    @pytest.mark.parametrize(
        "num_pages, expected",
        [
            (1, [0]),
            (2, [0, 1]),
            (3, [0, 1, 2]),
            (10, [0, 5, 9]),
        ],
    )
    def test_get_sample_indices(self, phase, num_pages, expected):
        assert phase._get_sample_indices(num_pages) == expected

    def test_calculate_entry_density_matches(self, phase):
        text = "KOWALSKI, Jan ur. 1890 ul. Wiejska 5. NOWAK, Anna zm. 1950."
        # Matches: 2 surnames, 2 dates, 1 address = 5
        density = phase._calculate_entry_density(text)
        assert density == 5

    def test_calculate_entry_density_empty(self, phase):
        assert phase._calculate_entry_density("Random text with no patterns.") == 0

    @pytest.mark.parametrize(
        "density, cols, table, keywords, expected_type, expected_conf",
        [
            (6, 2, False, False, DocumentType.DIRECTORY, 0.95),
            (3, 1, False, False, DocumentType.DIRECTORY, 0.70),
            (0, 1, True, True, DocumentType.CIVIL_RECORD, 0.85),
            (1, 4, False, False, DocumentType.NEWSPAPER, 0.80),
            (0, 1, False, False, DocumentType.UNKNOWN, 0.0),
        ],
    )
    def test_score_heuristic(
        self, phase, density, cols, table, keywords, expected_type, expected_conf
    ):
        doc_type, conf = phase._score_heuristic(density, cols, table, keywords)
        assert doc_type == expected_type
        assert conf == expected_conf

    def test_run_empty_bytes_returns_none(self, phase):
        input_data = AcquisitionResult(doc_id="test_1", pdf_bytes=b"")
        assert phase.run(input_data) is None

    @patch("fitz.open")
    def test_run_corrupt_pdf_raises_phase_error(self, mock_fitz, phase):
        import fitz

        mock_fitz.side_effect = fitz.FileDataError("Bad PDF")
        input_data = AcquisitionResult(doc_id="test_err", pdf_bytes=b"garbage")

        with pytest.raises(PhaseError) as excinfo:
            phase.run(input_data)
        assert "Corrupt PDF data" in str(excinfo.value)
