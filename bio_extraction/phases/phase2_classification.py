# REQUIRES: pymupdf
import re
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

import fitz  # PyMuPDF

from bio_extraction.contracts import AcquisitionResult, ClassificationResult, DocumentType
from bio_extraction.protocol import PhaseProtocol
from bio_extraction.exceptions import PhaseError
from bio_extraction.logging_config import get_phase_logger
from bio_extraction.config import get_settings


class ClassificationPhase(PhaseProtocol[AcquisitionResult, ClassificationResult]):
    """
    Phase 2: Document Classification.

    Classifies an acquired PDF into DIRECTORY, NEWSPAPER, CIVIL_RECORD, or UNKNOWN
    using rule-based structural and textual heuristics on sampled pages.
    """

    def __init__(self) -> None:
        self._phase_name = "phase2_classification"
        self.logger = get_phase_logger(self.phase_name)
        self.settings = get_settings()

        # Pre-compile regexes for textual density analysis
        # SURNAME, Name (e.g., KOWALSKI, Jan)
        self.re_surname_name = re.compile(
            r"\b[A-ZĄĆĘŁŃÓŚŹŻ]{2,},\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\b"
        )
        # Dates (e.g., ur. 1890, zm. 1945, *1890, †1945)
        self.re_dates = re.compile(r"(?:ur\.|zm\.|[*†])\s*\d{4}")
        # Address fragments (e.g., ul., pl., al.)
        self.re_addresses = re.compile(r"\b(?:ul\.|pl\.|al\.)\b", re.IGNORECASE)
        # Section keywords for civil records
        self.re_keywords = re.compile(
            r"\b(?:SPIS|KSIĘGA|REJESTR|ZGONÓW|URODZEŃ|MAŁŻEŃSTW)\b", re.IGNORECASE
        )

    @property
    def phase_name(self) -> str:
        """Unique snake_case identifier for this phase."""
        return self._phase_name

    def run(self, input_data: AcquisitionResult) -> ClassificationResult | None:
        """
        Executes the classification phase on a single acquired document.

        Args:
            input_data: The AcquisitionResult containing the PDF bytes and metadata.

        Returns:
            A ClassificationResult if classified as DIRECTORY, otherwise None.

        Raises:
            PhaseError: If the PDF is inherently corrupt or cannot be parsed.
        """
        self.logger.info(f"Starting classification for document: {input_data.doc_id}")

        if not input_data.pdf_bytes:
            self.logger.warning(f"Discarding {input_data.doc_id}: PDF bytes are empty.")
            return None

        try:
            doc = fitz.open(stream=input_data.pdf_bytes, filetype="pdf")
        except fitz.FileDataError as e:
            raise PhaseError(self.phase_name, input_data.doc_id, f"Corrupt PDF data: {e}") from e
        except Exception as e:
            raise PhaseError(
                self.phase_name, input_data.doc_id, f"Unexpected error opening PDF: {e}"
            ) from e

        num_pages = len(doc)
        if num_pages == 0:
            self.logger.warning(f"Discarding {input_data.doc_id}: PDF has 0 pages.")
            doc.close()
            return None

        sampled_indices = self._get_sample_indices(num_pages)
        self.logger.debug(f"Sampled page indices: {sampled_indices}")

        # Aggregate metrics across all sampled pages
        total_entry_density = 0
        max_columns = 1
        has_table = False
        has_civil_keywords = False

        for page_idx in sampled_indices:
            page = doc[page_idx]
            features = self._extract_features_from_page(page)

            total_entry_density += features["entry_density"]
            max_columns = max(max_columns, features["column_count"])
            if features["table_structure"]:
                has_table = True
            if features["keywords_present"]:
                has_civil_keywords = True

        doc.close()

        # Average entry density across sampled pages
        avg_entry_density = total_entry_density / len(sampled_indices)

        self.logger.debug(
            f"Doc {input_data.doc_id} features - Avg density: {avg_entry_density:.2f}, "
            f"Max cols: {max_columns}, Table: {has_table}, Civil kws: {has_civil_keywords}"
        )

        doc_type, confidence = self._score_heuristic(
            avg_entry_density, max_columns, has_table, has_civil_keywords
        )

        self.logger.info(
            f"Document {input_data.doc_id} classified as {doc_type.value} with {confidence} confidence."
        )

        # Discard rules based on MVP specification
        if doc_type == DocumentType.UNKNOWN:
            self.logger.info(f"Discarding {input_data.doc_id}: Classified as UNKNOWN.")
            return None

        if doc_type in (DocumentType.NEWSPAPER, DocumentType.CIVIL_RECORD):
            self.logger.warning(
                f"Discarding {input_data.doc_id}: Document type '{doc_type.value}' is not yet supported."
            )
            return None

        return ClassificationResult(
            doc_id=input_data.doc_id,
            doc_type=doc_type,
            confidence=confidence,
            sample_page_indices=sampled_indices,
            classified_at=datetime.now(timezone.utc),
        )

    def _get_sample_indices(self, num_pages: int) -> List[int]:
        """Returns up to 3 page indices: first, middle, last."""
        if num_pages == 1:
            return [0]
        elif num_pages == 2:
            return [0, 1]
        else:
            return [0, num_pages // 2, num_pages - 1]

    def _extract_features_from_page(self, page: fitz.Page) -> Dict[str, Any]:
        """Extracts structural and textual features from a single PyMuPDF page."""
        text = page.get_text()

        return {
            "column_count": self._estimate_column_count(page),
            "entry_density": self._calculate_entry_density(text),
            "table_structure": self._detect_table_structure(page),
            "keywords_present": bool(self.re_keywords.search(text)),
        }

    def _estimate_column_count(self, page: fitz.Page) -> int:
        """
        Estimates column count using a vertical projection profile.
        Renders page to 150 DPI grayscale image and counts peak density regions.
        """
        try:
            pix = page.get_pixmap(dpi=150, colorspace=fitz.csGRAY)
            width, height = pix.width, pix.height
            samples = pix.samples

            col_sums = [0] * width

            # Sum ink density per column (0 = black ink, 255 = white background)
            for y in range(height):
                row_offset = y * width
                for x in range(width):
                    col_sums[x] += 255 - samples[row_offset + x]

            if not col_sums:
                return 1

            max_sum = max(col_sums)
            # Threshold to consider a column "ink-heavy" vs a "gutter"
            threshold = max_sum * 0.15

            columns = 0
            in_column = False

            for val in col_sums:
                if val > threshold:
                    if not in_column:
                        in_column = True
                        columns += 1
                else:
                    in_column = False

            return max(1, columns)

        except Exception as e:
            self.logger.warning(f"Column estimation failed on page {page.number}: {e}")
            return 1

    def _calculate_entry_density(self, text: str) -> int:
        """Counts occurrences of biographical directory patterns."""
        surname_matches = len(self.re_surname_name.findall(text))
        date_matches = len(self.re_dates.findall(text))
        address_matches = len(self.re_addresses.findall(text))

        return surname_matches + date_matches + address_matches

    def _detect_table_structure(self, page: fitz.Page) -> bool:
        """Detects tables by counting orthogonal line/rect drawings on the page."""
        drawings = page.get_drawings()
        line_segments = 0

        for d in drawings:
            for item in d.get("items", []):
                if item[0] in ("l", "re"):  # 'l' = line, 're' = rectangle
                    line_segments += 1

        # Arbitrary threshold to differentiate tables from basic design lines
        return line_segments > 15

    def _score_heuristic(
        self, avg_density: float, max_cols: int, has_table: bool, has_civil_keywords: bool
    ) -> Tuple[DocumentType, float]:
        """
        Applies rules to determine document type and confidence.
        """
        if avg_density > 5 and max_cols >= 2:
            return DocumentType.DIRECTORY, 0.95

        if avg_density > 2:
            return DocumentType.DIRECTORY, 0.70

        if has_table and has_civil_keywords:
            return DocumentType.CIVIL_RECORD, 0.85

        if max_cols >= 3 and avg_density <= 2:
            return DocumentType.NEWSPAPER, 0.80

        return DocumentType.UNKNOWN, 0.0


# ==============================================================================
# Smoke Test Block
# ==============================================================================
if __name__ == "__main__":
    import sys
    import enum
    from dataclasses import dataclass
    from datetime import datetime

    # 1. Mock the specific bio_extraction modules layout
    class MockConfig:
        def __getattr__(self, name):
            return None

    class MockLogger:
        def info(self, msg):
            print(f"INFO: {msg}")

        def warning(self, msg):
            print(f"WARN: {msg}")

        def debug(self, msg):
            print(f"DEBUG: {msg}")

        def error(self, msg):
            print(f"ERROR: {msg}")

    # Enums matching contracts.py
    class _DemoDocumentType(enum.Enum):
        DIRECTORY = "directory"
        NEWSPAPER = "newspaper"
        CIVIL_RECORD = "civil_record"
        UNKNOWN = "unknown"

    @dataclass
    class MockAcquisitionResult:
        doc_id: str
        pdf_bytes: bytes

    @dataclass
    class MockClassificationResult:
        doc_id: str
        doc_type: DocumentType
        confidence: float
        sample_page_indices: List[int]
        classified_at: datetime

    class MockPhaseProtocol:
        @property
        def phase_name(self):
            return "mock_phase"

        def run(self, input_data):
            pass

    class MockPhaseError(Exception):
        def __init__(self, phase_name: str, doc_id: str, message: str) -> None:
            self.phase_name = phase_name
            self.doc_id = doc_id
            super().__init__(f"[{phase_name}] doc={doc_id}: {message}")

    # Injecting mocks into sys.modules
    sys.modules["bio_extraction"] = type("MockPackage", (), {})()
    sys.modules["bio_extraction.contracts"] = type(
        "MockContracts",
        (),
        {
            "AcquisitionResult": MockAcquisitionResult,
            "ClassificationResult": MockClassificationResult,
            "DocumentType": DocumentType,
        },
    )()
    sys.modules["bio_extraction.protocol"] = type(
        "MockProtocol", (), {"PhaseProtocol": MockPhaseProtocol}
    )()
    sys.modules["bio_extraction.exceptions"] = type(
        "MockExceptions", (), {"PhaseError": MockPhaseError}
    )()
    sys.modules["bio_extraction.logging_config"] = type(
        "MockLog", (), {"get_phase_logger": lambda name: MockLogger()}
    )()
    sys.modules["bio_extraction.config"] = type(
        "MockConf", (), {"get_settings": lambda: MockConfig()}
    )()

    # Apply to globals so the script parses successfully when run natively
    globals()["AcquisitionResult"] = MockAcquisitionResult
    globals()["ClassificationResult"] = MockClassificationResult
    globals()["DocumentType"] = DocumentType
    globals()["PhaseProtocol"] = MockPhaseProtocol
    globals()["PhaseError"] = MockPhaseError

    print("--- Running Smoke Test ---")

    # Create a minimal valid synthetic PDF using fitz
    test_doc = fitz.open()
    page = test_doc.new_page()
    page.insert_text((50, 50), "KOWALSKI, Jan ur. 1890 ul. Warszawska 1")
    page.insert_text((50, 70), "NOWAK, Piotr zm. 1945 pl. Wolności 2")
    page.insert_text((50, 90), "WIŚNIEWSKI, Adam *1900 al. Jerozolimskie")
    # Draw a line down the middle to simulate 2 columns
    page.draw_line(fitz.Point(300, 0), fitz.Point(300, 800))
    pdf_bytes = test_doc.write()
    test_doc.close()

    acq_data = MockAcquisitionResult(doc_id="doc_123_test", pdf_bytes=pdf_bytes)

    # Run Phase
    phase = ClassificationPhase()
    result = phase.run(acq_data)  # type: ignore[arg-type]

    print("\n--- Output ---")
    if result:
        print(f"Success! Document ID: {result.doc_id}")
        print(f"Type: {result.doc_type.value}")
        print(f"Confidence: {result.confidence}")
        print(f"Sampled Pages: {result.sample_page_indices}")
        print(f"Classified At: {result.classified_at.isoformat()}")
    else:
        print("Document discarded (returned None).")

# Public alias used by test_e2e_pipeline.py
Phase2Classification = ClassificationPhase
