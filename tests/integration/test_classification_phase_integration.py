import fitz
from bio_extraction.contracts import AcquisitionResult, DocumentType


class TestClassificationPhaseIntegration:
    def _create_test_pdf(self, content_lines: list, draw_lines: bool = False) -> bytes:
        """Helper to create a PDF in memory."""
        doc = fitz.open()
        page = doc.new_page()
        y = 50
        for line in content_lines:
            page.insert_text((50, y), line)
            y += 20

        if draw_lines:
            # Draw lines to simulate table or columns
            for i in range(20):
                page.draw_line(fitz.Point(10, i * 20), fitz.Point(500, i * 20))

        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes

    def test_integration_directory_classification(self, phase):
        # Create a PDF that looks like a Directory
        pdf_bytes = self._create_test_pdf(
            [
                "KOWALSKI, Jan ur. 1880 ul. Mazowiecka 10",
                "ZELIŃSKI, Adam zm. 1920 pl. Bankowy 1",
                "NOWAK, Maria *1895 al. Ujazdowskie 4",
            ]
        )

        input_data = AcquisitionResult(doc_id="dir_doc", pdf_bytes=pdf_bytes)
        result = phase.run(input_data)

        assert result is not None
        assert result.doc_type == DocumentType.DIRECTORY
        assert result.confidence >= 0.70

    def test_integration_civil_record_discard(self, phase):
        # Create a PDF that looks like a Civil Record (Table + Keywords)
        pdf_bytes = self._create_test_pdf(
            content_lines=["REJESTR ZGONÓW", "KSIĘGA MAŁŻEŃSTW"], draw_lines=True
        )

        input_data = AcquisitionResult(doc_id="civil_doc", pdf_bytes=pdf_bytes)
        result = phase.run(input_data)

        # Code is designed to return None for CIVIL_RECORD (unsupported)
        assert result is None

    def test_integration_unknown_discard(self, phase):
        # Create a blank-ish PDF
        pdf_bytes = self._create_test_pdf(["Hello world", "Not much here"])

        input_data = AcquisitionResult(doc_id="unknown_doc", pdf_bytes=pdf_bytes)
        result = phase.run(input_data)

        assert result is None
