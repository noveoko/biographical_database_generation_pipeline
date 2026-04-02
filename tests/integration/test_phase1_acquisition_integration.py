import pytest
import fitz
from pathlib import Path
from datetime import datetime

from bio_extraction.phases.phase1_acquisition import AcquisitionPhase, enumerate_local_inputs
from bio_extraction.contracts import PhaseOneInput, DocumentSource


# ---------------------------
# Helpers
# ---------------------------


def make_pdf(text: str, path: Path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


class FakeSettings:
    lang_score_threshold = 0.3
    cc_request_timeout = 5


# ---------------------------
# Fixture: patch settings
# ---------------------------


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    import bio_extraction.config as cfg

    monkeypatch.setattr(cfg, "get_settings", lambda: FakeSettings())


# ---------------------------
# LOCAL MODE
# ---------------------------


def test_local_pdf_success(tmp_path):
    pdf_path = tmp_path / "polish.pdf"
    make_pdf("Pan Jan Kowalski ur w roku 1887", pdf_path)

    phase = AcquisitionPhase()
    result = phase.run(PhaseOneInput(source=DocumentSource.LOCAL, local_path=pdf_path))

    assert result is not None
    assert result.source == DocumentSource.LOCAL
    assert result.filename == "polish.pdf"
    assert isinstance(result.acquired_at, datetime)
    assert len(result.doc_id) == 16


def test_local_pdf_filtered(tmp_path):
    pdf_path = tmp_path / "english.pdf"
    make_pdf("This is purely English text", pdf_path)

    phase = AcquisitionPhase()
    result = phase.run(PhaseOneInput(source=DocumentSource.LOCAL, local_path=pdf_path))

    assert result is None


def test_local_file_not_found():
    phase = AcquisitionPhase()

    with pytest.raises(Exception):
        phase.run(
            PhaseOneInput(
                source=DocumentSource.LOCAL,
                local_path=Path("/no/file.pdf"),
            )
        )


# ---------------------------
# COMMONCRAWL MODE (mocked)
# ---------------------------


def test_commoncrawl_success(monkeypatch):
    phase = AcquisitionPhase()

    # Fake PDF bytes
    fake_pdf = fitz.open()
    page = fake_pdf.new_page()
    page.insert_text((72, 72), "Pan Kowalski ur w roku 1887")
    pdf_bytes = fake_pdf.write()
    fake_pdf.close()

    # Mock fetch + extract
    monkeypatch.setattr(
        "bio_extraction.phases.phase1_acquisition._fetch_warc_bytes",
        lambda *args, **kwargs: b"warc_bytes",
    )

    monkeypatch.setattr(
        "bio_extraction.phases.phase1_acquisition._extract_pdf_from_warc",
        lambda _: (pdf_bytes, "warc-id-123"),
    )

    input_data = PhaseOneInput(
        source=DocumentSource.COMMONCRAWL,
        cc_warc_record={
            "url": "https://example.com/doc.pdf",
            "offset": 0,
            "length": 100,
            "warc_filename": "file.warc.gz",
        },
    )

    result = phase.run(input_data)

    assert result is not None
    assert result.source == DocumentSource.COMMONCRAWL
    assert result.warc_id == "warc-id-123"
    assert result.source_url == "https://example.com/doc.pdf"


def test_commoncrawl_missing_key():
    phase = AcquisitionPhase()

    bad_input = PhaseOneInput(
        source=DocumentSource.COMMONCRAWL,
        cc_warc_record={"url": "x"},  # missing fields
    )

    with pytest.raises(Exception):
        phase.run(bad_input)


def test_corrupt_pdf(tmp_path):
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not a pdf")

    phase = AcquisitionPhase()

    with pytest.raises(Exception):
        phase.run(PhaseOneInput(source=DocumentSource.LOCAL, local_path=bad_pdf))


def test_threshold_boundary(tmp_path):
    pdf_path = tmp_path / "border.pdf"
    make_pdf("Pan ur w roku", pdf_path)

    class Settings:
        lang_score_threshold = 0.0

    # ensure everything passes


def test_enumerate_sorted(tmp_path):
    (tmp_path / "b.pdf").write_bytes(b"x")
    (tmp_path / "a.pdf").write_bytes(b"x")

    inputs = enumerate_local_inputs(tmp_path)
    names = [i.local_path.name for i in inputs]

    assert names == ["a.pdf", "b.pdf"]
