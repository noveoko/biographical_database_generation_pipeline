import pytest
from pathlib import Path

from bio_extraction.phases.phase1_acquisition import (
    _compute_doc_id,
    _compute_lang_score,
    _slug_from_url,
    enumerate_local_inputs,
    _UNKNOWN_DOC_ID,
    _PHASE_NAME,
)
from bio_extraction.exceptions import AcquisitionError
from bio_extraction.contracts import DocumentSource


# ---------------------------
# _compute_doc_id
# ---------------------------


def test_compute_doc_id_deterministic():
    data = b"test pdf bytes"
    id1 = _compute_doc_id(data)
    id2 = _compute_doc_id(data)

    assert id1 == id2
    assert len(id1) == 16


def test_compute_doc_id_differs():
    assert _compute_doc_id(b"a") != _compute_doc_id(b"b")


# ---------------------------
# _compute_lang_score
# ---------------------------


def test_lang_score_polish_high():
    sample = "Pan Jan Kowalski ur w roku 1887 w Warszawie"
    score = _compute_lang_score(sample)
    assert score >= 0.3


def test_lang_score_english_low():
    sample = "The quick brown fox jumps over the lazy dog"
    score = _compute_lang_score(sample)
    assert score < 0.3


def test_lang_score_empty():
    assert _compute_lang_score("") == 0.0


# ---------------------------
# _slug_from_url
# ---------------------------


def test_slug_from_url_with_path():
    url = "https://example.com/files/doc.pdf"
    assert _slug_from_url(url) == "doc.pdf"


def test_slug_from_url_no_path():
    url = "https://example.com"
    assert _slug_from_url(url) == "example.com"


# ---------------------------
# enumerate_local_inputs
# ---------------------------


def test_enumerate_local_inputs(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"data")
    (tmp_path / "b.pdf").write_bytes(b"data")
    (tmp_path / "c.txt").write_text("ignore")

    inputs = enumerate_local_inputs(tmp_path)

    assert len(inputs) == 2
    assert all(i.source == DocumentSource.LOCAL for i in inputs)


def test_enumerate_local_inputs_invalid_dir():
    with pytest.raises(AcquisitionError) as exc:
        enumerate_local_inputs(Path("/nonexistent"))

    assert exc.value.phase_name == _PHASE_NAME
    assert exc.value.doc_id == _UNKNOWN_DOC_ID
