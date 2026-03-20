# REQUIRES: pymupdf, warcio, requests

"""
bio_extraction/phases/phase1_acquisition.py
============================================
Phase 1 — Acquisition & Filtering

Responsibilities
----------------
* Accept a ``PhaseOneInput`` (local PDF path **or** CommonCrawl WARC record
  pointer) and return a populated ``AcquisitionResult``, or ``None`` to
  discard the document.
* Run a lightweight Polish-language heuristic on the first 500 chars of
  extracted text; documents that score below the configured threshold are
  silently dropped (``run()`` returns ``None`` — the runner logs the discard
  and does not enqueue the document downstream).
* Expose two helper functions used by the runner to seed the Phase-1 queue:
  ``enumerate_local_inputs`` and ``enumerate_commoncrawl_inputs``.

Design notes
------------
* All I/O is synchronous.  The runner is responsible for concurrency.
* SHA-256 is truncated to 16 hex characters as the ``doc_id`` — enough
  entropy for deduplication within a single pipeline run (~2^64 space).
* The lang-score formula is:
      lang_score = (char_density × 0.6) + (word_density × 0.4)
  Both sub-scores are individually clamped to [0, 1] before weighting,
  so the composite score is always in [0.0, 1.0].
* ``AcquisitionError`` inherits from ``PhaseError``, which requires
  ``(phase_name, doc_id, message)``.  When an error occurs before the
  ``doc_id`` has been computed (e.g. file-not-found, network failure),
  the sentinel string ``"unknown"`` is used as the ``doc_id`` argument.
* ``_extract_pdf_from_warc`` returns a ``(pdf_bytes, warc_record_id)`` tuple
  so the WARC-Record-ID header value can be stored in ``AcquisitionResult``.
"""

from __future__ import annotations

import hashlib
import io
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse

import fitz  # PyMuPDF
import requests
from warcio.archiveiterator import ArchiveIterator

from bio_extraction.contracts import AcquisitionResult, DocumentSource, PhaseOneInput
from bio_extraction.exceptions import AcquisitionError
from bio_extraction.logging_config import get_phase_logger
from bio_extraction.config import get_settings
from bio_extraction.protocol import PhaseProtocol

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Phase identifier — kept as a module constant so standalone helper functions
# can construct ``AcquisitionError`` without needing a phase instance.
_PHASE_NAME: str = "phase1_acquisition"

_POLISH_CHARS: frozenset[str] = frozenset("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

# Common Polish function-words and genealogical abbreviations, matched as
# whole tokens (word-boundary anchored) to avoid false-positive hits inside
# longer foreign words.
_POLISH_WORD_PATTERN: re.Pattern[str] = re.compile(
    r"\b(i|w|z|na|do|od|pan|pani|roku|ur|zm|syn|córka|małż|wdowa)\b",
    re.IGNORECASE,
)

# Number of characters examined by the language heuristic.
_HEURISTIC_SAMPLE_SIZE: int = 500

# Sentinel used as ``doc_id`` in ``AcquisitionError`` when the hash has not
# been computed yet (e.g. file-not-found, network failure before PDF bytes
# are available).
_UNKNOWN_DOC_ID: str = "unknown"


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------


def _compute_doc_id(pdf_bytes: bytes) -> str:
    """Return the first 16 hex characters of the SHA-256 digest of *pdf_bytes*.

    Parameters
    ----------
    pdf_bytes:
        Raw bytes of the PDF file.

    Returns
    -------
    str
        A 16-character lowercase hex string, e.g. ``"3a9f1c02b8e47d05"``.
        Provides ~2^64 collision resistance — sufficient for pipeline-run
        deduplication without the storage overhead of a full 64-char digest.
    """
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def _extract_sample_text(
    pdf_bytes: bytes,
    doc_id: str,
    sample_size: int = _HEURISTIC_SAMPLE_SIZE,
) -> str:
    """Open a PDF from *pdf_bytes* and extract up to *sample_size* characters.

    Only page 0 is consulted for speed; the assumption is that the opening
    page of a biographical directory entry carries the most representative
    language signal.

    Parameters
    ----------
    pdf_bytes:
        Raw bytes of a valid PDF.
    doc_id:
        Pipeline document identifier, threaded through for error reporting.
    sample_size:
        Maximum number of characters to return.

    Returns
    -------
    str
        Extracted plain text, potentially shorter than *sample_size* if the
        first page contains fewer characters.

    Raises
    ------
    AcquisitionError
        If PyMuPDF cannot open the document (corrupt or unsupported format).
    """
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.page_count == 0:
                return ""
            text: str = doc.load_page(0).get_text("text")
    except Exception as exc:
        raise AcquisitionError(
            _PHASE_NAME, doc_id, f"PyMuPDF could not open document: {exc}"
        ) from exc

    return text[:sample_size]


def _compute_lang_score(sample: str) -> float:
    """Estimate the probability that *sample* is Polish text.

    The score is a weighted combination of two sub-signals, each normalised
    to ``[0, 1]`` independently before combining:

    1. **Polish-char density** (weight 0.6):
       Fraction of characters in *sample* that belong to the set of diacritic
       letters exclusive to Polish (ą, ć, ę, ł, ń, ó, ś, ź, ż), normalised
       against the expected density in typical Polish prose (~6 %).

    2. **Polish-word density** (weight 0.4):
       Fraction of word tokens matching common Polish function-words and
       genealogical abbreviations, normalised against the expected rate in
       genealogical text (~20 %).

    Parameters
    ----------
    sample:
        The text sample to score (usually the first 500 characters of a PDF).

    Returns
    -------
    float
        A composite score in ``[0.0, 1.0]``.  Values ≥ 0.3 pass the default
        Polish-language filter.
    """
    if not sample:
        return 0.0

    # Sub-score 1: diacritic-character density.
    # Polish prose contains ~6 % diacritic characters; normalising by that
    # baseline means "typical Polish" gives a sub-score close to 1.0.
    polish_char_count = sum(1 for ch in sample if ch in _POLISH_CHARS)
    char_density = min(polish_char_count / max(len(sample), 1) / 0.06, 1.0)

    # Sub-score 2: Polish function-word / abbreviation density.
    words = sample.split()
    if words:
        matched = len(_POLISH_WORD_PATTERN.findall(sample))
        word_density = min(matched / len(words) / 0.20, 1.0)
    else:
        word_density = 0.0

    return round(char_density * 0.6 + word_density * 0.4, 4)


def _slug_from_url(url: str) -> str:
    """Derive a short, filesystem-safe slug from a URL for use as ``filename``.

    Takes the last non-empty path segment, falling back to the netloc if the
    path is empty.

    Parameters
    ----------
    url:
        Any HTTP/HTTPS URL string.

    Returns
    -------
    str
        A non-empty slug, e.g. ``"kowalski_1887.pdf"`` or ``"polona.pl"``.
    """
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    return segments[-1] if segments else (parsed.netloc or url)


def _fetch_warc_bytes(
    warc_filename: str,
    offset: int,
    length: int,
    *,
    timeout: int = 30,
) -> bytes:
    """Fetch a slice of a CommonCrawl WARC file via an HTTP range request.

    Parameters
    ----------
    warc_filename:
        Path component of the WARC, e.g.
        ``"crawl-data/CC-MAIN-2024-10/segments/.../warc/..."``
    offset:
        Byte offset into the WARC file where the record begins.
    length:
        Byte length of the record.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    bytes
        Raw WARC-record bytes (may be gzip-compressed).

    Raises
    ------
    AcquisitionError
        On any network or HTTP error.  ``doc_id`` is ``_UNKNOWN_DOC_ID``
        because PDF hashing has not occurred yet at this stage.
    """
    url = f"https://data.commoncrawl.org/{warc_filename}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise AcquisitionError(
            _PHASE_NAME,
            _UNKNOWN_DOC_ID,
            f"Network failure fetching WARC record from {url}: {exc}",
        ) from exc

    if response.status_code not in (200, 206):
        raise AcquisitionError(
            _PHASE_NAME,
            _UNKNOWN_DOC_ID,
            f"CommonCrawl HTTP {response.status_code} for {url}",
        )

    return response.content


def _extract_pdf_from_warc(warc_bytes: bytes) -> tuple[bytes, str | None]:
    """Parse a WARC segment and return the PDF payload plus the WARC-Record-ID.

    Iterates over records in *warc_bytes* using ``warcio`` and returns the
    HTTP response body of the first record whose ``Content-Type`` indicates
    a PDF, together with the value of the ``WARC-Record-ID`` header.

    Parameters
    ----------
    warc_bytes:
        Raw bytes of a (possibly gzip-compressed) WARC segment.

    Returns
    -------
    tuple[bytes, str | None]
        ``(pdf_bytes, warc_record_id)`` where *warc_record_id* is the raw
        header value (e.g. ``"<urn:uuid:…>"``), or ``None`` if absent.

    Raises
    ------
    AcquisitionError
        If no PDF-bearing response record is found in the segment.
    """
    stream = io.BytesIO(warc_bytes)
    for record in ArchiveIterator(stream):
        if record.rec_type != "response":
            continue
        content_type: str = (
            record.http_headers.get_header("Content-Type", "") if record.http_headers else ""
        )
        if "pdf" not in content_type.lower():
            continue

        warc_record_id: str | None = record.rec_headers.get_header("WARC-Record-ID", None)
        return record.content_stream().read(), warc_record_id

    raise AcquisitionError(
        _PHASE_NAME,
        _UNKNOWN_DOC_ID,
        "No PDF response record found in WARC segment",
    )


# ---------------------------------------------------------------------------
# Public enumeration helpers (called by the runner to seed the queue)
# ---------------------------------------------------------------------------


def enumerate_local_inputs(input_dir: Path) -> list[PhaseOneInput]:
    """Walk *input_dir* (non-recursively) and return one ``PhaseOneInput`` per PDF.

    Results are sorted by filename for deterministic ordering across runs.

    Parameters
    ----------
    input_dir:
        Directory to scan for ``*.pdf`` files.

    Returns
    -------
    list[PhaseOneInput]
        One entry per ``.pdf`` file discovered.

    Raises
    ------
    AcquisitionError
        If *input_dir* does not exist or is not a directory.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise AcquisitionError(
            _PHASE_NAME,
            _UNKNOWN_DOC_ID,
            f"Input directory not found or not a directory: {input_dir}",
        )

    pdf_files = sorted(input_dir.glob("*.pdf"))
    return [PhaseOneInput(source=DocumentSource.LOCAL, local_path=p) for p in pdf_files]


def enumerate_commoncrawl_inputs(settings) -> list[PhaseOneInput]:
    """Query the CommonCrawl CDX Index API and return ``PhaseOneInput`` records.

    Pagination is handled transparently: the CDX API is polled in pages of
    500 records until no further results are returned or
    ``settings.cc_max_records`` is reached.

    Expected settings attributes
    ----------------------------
    ``cc_index_id`` : str
        CC crawl index to query, e.g. ``"CC-MAIN-2024-10"``.
    ``cc_domains`` : list[str]
        Domain patterns to search (e.g. ``["polona.pl", "pbc.gda.pl"]``).
    ``cc_request_timeout`` : int
        Per-request HTTP timeout in seconds (optional, default 30).
    ``cc_max_records`` : int | None
        Hard cap on the number of inputs returned (optional, default None).

    Parameters
    ----------
    settings:
        The application settings object returned by ``get_settings()``.

    Returns
    -------
    list[PhaseOneInput]
        One entry per PDF record found, capped at ``settings.cc_max_records``
        if that attribute is set.

    Raises
    ------
    AcquisitionError
        On network or CDX API errors.
    """
    import json

    logger = get_phase_logger(_PHASE_NAME)
    base_url = f"https://index.commoncrawl.org/{settings.cc_index_id}-index"
    timeout: int = getattr(settings, "cc_request_timeout", 30)
    max_records: int | None = getattr(settings, "cc_max_records", None)
    domains: list[str] = settings.cc_domains

    inputs: list[PhaseOneInput] = []

    for domain in domains:
        page = 0
        while True:
            params: dict[str, str | int] = {
                "url": f"{domain}/*",
                "output": "json",
                "filter": "mime:application/pdf",
                "fl": "url,offset,length,filename",
                "pageSize": 500,
                "page": page,
            }
            query_url = f"{base_url}?{urlencode(params)}"
            logger.debug("Fetching CDX page %d for domain %s", page, domain)

            try:
                resp = requests.get(query_url, timeout=timeout)
            except requests.RequestException as exc:
                raise AcquisitionError(
                    _PHASE_NAME,
                    _UNKNOWN_DOC_ID,
                    f"CDX API network failure for domain {domain}: {exc}",
                ) from exc

            if resp.status_code == 404:
                break  # CC returns 404 when page index is out of range
            if resp.status_code != 200:
                raise AcquisitionError(
                    _PHASE_NAME,
                    _UNKNOWN_DOC_ID,
                    f"CDX API HTTP {resp.status_code} for domain {domain}",
                )

            lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
            if not lines:
                break  # exhausted

            for raw_line in lines:
                try:
                    rec = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed CDX line: %s", raw_line[:120])
                    continue

                inputs.append(
                    PhaseOneInput(
                        source=DocumentSource.COMMONCRAWL,
                        cc_warc_record={
                            "url": rec.get("url", ""),
                            "offset": int(rec.get("offset", 0)),
                            "length": int(rec.get("length", 0)),
                            "warc_filename": rec.get("filename", ""),
                        },
                    )
                )

                if max_records is not None and len(inputs) >= max_records:
                    logger.info("Reached cc_max_records limit (%d); stopping.", max_records)
                    return inputs

            page += 1

    logger.info(
        "enumerate_commoncrawl_inputs: %d record(s) across %d domain(s).",
        len(inputs),
        len(domains),
    )
    return inputs


# ---------------------------------------------------------------------------
# Phase class
# ---------------------------------------------------------------------------


class AcquisitionPhase(PhaseProtocol[PhaseOneInput, AcquisitionResult]):
    """Phase 1 — acquire and language-filter a single PDF document.

    Supports two source modes driven by ``PhaseOneInput.source``:

    * ``DocumentSource.LOCAL``        — read the PDF from the local filesystem.
    * ``DocumentSource.COMMONCRAWL``  — fetch the PDF from a CommonCrawl WARC
      record via an HTTP range request.

    In both cases the same Polish-language heuristic is applied, and documents
    that fail the threshold are discarded (``run()`` returns ``None``).

    Usage
    -----
    >>> phase = AcquisitionPhase()
    >>> inp = PhaseOneInput(source=DocumentSource.LOCAL, local_path=Path("doc.pdf"))
    >>> result = phase.run(inp)
    >>> if result is not None:
    ...     print(result.doc_id, result.lang_score)
    """

    @property
    def phase_name(self) -> str:
        """Unique snake_case identifier for this phase."""
        return _PHASE_NAME

    def __init__(self) -> None:
        self._settings = get_settings()
        self._logger = get_phase_logger(self.phase_name)

    # ------------------------------------------------------------------
    # PhaseProtocol interface
    # ------------------------------------------------------------------

    def run(self, input_data: PhaseOneInput) -> AcquisitionResult | None:
        """Process one document input and return an ``AcquisitionResult`` or ``None``.

        Parameters
        ----------
        input_data:
            A ``PhaseOneInput`` carrying either ``local_path`` (for
            ``DocumentSource.LOCAL``) or ``cc_warc_record`` (for
            ``DocumentSource.COMMONCRAWL``).

        Returns
        -------
        AcquisitionResult | None
            Populated result on success; ``None`` if the document is discarded
            because its ``lang_score`` falls below the configured threshold.

        Raises
        ------
        AcquisitionError
            On file-not-found, read failure, network failure, corrupt PDF, or
            missing/invalid ``cc_warc_record`` keys.
        """
        lang_threshold: float = getattr(self._settings, "lang_score_threshold", 0.3)

        if input_data.source is DocumentSource.LOCAL:
            return self._run_local(input_data, lang_threshold)
        elif input_data.source is DocumentSource.COMMONCRAWL:
            return self._run_commoncrawl(input_data, lang_threshold)
        else:
            raise AcquisitionError(
                self.phase_name,
                _UNKNOWN_DOC_ID,
                f"Unknown DocumentSource value: {input_data.source!r}",
            )

    # ------------------------------------------------------------------
    # Private acquisition paths
    # ------------------------------------------------------------------

    def _run_local(
        self,
        input_data: PhaseOneInput,
        lang_threshold: float,
    ) -> AcquisitionResult | None:
        """Handle the local-filesystem acquisition path.

        Parameters
        ----------
        input_data:
            Must have ``local_path`` set to a valid ``Path``.
        lang_threshold:
            Minimum ``lang_score`` required to pass the filter.

        Returns
        -------
        AcquisitionResult | None
        """
        if input_data.local_path is None:
            raise AcquisitionError(
                self.phase_name,
                _UNKNOWN_DOC_ID,
                "local mode requires 'local_path' on PhaseOneInput",
            )

        path: Path = input_data.local_path
        self._logger.debug("Acquiring local PDF: %s", path)

        if not path.exists():
            raise AcquisitionError(
                self.phase_name,
                _UNKNOWN_DOC_ID,
                f"PDF file not found: {path}",
            )
        if not path.is_file():
            raise AcquisitionError(
                self.phase_name,
                _UNKNOWN_DOC_ID,
                f"Path is not a regular file: {path}",
            )

        try:
            pdf_bytes = path.read_bytes()
        except OSError as exc:
            raise AcquisitionError(
                self.phase_name,
                _UNKNOWN_DOC_ID,
                f"Could not read file {path}: {exc}",
            ) from exc

        return self._process_pdf_bytes(
            pdf_bytes=pdf_bytes,
            source=DocumentSource.LOCAL,
            source_url=None,
            filename=path.name,
            warc_id=None,
            lang_threshold=lang_threshold,
        )

    def _run_commoncrawl(
        self,
        input_data: PhaseOneInput,
        lang_threshold: float,
    ) -> AcquisitionResult | None:
        """Handle the CommonCrawl WARC acquisition path.

        Parameters
        ----------
        input_data:
            Must have ``cc_warc_record`` set with keys
            ``url``, ``offset``, ``length``, ``warc_filename``.
        lang_threshold:
            Minimum ``lang_score`` required to pass the filter.

        Returns
        -------
        AcquisitionResult | None
        """
        if input_data.cc_warc_record is None:
            raise AcquisitionError(
                self.phase_name,
                _UNKNOWN_DOC_ID,
                "commoncrawl mode requires 'cc_warc_record' on PhaseOneInput",
            )

        rec = input_data.cc_warc_record
        for key in ("url", "offset", "length", "warc_filename"):
            if key not in rec:
                raise AcquisitionError(
                    self.phase_name,
                    _UNKNOWN_DOC_ID,
                    f"cc_warc_record is missing required key: '{key}'",
                )

        self._logger.debug(
            "Fetching WARC record: %s @ offset=%s length=%s",
            rec["warc_filename"],
            rec["offset"],
            rec["length"],
        )

        warc_bytes = _fetch_warc_bytes(
            warc_filename=rec["warc_filename"],
            offset=int(rec["offset"]),
            length=int(rec["length"]),
            timeout=getattr(self._settings, "cc_request_timeout", 30),
        )

        pdf_bytes, warc_record_id = _extract_pdf_from_warc(warc_bytes)
        source_url: str = rec["url"]

        return self._process_pdf_bytes(
            pdf_bytes=pdf_bytes,
            source=DocumentSource.COMMONCRAWL,
            source_url=source_url,
            filename=_slug_from_url(source_url),
            warc_id=warc_record_id,
            lang_threshold=lang_threshold,
        )

    # ------------------------------------------------------------------
    # Shared core pipeline
    # ------------------------------------------------------------------

    def _process_pdf_bytes(
        self,
        *,
        pdf_bytes: bytes,
        source: DocumentSource,
        source_url: str | None,
        filename: str,
        warc_id: str | None,
        lang_threshold: float,
    ) -> AcquisitionResult | None:
        """Shared pipeline: hash → extract text → score → filter → build result.

        Both ``_run_local`` and ``_run_commoncrawl`` delegate here once they
        have raw PDF bytes, making this the single location for hashing,
        text extraction, scoring, and result construction.

        Parameters
        ----------
        pdf_bytes:
            Raw bytes of the PDF.
        source:
            The ``DocumentSource`` enum value for this document.
        source_url:
            Original URL for CC documents; ``None`` for local documents.
        filename:
            Original filename (local) or URL-derived slug (CC).
        warc_id:
            WARC-Record-ID header value for CC documents; ``None`` otherwise.
        lang_threshold:
            Documents scoring below this value are discarded.

        Returns
        -------
        AcquisitionResult | None
        """
        doc_id = _compute_doc_id(pdf_bytes)
        self._logger.debug("doc_id=%s  filename=%s", doc_id, filename)

        # _extract_sample_text may raise AcquisitionError(phase, doc_id, msg)
        sample = _extract_sample_text(pdf_bytes, doc_id)
        lang_score = _compute_lang_score(sample)

        self._logger.info(
            "doc_id=%s  lang_score=%.4f  threshold=%.2f  filename=%s",
            doc_id,
            lang_score,
            lang_threshold,
            filename,
        )

        if lang_score < lang_threshold:
            self._logger.info(
                "Discarding doc_id=%s (lang_score=%.4f < threshold=%.2f)",
                doc_id,
                lang_score,
                lang_threshold,
            )
            return None

        return AcquisitionResult(
            doc_id=doc_id,
            pdf_bytes=pdf_bytes,
            source=source,
            source_url=source_url,
            warc_id=warc_id,
            lang_score=lang_score,
            filename=filename,
            acquired_at=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    from datetime import datetime
    import fitz  # PyMuPDF

    # Ensure these are imported from your local module
    # Adjust the import paths if this code is inside phase1_acquisition.py
    # from bio_extraction.phases.phase1_acquisition import (
    #     AcquisitionPhase, PhaseOneInput, DocumentSource, AcquisitionError, _compute_lang_score
    # )

    print("=== Phase 1 smoke test ===\n")

    # 1. lang-score function directly
    polish_sample = (
        "Akt urodzenia. Pan Jan Kowalski ur. w roku 1887 w Grodnie, "
        "syn Józefa i Marii z Nowaków. Zm. w roku 1945 w Warszawie. "
        "Małżonek Anny z domu Wiśniewska."
    )
    english_sample = (
        "The quick brown fox jumps over the lazy dog. "
        "This document contains no Polish content whatsoever."
    )

    ps = _compute_lang_score(polish_sample)
    es = _compute_lang_score(english_sample)
    print(f"Polish sample   lang_score: {ps:.4f}  (expect ≥ 0.30)")
    print(f"English sample lang_score: {es:.4f}  (expect < 0.30)")
    assert ps >= 0.30, f"Polish sample scored too low: {ps}"
    assert es < 0.30, f"English sample scored too high: {es}"
    print("✓  _compute_lang_score\n")

    # 2. AcquisitionError constructor
    # Note: Using literal strings if constants aren't available in scope
    test_phase = "Phase1Acquisition"
    try:
        raise AcquisitionError(test_phase, "abc123", "test error")
    except AcquisitionError as exc:
        assert exc.phase_name == test_phase
        assert exc.doc_id == "abc123"
        assert "test error" in str(exc)
    print("✓  AcquisitionError constructor\n")

    # 3 & 4. Full local-mode run + discard path
    def _make_pdf(text: str, path: Path) -> None:
        """Write a single-page PDF with *text* to *path*."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
        doc.save(str(path))
        doc.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        polish_pdf = td / "polish.pdf"
        english_pdf = td / "english.pdf"
        _make_pdf(polish_sample, polish_pdf)
        _make_pdf(english_sample, english_pdf)

        # Stub settings to fix the Mypy "Cannot infer type of lambda" error
        class _FakeSettings:
            lang_score_threshold: float = 0.30
            cc_request_timeout: int = 30
            # Add other required settings attributes here if AcquisitionPhase needs them
            input_dir: str = tmpdir
            source: str = "local"

        import bio_extraction.config as _cfg

        _original = _cfg.get_settings
        # Fixed lambda to accept arguments if the real get_settings does
        _cfg.get_settings = lambda *args, **kwargs: _FakeSettings()

        try:
            phase = AcquisitionPhase()

            # Should produce a result
            result = phase.run(PhaseOneInput(source=DocumentSource.LOCAL, local_path=polish_pdf))
            assert result is not None
            assert len(result.doc_id) == 16
            assert result.lang_score >= 0.30
            assert result.source is DocumentSource.LOCAL
            assert result.filename == "polish.pdf"
            assert isinstance(result.acquired_at, datetime)
            print(
                f"✓  Local run passed: doc_id={result.doc_id}  "
                f"lang_score={result.lang_score:.4f}  filename={result.filename}"
            )

            # Should be discarded (None returned)
            discarded = phase.run(
                PhaseOneInput(source=DocumentSource.LOCAL, local_path=english_pdf)
            )
            assert discarded is None
            print("✓  English PDF correctly discarded\n")

            # 5. File-not-found → AcquisitionError
            try:
                phase.run(
                    PhaseOneInput(
                        source=DocumentSource.LOCAL,
                        local_path=Path("/no/such/file.pdf"),
                    )
                )
                assert False, "Expected AcquisitionError"
            except AcquisitionError as exc:
                # Use the actual attribute if _PHASE_NAME is a constant in the module
                print(f"✓  File-not-found raises AcquisitionError: {exc}\n")

        finally:
            _cfg.get_settings = _original
    # ------------------------------------------------------------------ #
    # 6. enumerate_local_inputs                                            #
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        for name in ("alfa.pdf", "beta.pdf", "gamma.txt", "delta.csv"):
            (td / name).write_bytes(b"%PDF-1.4 fake")

        inputs = enumerate_local_inputs(td)
        assert len(inputs) == 2, f"Expected 2 PDFs, got {len(inputs)}"
        assert all(i.source is DocumentSource.LOCAL for i in inputs)
        assert all(i.local_path is not None and i.local_path.suffix == ".pdf" for i in inputs)
        print(f"✓  enumerate_local_inputs: {len(inputs)} PDFs found, .txt/.csv ignored")

    try:
        enumerate_local_inputs(Path("/no/such/dir"))
        assert False, "Expected AcquisitionError"
    except AcquisitionError as exc:
        assert exc.phase_name == _PHASE_NAME
        print(f"✓  enumerate_local_inputs raises AcquisitionError for bad dir: {exc}")

    print("\n=== All smoke tests passed ===")
