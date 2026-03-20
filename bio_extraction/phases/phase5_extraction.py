# REQUIRES: requests

"""
phase5_extraction.py
====================
Phase 5 — Hybrid Entity Extraction.

Extracts structured ``PersonEntity`` objects from OCR text using a two-tier
strategy:

  Tier 1 — Cached regex   (fast, O(1) once the cache is warm)
  Tier 2 — Ollama LLM     (fallback for unseen structural layouts)

After a successful LLM extraction the phase attempts to synthesise a candidate
regex for the detected structural fingerprint, validates it against the current
batch and the SurnameBloomFilter, and — if it passes — stores it in the
PatternCache so that future documents with the same layout skip the LLM entirely.

Dependency contracts
--------------------
- Reads from : ``OCRResult``  (one ``OCREntry`` per content slice)
- Writes to  : ``ExtractionResult``  (one or many ``PersonEntity`` per document)
- Utilities  : ``bio_extraction.utilities.pattern_cache.PatternCache``
               ``bio_extraction.utilities.bloom_filter.SurnameBloomFilter``
- Config     : ``bio_extraction.config.get_settings()``
- Logger     : ``bio_extraction.logging_config.get_phase_logger(self.phase_name)``
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import defaultdict
from typing import Any

import requests

from bio_extraction.config import get_settings
from bio_extraction.contracts import (
    ExtractionMethod,
    ExtractionResult,
    OCREntry,
    OCRResult,
    PersonEntity,
)
from bio_extraction.exceptions import ExtractionError
from bio_extraction.logging_config import get_phase_logger
from bio_extraction.protocol import PhaseProtocol
from bio_extraction.utilities.bloom_filter import SurnameBloomFilter
from bio_extraction.utilities.pattern_cache import PatternCache


# ---------------------------------------------------------------------------
# Structural fingerprint helpers
# ---------------------------------------------------------------------------

# Pre-compiled substitution patterns for fingerprint generation (compiled once
# at import time for performance).
_RE_UPPERCASE_WORD = re.compile(r"\b[A-ZŁŚŹŻĆŃÓĄ]{2,}\b")  # ≥2 uppercase letters
_RE_FOUR_DIGIT = re.compile(r"\b\d{4}\b")  # exactly 4 digits (year)
_RE_SHORT_DIGIT = re.compile(r"\b\d{1,2}\b")  # 1–2 digit numbers
_RE_LOWERCASE_WORD = re.compile(r"\b[a-złśźżćńóą]{2,}\b")  # ≥2 lowercase letters


def _compute_fingerprint(text: str) -> str:
    """
    Derive a structural fingerprint from an OCR entry's raw text.

    The fingerprint collapses lexical content into a structural template so
    that entries sharing the same layout (e.g. "SURNAME, Given, ur. YEAR")
    produce the same hash key regardless of specific names or dates.

    Substitution rules (applied in order):
      * UPPERCASE words  → ``U``
      * 4-digit numbers  → ``Y``
      * 1-2 digit nums   → ``N``
      * lowercase words  → ``l``
      * All punctuation and whitespace is preserved unchanged.

    The resulting template string is then SHA-256 hashed and truncated to 16
    hex chars — long enough to avoid collisions in a typical batch, short
    enough to be a readable cache key.

    Parameters
    ----------
    text:
        Raw OCR text for a single content slice.

    Returns
    -------
    str
        A 16-character hexadecimal fingerprint string.
    """
    # Apply substitutions sequentially; order matters — uppercase before
    # lowercase so "KOWALSKI" becomes "U" not "llllllll".
    template = _RE_UPPERCASE_WORD.sub("U", text)
    template = _RE_FOUR_DIGIT.sub("Y", template)
    template = _RE_SHORT_DIGIT.sub("N", template)
    template = _RE_LOWERCASE_WORD.sub("l", template)

    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
Extract biographical information from this Polish directory entry.
Return ONLY a JSON object with these fields:
- surname (string)
- given_names (list of strings)
- birth_date (string or null, format: YYYY or YYYY-MM-DD)
- death_date (string or null)
- locations (list of strings)
- roles (list of strings, e.g. job titles)

Entry text:
{ocr_text}
"""

_REGEX_GEN_PROMPT = """\
Given this Polish directory entry and the extracted data, write a Python regex \
pattern with named groups that would extract the same fields from similarly \
structured entries.
The regex should use these named groups: surname, given_names, birth_date, \
death_date, locations, roles.
Return ONLY the raw regex string, no explanation.

Entry: {ocr_text}
Extracted: {extracted_json}
"""


# ---------------------------------------------------------------------------
# Phase implementation
# ---------------------------------------------------------------------------


class ExtractionPhase(PhaseProtocol[OCRResult, ExtractionResult]):
    """
    Phase 5 — Hybrid Entity Extraction.

    Processes every ``OCREntry`` in the supplied ``OCRResult`` and produces a
    list of ``PersonEntity`` objects which are aggregated into an
    ``ExtractionResult``.

    The two-tier extraction strategy works as follows:

    **Tier 1 — Cached regex**
      Each entry's text is reduced to a structural fingerprint.  If the
      ``PatternCache`` already holds a compiled pattern for that fingerprint, the
      pattern is applied and — if it matches and the extracted surname passes the
      ``SurnameBloomFilter`` — the result is accepted immediately with method
      ``REGEX_CACHED``.

    **Tier 2 — Ollama LLM**
      Entries that miss the regex cache (or where the pattern does not match)
      are sent to the locally-running Ollama instance for extraction.  On a
      successful extraction the phase asks Ollama for a candidate regex,
      validates it, and — if valid — stores it in the cache for future runs.

    Parameters
    ----------
    pattern_cache:
        Pre-constructed ``PatternCache`` instance, injected for testability.
    bloom_filter:
        Pre-constructed ``SurnameBloomFilter`` instance, injected for
        testability.
    """

    def __init__(
        self,
        pattern_cache: PatternCache | None = None,
        bloom_filter: SurnameBloomFilter | None = None,
    ) -> None:
        self._log = get_phase_logger(self.phase_name)
        settings = get_settings()
        self._cfg = settings.extraction  # sub-config block for this phase

        self._cache: PatternCache = pattern_cache or PatternCache()
        self._bloom: SurnameBloomFilter = bloom_filter or SurnameBloomFilter()

        # Lazily-populated map: fingerprint → [raw_text, …] for the current
        # document.  Used when validating a newly generated candidate regex
        # against peer entries that share the same structural layout.
        self._fingerprint_peers: dict[str, list[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # PhaseProtocol interface
    # ------------------------------------------------------------------

    @property
    def phase_name(self) -> str:
        """Unique snake_case identifier used for checkpoints and dead-letter routing."""
        return "phase5_extraction"

    def run(self, input_data: OCRResult) -> ExtractionResult | None:
        """
        Extract ``PersonEntity`` objects from every OCR entry in the document.

        The method is intentionally tolerant of individual-entry failures: a
        single bad OCR entry logs a warning and is skipped rather than aborting
        the entire document.  If *every* entry fails, an ``ExtractionError`` is
        raised so the runner routes the document to the dead-letter queue.

        Parameters
        ----------
        input_data:
            ``OCRResult`` produced by Phase 4 for a single document.

        Returns
        -------
        ExtractionResult
            Populated with all successfully extracted entities.
        None
            Returned when the document has zero OCR entries (silently discard).

        Raises
        ------
        ExtractionError
            When all entries fail extraction, or the Ollama service is
            unreachable.
        """
        doc_id = input_data.doc_id

        if not input_data.ocr_entries:
            self._log.warning("doc=%s has no OCR entries — discarding silently", doc_id)
            return None

        # First pass: build the per-document fingerprint → peer-texts map so
        # that regex validation can sample multiple structurally-identical
        # entries within the same batch.
        self._fingerprint_peers.clear()
        for entry in input_data.ocr_entries:
            fp = _compute_fingerprint(entry.text)
            self._fingerprint_peers[fp].append(entry.text)

        entities: list[PersonEntity] = []
        failure_count = 0

        for entry in input_data.ocr_entries:
            try:
                entity = self._process_entry(entry, doc_id)
                if entity is not None:
                    entities.append(entity)
            except ExtractionError:
                # Ollama is down — propagate immediately; no point processing
                # further entries without the fallback tier.
                raise
            except Exception as exc:  # noqa: BLE001
                failure_count += 1
                self._log.warning(
                    "doc=%s slice=%s — entry extraction failed (%s: %s); skipping",
                    doc_id,
                    entry.slice_id,
                    type(exc).__name__,
                    exc,
                )

        if not entities and failure_count == len(input_data.ocr_entries):
            raise ExtractionError(
                phase_name=self.phase_name,
                doc_id=doc_id,
                message=(
                    f"All {failure_count} OCR entries failed extraction. "
                    "Check OCR quality and Ollama availability."
                ),
            )

        self._log.info(
            "doc=%s — extracted %d entities from %d entries (%d failures)",
            doc_id,
            len(entities),
            len(input_data.ocr_entries),
            failure_count,
        )

        from datetime import datetime, timezone  # local import to keep top-level clean

        return ExtractionResult(
            doc_id=doc_id,
            entities=entities,
            extracted_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Per-entry orchestration
    # ------------------------------------------------------------------

    def _process_entry(self, entry: OCREntry, doc_id: str) -> PersonEntity | None:
        """
        Run the two-tier extraction strategy for a single OCR entry.

        Parameters
        ----------
        entry:
            A single ``OCREntry`` whose ``.text`` will be parsed.
        doc_id:
            Parent document identifier — included in any raised exceptions.

        Returns
        -------
        PersonEntity | None
            ``None`` if the entry text is empty (harmlessly skipped).
        """
        text = entry.text.strip()
        if not text:
            self._log.debug("slice=%s is empty — skipping", entry.slice_id)
            return None

        fingerprint = _compute_fingerprint(text)

        # ---- Tier 1: cached regex ----------------------------------------
        entity = self._try_regex_tier(text, fingerprint, entry.slice_id)
        if entity is not None:
            return entity

        # ---- Tier 2: LLM fallback ----------------------------------------
        return self._try_llm_tier(text, fingerprint, entry.slice_id, doc_id)

    # ------------------------------------------------------------------
    # Tier 1 — Regex cache
    # ------------------------------------------------------------------

    def _try_regex_tier(
        self,
        text: str,
        fingerprint: str,
        slice_id: str,
    ) -> PersonEntity | None:
        """
        Attempt extraction using a previously cached regex pattern.

        Looks up *fingerprint* in the ``PatternCache``.  If a compiled pattern
        is found it is applied to *text*; the result is accepted only when the
        extracted surname also passes the ``SurnameBloomFilter`` (guards against
        patterns that have drifted due to minor OCR noise).

        Parameters
        ----------
        text:
            Raw OCR text for this slice.
        fingerprint:
            16-char structural fingerprint acting as the cache key.
        slice_id:
            Provenance identifier forwarded to the returned entity.

        Returns
        -------
        PersonEntity | None
            A fully populated entity on a successful cache hit, ``None``
            otherwise (falls through to the LLM tier).
        """
        pattern: re.Pattern[str] | None = self._cache.get(fingerprint)
        if pattern is None:
            self._log.debug("slice=%s fingerprint=%s — cache miss", slice_id, fingerprint)
            return None

        match = pattern.search(text)
        if match is None:
            self._log.debug(
                "slice=%s fingerprint=%s — cached pattern did not match", slice_id, fingerprint
            )
            return None

        groups = match.groupdict()
        surname = groups.get("surname", "").strip()

        if not surname:
            self._log.debug("slice=%s — cached pattern matched but extracted no surname", slice_id)
            return None

        if not self._bloom.check(surname):
            self._log.debug("slice=%s — surname '%s' rejected by bloom filter", slice_id, surname)
            return None

        self._log.debug("slice=%s — cache hit (fingerprint=%s)", slice_id, fingerprint)
        return self._build_entity(
            slice_id=slice_id,
            groups=groups,
            raw_text=text,
            method=ExtractionMethod.REGEX_CACHED,
            confidence=0.8,
        )

    # ------------------------------------------------------------------
    # Tier 2 — LLM (Ollama)
    # ------------------------------------------------------------------

    def _try_llm_tier(
        self,
        text: str,
        fingerprint: str,
        slice_id: str,
        doc_id: str,
    ) -> PersonEntity | None:
        """
        Extract a ``PersonEntity`` via the local Ollama LLM.

        After a successful extraction the method attempts to generate and
        validate a candidate regex pattern for the structural fingerprint so
        that future entries with the same layout can be handled by Tier 1.

        Parameters
        ----------
        text:
            Raw OCR text for this slice.
        fingerprint:
            Structural fingerprint used for cache-write-back.
        slice_id:
            Provenance identifier forwarded to the returned entity.
        doc_id:
            Parent document ID included in any raised ``ExtractionError``.

        Returns
        -------
        PersonEntity | None
            A populated entity, or ``None`` if the LLM returns an empty surname
            (considered an extraction failure for this slice; caller logs a
            warning and continues).

        Raises
        ------
        ExtractionError
            When the Ollama service is unreachable or all JSON parsing
            strategies are exhausted.
        """
        self._log.debug("slice=%s — falling back to LLM tier", slice_id)

        # ---- Call 1: extract fields --------------------------------------
        extracted = self._ollama_extract(text, slice_id, doc_id)

        surname = (extracted.get("surname") or "").strip()
        if not surname:
            self._log.warning(
                "doc=%s slice=%s — LLM returned no surname; skipping entry", doc_id, slice_id
            )
            return None

        # Bloom-filter sanity check on the LLM result.
        bloom_hit = self._bloom.check(surname)
        if not bloom_hit:
            self._log.debug(
                "slice=%s — LLM surname '%s' not in bloom filter (confidence penalised)",
                slice_id,
                surname,
            )

        confidence = 0.6 if bloom_hit else 0.4

        # ---- Call 2: generate + validate candidate regex -----------------
        self._maybe_cache_regex(text, extracted, fingerprint, slice_id, doc_id)

        return self._build_entity(
            slice_id=slice_id,
            groups=extracted,
            raw_text=text,
            method=ExtractionMethod.LLM_OLLAMA,
            confidence=confidence,
        )

    def _ollama_extract(
        self,
        text: str,
        slice_id: str,
        doc_id: str,
    ) -> dict[str, Any]:
        """
        Call Ollama to extract biographical fields from *text*.

        Sends ``_EXTRACTION_PROMPT`` to ``{ollama_url}/api/generate`` and
        parses the response as JSON.  Three JSON-recovery strategies are
        attempted in order:

        1. Direct ``json.loads`` of the full response string.
        2. Extraction of a fenced code block (```json … ```).
        3. Extraction of the first ``{…}`` substring.

        Parameters
        ----------
        text:
            Raw OCR entry text to be parsed by the LLM.
        slice_id:
            Used in log and error messages for traceability.
        doc_id:
            Parent document ID passed to any ``ExtractionError``.

        Returns
        -------
        dict[str, Any]
            Parsed extraction result with keys: ``surname``, ``given_names``,
            ``birth_date``, ``death_date``, ``locations``, ``roles``.

        Raises
        ------
        ExtractionError
            On connection failure or exhausted JSON recovery strategies.
        """
        prompt = _EXTRACTION_PROMPT.format(ocr_text=text)
        raw_response = self._call_ollama(prompt, slice_id, doc_id)
        return self._parse_json_response(raw_response, slice_id, doc_id)

    def _maybe_cache_regex(
        self,
        text: str,
        extracted: dict[str, Any],
        fingerprint: str,
        slice_id: str,
        doc_id: str,
    ) -> None:
        """
        Ask the LLM for a candidate regex and conditionally store it in the cache.

        The candidate is rejected (and logged for review) if any of these checks
        fail:

        * ``re.compile`` raises ``re.error`` — the pattern is syntactically
          invalid.
        * The pattern does not match the original *text* — coverage failure.
        * The rate of bloom-filter surname hits across peer entries with the
          same fingerprint falls below ``config.extraction.bloom_hit_rate_threshold``
          — the pattern is over-fitted or noisy.

        Failures here are *non-fatal*: the entity was already extracted by the
        LLM; we simply won't accelerate future entries with a cached pattern.

        Parameters
        ----------
        text:
            The OCR entry used as the positive example for the LLM.
        extracted:
            The previously extracted fields dictionary (used in the prompt).
        fingerprint:
            Cache key under which a validated pattern will be stored.
        slice_id, doc_id:
            Used in log messages.
        """
        prompt = _REGEX_GEN_PROMPT.format(
            ocr_text=text,
            extracted_json=json.dumps(extracted, ensure_ascii=False),
        )
        try:
            raw_pattern = self._call_ollama(prompt, slice_id, doc_id).strip()
        except ExtractionError as exc:
            self._log.warning(
                "slice=%s — could not generate candidate regex (Ollama error: %s)", slice_id, exc
            )
            return

        # Strip accidental backtick fences that the LLM may add even when told not to.
        raw_pattern = re.sub(r"^```[a-z]*\n?", "", raw_pattern)
        raw_pattern = re.sub(r"\n?```$", "", raw_pattern).strip()

        # --- Syntactic validation -----------------------------------------
        try:
            compiled = re.compile(raw_pattern, re.UNICODE | re.VERBOSE)
        except re.error as exc:
            self._log.warning(
                "slice=%s — candidate regex compile error (%s); pattern rejected: %r",
                slice_id,
                exc,
                raw_pattern,
            )
            return

        # --- Coverage: must match the original entry ----------------------
        if not compiled.search(text):
            self._log.warning(
                "slice=%s fingerprint=%s — candidate regex did not match origin text; rejected",
                slice_id,
                fingerprint,
            )
            return

        # --- Bloom-filter validation on peer entries ----------------------
        peers = [t for t in self._fingerprint_peers.get(fingerprint, []) if t != text]
        sample = peers[:5]  # spec says up to 5 peers

        if sample:
            hits = 0
            for peer_text in sample:
                peer_match = compiled.search(peer_text)
                if peer_match:
                    peer_surname = (peer_match.groupdict().get("surname") or "").strip()
                    if peer_surname and self._bloom.check(peer_surname):
                        hits += 1

            hit_rate = hits / len(sample)
            threshold = self._cfg.bloom_hit_rate_threshold
            if hit_rate < threshold:
                self._log.warning(
                    "slice=%s fingerprint=%s — candidate regex bloom hit rate %.2f < %.2f; rejected",
                    slice_id,
                    fingerprint,
                    hit_rate,
                    threshold,
                )
                return

        # --- Store in cache -----------------------------------------------
        self._cache.set(fingerprint, compiled)
        self._log.info(
            "slice=%s fingerprint=%s — cached new regex pattern",
            slice_id,
            fingerprint,
        )

    # ------------------------------------------------------------------
    # Ollama HTTP helper
    # ------------------------------------------------------------------

    def _call_ollama(
        self,
        prompt: str,
        slice_id: str,
        doc_id: str,
    ) -> str:
        """
        Send a single prompt to the Ollama ``/api/generate`` endpoint.

        Parameters
        ----------
        prompt:
            Full prompt string sent to the model.
        slice_id:
            Used in error messages for traceability.
        doc_id:
            Parent document ID passed to any raised ``ExtractionError``.

        Returns
        -------
        str
            The ``response`` field from Ollama's JSON reply.

        Raises
        ------
        ExtractionError
            On ``ConnectionError``, ``Timeout``, HTTP error status, or a
            malformed Ollama response body.
        """
        url = f"{self._cfg.ollama_url.rstrip('/')}/api/generate"
        payload: dict[str, Any] = {
            "model": self._cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self._cfg.ollama_timeout_seconds,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ExtractionError(
                phase_name=self.phase_name,
                doc_id=doc_id,
                message=(
                    f"Cannot reach Ollama at {url}. "
                    "Ensure Ollama is running (`ollama serve`) and "
                    f"config.extraction.ollama_url is correct. Detail: {exc}"
                ),
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise ExtractionError(
                phase_name=self.phase_name,
                doc_id=doc_id,
                message=(
                    f"Ollama request timed out after "
                    f"{self._cfg.ollama_timeout_seconds}s (slice={slice_id}). "
                    f"Detail: {exc}"
                ),
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise ExtractionError(
                phase_name=self.phase_name,
                doc_id=doc_id,
                message=f"Ollama HTTP error for slice={slice_id}: {exc}",
            ) from exc

        try:
            body = response.json()
            return str(body["response"])
        except (KeyError, ValueError) as exc:
            raise ExtractionError(
                phase_name=self.phase_name,
                doc_id=doc_id,
                message=f"Unexpected Ollama response shape for slice={slice_id}: {exc}",
            ) from exc

    # ------------------------------------------------------------------
    # JSON parsing (with recovery strategies)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(
        raw: str,
        slice_id: str,
        doc_id: str,
    ) -> dict[str, Any]:
        """
                Parse the LLM response string into a Python dictionary.

                Three recovery strategies are attempted in sequence:

                1. **Direct parse** — the LLM correctly followed instructions and
                   the string is already valid JSON.
                2. **Fenced code block** — extract content between ```json and
        ``` markers.
                3. **Brace extraction** — find the first ``{`` and last ``}`` and
                   attempt to parse the substring.

                Parameters
                ----------
                raw:
                    Raw string returned by ``_call_ollama``.
                slice_id, doc_id:
                    Traceability context for any ``ExtractionError``.

                Returns
                -------
                dict[str, Any]
                    A dictionary guaranteed to contain at least the keys from the
                    extraction prompt (values may be ``None`` or empty lists if the
                    LLM omitted them).

                Raises
                ------
                ExtractionError
                    When all three strategies fail.
        """
        # Strategy 1: direct parse
        try:
            result = json.loads(raw)
            return _normalise_extraction(result)
        except json.JSONDecodeError:
            pass

        # Strategy 2: fenced code block
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fence_match:
            try:
                result = json.loads(fence_match.group(1))
                return _normalise_extraction(result)
            except json.JSONDecodeError:
                pass

        # Strategy 3: first { ... }
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                result = json.loads(brace_match.group(0))
                return _normalise_extraction(result)
            except json.JSONDecodeError:
                pass

        raise ExtractionError(
            phase_name="phase5_extraction",
            doc_id=doc_id,
            message=(
                f"Could not parse LLM JSON response for slice={slice_id}. "
                f"Raw response (first 200 chars): {raw[:200]!r}"
            ),
        )

    # ------------------------------------------------------------------
    # Entity assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_entity(
        *,
        slice_id: str,
        groups: dict[str, Any],
        raw_text: str,
        method: ExtractionMethod,
        confidence: float,
    ) -> PersonEntity:
        """
        Assemble a ``PersonEntity`` from extracted field groups.

        Handles the variance between the regex tier (which returns named match
        groups, all strings) and the LLM tier (which returns a parsed
        dictionary, values may be lists or None).

        Parameters
        ----------
        slice_id:
            Provenance link back to the source content slice.
        groups:
            Flat dictionary of extracted fields.  Accepted key names mirror
            the named groups in generated regex patterns and the JSON schema
            sent to the LLM.
        raw_text:
            Original OCR text retained for auditability.
        method:
            ``ExtractionMethod`` enum value indicating which tier produced
            this entity.
        confidence:
            Base confidence float (0–1).

        Returns
        -------
        PersonEntity
            Fully populated entity ready for inclusion in ``ExtractionResult``.
        """
        # -- given_names: may arrive as "Jan Józef" (str) or ["Jan", "Józef"]
        raw_given = groups.get("given_names") or ""
        if isinstance(raw_given, list):
            given_names = [n.strip() for n in raw_given if n and n.strip()]
        else:
            given_names = [n.strip() for n in str(raw_given).split() if n.strip()]

        # -- locations / roles: may arrive as comma-joined str or list
        locations = _coerce_to_list(groups.get("locations"))
        roles = _coerce_to_list(groups.get("roles"))

        return PersonEntity(
            entity_id=str(uuid.uuid4()),
            slice_id=slice_id,
            surname=(groups.get("surname") or "").strip().title(),
            given_names=given_names,
            birth_date=_clean_optional_str(groups.get("birth_date")),
            death_date=_clean_optional_str(groups.get("death_date")),
            locations=locations,
            roles=roles,
            raw_text=raw_text,
            confidence=confidence,
            extraction_method=method,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _normalise_extraction(data: Any) -> dict[str, Any]:
    """
    Ensure the parsed LLM response has all required keys with sane defaults.

    Missing keys are filled with ``None`` or ``[]`` so that downstream code
    never has to guard against ``KeyError``.

    Parameters
    ----------
    data:
        Parsed JSON value from the LLM.  Expected to be a dict; if it is not,
        a ``ValueError`` is raised.

    Returns
    -------
    dict[str, Any]
        Normalised dictionary with all expected keys present.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict from LLM, got {type(data)}")
    return {
        "surname": data.get("surname"),
        "given_names": data.get("given_names") or [],
        "birth_date": data.get("birth_date"),
        "death_date": data.get("death_date"),
        "locations": data.get("locations") or [],
        "roles": data.get("roles") or [],
    }


def _coerce_to_list(value: Any) -> list[str]:
    """
    Convert a value that may be a string, list, or None into a clean list of
    non-empty strings.

    A bare string is split on commas; each element is stripped of whitespace.

    Parameters
    ----------
    value:
        Raw field value from extraction groups.

    Returns
    -------
    list[str]
        Zero or more non-empty strings.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v and str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _clean_optional_str(value: Any) -> str | None:
    """
    Normalise an optional string field.

    Returns ``None`` for falsy values (empty string, None) and a stripped
    string otherwise.
    """
    if not value:
        return None
    s = str(value).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal smoke test exercising the fingerprint logic and entity assembly
    without any external services (no Ollama, no real cache/bloom instances).

    Run with:
        python -m bio_extraction.phases.phase5_extraction
    or:
        python phase5_extraction.py
    """

    print("=== Phase 5 smoke test ===\n")

    # --- 1. Fingerprint stability ----------------------------------------
    text_a = "KOWALSKI, Jan, ur. 1892, notariusz, Warszawa"
    text_b = "NOWACKI, Piotr, ur. 1903, lekarz, Kraków"
    # These two entries have the same structure → same fingerprint.
    fp_a = _compute_fingerprint(text_a)
    fp_b = _compute_fingerprint(text_b)
    assert fp_a == fp_b, f"Expected identical fingerprints, got {fp_a!r} and {fp_b!r}"
    print(f"[OK] Structural fingerprint is layout-stable: {fp_a!r}")

    # Different structure → different fingerprint.
    text_c = "Jan Kowalski (1892-1945)"
    fp_c = _compute_fingerprint(text_c)
    assert fp_c != fp_a, "Expected different fingerprint for different structure"
    print(f"[OK] Distinct structures produce distinct fingerprints: {fp_c!r}\n")

    # --- 2. JSON parsing recovery strategies --------------------------------
    valid_json = '{"surname": "Kowalski", "given_names": ["Jan"], "birth_date": "1892", "death_date": null, "locations": ["Warszawa"], "roles": ["notariusz"]}'
    parsed = ExtractionPhase._parse_json_response(valid_json, "slice_001", "doc_001")
    assert parsed["surname"] == "Kowalski"
    print("[OK] Direct JSON parse succeeded")

    fenced = '```json\n{"surname": "Nowak", "given_names": ["Anna"], "birth_date": null, "death_date": null, "locations": [], "roles": []}\n```'
    parsed2 = ExtractionPhase._parse_json_response(fenced, "slice_002", "doc_001")
    assert parsed2["surname"] == "Nowak"
    print("[OK] Fenced code-block JSON recovery succeeded")

    brace_noise = 'Sure, here you go: {"surname": "Wiśniewska", "given_names": ["Maria"], "birth_date": "1910", "death_date": null, "locations": [], "roles": ["nauczycielka"]}'
    parsed3 = ExtractionPhase._parse_json_response(brace_noise, "slice_003", "doc_001")
    assert parsed3["surname"] == "Wiśniewska"
    print("[OK] Brace-extraction JSON recovery succeeded\n")

    # --- 3. Entity assembly -------------------------------------------------
    entity = ExtractionPhase._build_entity(
        slice_id="test_p0_e0",
        groups=parsed,
        raw_text=text_a,
        method=ExtractionMethod.REGEX_CACHED,
        confidence=0.8,
    )
    assert entity.surname == "Kowalski"
    assert entity.given_names == ["Jan"]
    assert entity.birth_date == "1892"
    assert entity.locations == ["Warszawa"]
    assert entity.roles == ["notariusz"]
    assert entity.extraction_method == ExtractionMethod.REGEX_CACHED
    print(
        f"[OK] PersonEntity assembled correctly: {entity.surname}, {entity.given_names}, born {entity.birth_date}"
    )

    # --- 4. Helper utilities ------------------------------------------------
    assert _coerce_to_list("Warszawa, Kraków") == ["Warszawa", "Kraków"]
    assert _coerce_to_list(["Gdańsk", "Łódź"]) == ["Gdańsk", "Łódź"]
    assert _coerce_to_list(None) == []
    print("[OK] _coerce_to_list handles str, list, and None\n")

    print("=== All smoke tests passed ✓ ===")
