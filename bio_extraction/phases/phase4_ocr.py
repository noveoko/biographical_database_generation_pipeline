# REQUIRES: pillow, pytesseract, numpy, scikit-image
"""
phase4_ocr.py
=============
Phase 4 — Preprocessing & OCR.

Receives a ``LayoutResult`` containing one or more ``ContentSlice`` objects.
For every slice that carries ``image_bytes`` this phase:

1. Decodes the PNG crop to a PIL ``Image``.
2. Converts to greyscale and applies **Sauvola adaptive binarisation** to
   handle the uneven illumination typical of document scans.
3. **Deskews** the binarised image by sweeping a ±5° rotation range in 0.5°
   steps and choosing the angle that maximises the variance of the horizontal
   projection profile (a proxy for text-line sharpness).
4. Applies a **median filter** (kernel 3×3) to suppress salt-and-pepper noise
   introduced by digitisation artefacts.
5. Runs **Tesseract** with the configured language pack(s).

   For historical Polish documents printed in Fraktur, the recommended
   configuration is:

       tesseract input.jpg output -l pol+Fraktur

   This combines:
   - ``pol``      → dictionary, morphology, Polish language model
   - ``Fraktur``  → character recognition for Gothic/blackletter script

   In this implementation the same effect is achieved via:

       self._tesseract_langs = "pol+Fraktur"

   (configured externally via ``bio_extraction.config``)
6. Derives an aggregate ``confidence_score`` (mean of valid per-word scores)
   and sets ``needs_review`` when it falls below the configured threshold.
7. Packs everything into an ``OCREntry`` and returns an ``OCRResult``.

Error handling
--------------
- Slices without ``image_bytes`` are skipped with a WARNING.
- Tesseract not found → ``OCRError`` with remediation hint.
- All slices fail → ``OCRError``.
- Individual slice errors are logged and skipped; the document is not aborted
  unless *every* slice fails.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytesseract
from PIL import Image, ImageFilter
from skimage.filters import threshold_sauvola

from bio_extraction.config import get_settings
from bio_extraction.contracts import (
    ContentSlice,
    LayoutResult,
    OCREntry,
    OCRResult,
)
from bio_extraction.exceptions import OCRError
from bio_extraction.logging_config import get_phase_logger
from bio_extraction.protocol import PhaseProtocol

# ---------------------------------------------------------------------------
# Constants (algorithm hyper-parameters that are *not* user-configurable)
# ---------------------------------------------------------------------------

_SAUVOLA_WINDOW_SIZE: int = 25  # must be odd; matches spec
_SAUVOLA_K: float = 0.15  # Sauvola sensitivity parameter
_DESKEW_RANGE_DEG: float = 5.0  # sweep ±5°
_DESKEW_STEP_DEG: float = 0.5  # resolution of sweep
_MEDIAN_KERNEL: int = 3  # square kernel for median denoise
_TESSERACT_CONFIG: str = "--oem 3 --psm 6"  # LSTM engine, uniform text block


# ---------------------------------------------------------------------------
# OCRPhase
# ---------------------------------------------------------------------------


class OCRPhase(PhaseProtocol[LayoutResult, OCRResult]):
    """
    Pipeline Phase 4: image preprocessing and Tesseract OCR.

    Instantiate once and call ``run()`` for each document fed by the runner.
    All configuration is read from ``bio_extraction.config.get_settings()``
    on construction, so the object is safe to reuse across documents without
    re-reading config on every call.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._log = get_phase_logger(self.phase_name)
        self._confidence_threshold: float = self._settings.ocr.confidence_threshold
        self._tesseract_langs: str = self._settings.ocr.tesseract_langs
        self._verify_tesseract()
        if "Fraktur" not in self._tesseract_langs:
            self._log.warning(
                "Fraktur model not enabled — OCR quality may be poor for historical documents",
                extra={"tesseract_langs": self._tesseract_langs},
            )
        if "pol" not in self._tesseract_langs:
            self._log.warning(
                "Polish language model not enabled — dictionary correction will be limited",
                extra={"tesseract_langs": self._tesseract_langs},
            )

    # ------------------------------------------------------------------
    # PhaseProtocol interface
    # ------------------------------------------------------------------

    @property
    def phase_name(self) -> str:
        """Unique snake_case identifier consumed by the runner and checkpoint system."""
        return "phase4_ocr"

    def run(self, input_data: LayoutResult) -> OCRResult | None:
        """
        Process all content slices in ``input_data`` and return an ``OCRResult``.

        Parameters
        ----------
        input_data:
            ``LayoutResult`` produced by Phase 3 for one document.

        Returns
        -------
        OCRResult
            One ``OCREntry`` per successfully OCR-ed slice.
        None
            Never returned by this phase — an empty result set raises instead.

        Raises
        ------
        OCRError
            If every slice in the document fails to produce OCR output.
        """
        doc_id = input_data.doc_id
        self._log.info("Starting OCR", extra={"doc_id": doc_id, "n_slices": len(input_data.slices)})

        ocr_entries: list[OCREntry] = []
        failed: int = 0

        for slice_obj in input_data.slices:
            entry = self._process_slice(slice_obj, doc_id)
            if entry is None:
                failed += 1
            else:
                ocr_entries.append(entry)

        total = len(input_data.slices)
        succeeded = total - failed
        self._log.info(
            "OCR complete",
            extra={"doc_id": doc_id, "succeeded": succeeded, "failed": failed},
        )

        if succeeded == 0 and total > 0:
            raise OCRError(
                phase_name=self.phase_name,
                doc_id=doc_id,
                message=f"All {total} slice(s) failed OCR — document cannot be processed.",
            )

        return OCRResult(
            doc_id=doc_id,
            ocr_entries=ocr_entries,
            processed_at=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Per-slice orchestration
    # ------------------------------------------------------------------

    def _process_slice(self, slice_obj: ContentSlice, doc_id: str) -> OCREntry | None:
        """
        Run the full preprocessing + OCR pipeline for a single ``ContentSlice``.

        Returns ``None`` and logs a warning when the slice should be skipped
        (empty ``image_bytes``, Pillow decode failure, or Tesseract produces
        no usable word tokens).
        """
        sid = slice_obj.slice_id

        # ── 1. Decode image bytes ──────────────────────────────────────
        if not slice_obj.image_bytes:
            self._log.warning("Skipping slice with no image_bytes", extra={"slice_id": sid})
            return None

        try:
            pil_image: Image.Image = Image.open(io.BytesIO(slice_obj.image_bytes))
            pil_image.load()  # force decode so I/O errors surface here
        except Exception as exc:
            self._log.warning(
                "Failed to decode image bytes",
                extra={"slice_id": sid, "error": str(exc)},
            )
            return None

        # ── 2. Adaptive binarisation (Sauvola) ────────────────────────
        try:
            binarised = self._binarise_sauvola(pil_image)
        except Exception as exc:
            self._log.warning(
                "Binarisation failed",
                extra={"slice_id": sid, "error": str(exc)},
            )
            return None

        # ── 3. Deskew ─────────────────────────────────────────────────
        try:
            deskewed = self._deskew(binarised)
        except Exception as exc:
            self._log.warning(
                "Deskew failed; proceeding with binarised image",
                extra={"slice_id": sid, "error": str(exc)},
            )
            deskewed = binarised  # graceful fallback

        # ── 4. Denoise (median filter) ────────────────────────────────
        denoised = self._denoise(deskewed)

        # ── 5–8. Tesseract OCR + assemble result ──────────────────────
        try:
            entry = self._run_tesseract(denoised, slice_obj)
        except Exception as exc:
            self._log.warning(
                "Tesseract failed for slice",
                extra={"slice_id": sid, "error": str(exc)},
            )
            return None

        return entry

    # ------------------------------------------------------------------
    # Image preprocessing helpers
    # ------------------------------------------------------------------

    def _binarise_sauvola(self, image: Image.Image) -> Image.Image:
        """
        Convert *image* to a binary (black-on-white) PIL Image using Sauvola
        thresholding.

        Sauvola's local threshold at pixel (x, y) is:

            T(x,y) = μ(x,y) · [1 + k · (σ(x,y)/R − 1)]

        where μ and σ are the local mean and standard deviation inside a
        ``window_size × window_size`` neighbourhood, R is the dynamic range of
        the standard deviation (128 for uint8 images), and k controls
        sensitivity to local contrast (_SAUVOLA_K = 0.15).

        Lower k → more pixels are classified as foreground (text); higher k →
        stricter threshold, useful for high-contrast prints.
        """
        grey = image.convert("L")
        _arr = np.array(grey, dtype=np.uint8)
        thresh = threshold_sauvola(_arr, window_size=_SAUVOLA_WINDOW_SIZE, k=_SAUVOLA_K)
        binary = (_arr > thresh).astype(np.uint8) * 255  # white text regions = 255
        # Invert so text is black on white (Tesseract convention)
        binary = 255 - binary
        return Image.fromarray(binary, mode="L")

    def _deskew(self, image: Image.Image) -> Image.Image:
        """
        Correct small rotational skew by sweeping ±``_DESKEW_RANGE_DEG``° in
        ``_DESKEW_STEP_DEG``° increments and selecting the angle that maximises
        the **variance of the horizontal projection profile**.

        Intuition
        ---------
        When text lines are axis-aligned the sum of black pixels per row forms
        a spiky distribution (high variance: rows *through* text have many
        black pixels, gaps between lines have few).  Any tilt blurs these
        spikes → variance drops.  We exploit this to find the true upright
        angle without needing to detect individual text lines.

        Only angles in [-5°, +5°] are considered — larger corrections indicate
        a layout problem, not a scan artefact.
        """
        _arr = np.array(image, dtype=np.float32)
        angles = np.arange(-_DESKEW_RANGE_DEG, _DESKEW_RANGE_DEG + 1e-9, _DESKEW_STEP_DEG)

        best_angle: float = 0.0
        best_variance: float = -1.0

        for angle in angles:
            rotated_img = image.rotate(angle, expand=False, fillcolor=255)
            rotated_arr = np.array(rotated_img, dtype=np.float32)
            # Horizontal projection: row-wise sum of *inverted* image
            # (black pixels = 0 in our convention → invert to make them 255)
            projection = (255.0 - rotated_arr).sum(axis=1)
            variance = float(np.var(projection))
            if variance > best_variance:
                best_variance = variance
                best_angle = float(angle)

        if best_angle != 0.0:
            self._log.debug(
                "Deskew applied",
                extra={"angle_deg": best_angle, "projection_variance": best_variance},
            )
            return image.rotate(best_angle, expand=True, fillcolor=255)
        return image

    def _denoise(self, image: Image.Image) -> Image.Image:
        """
        Suppress salt-and-pepper noise with a ``_MEDIAN_KERNEL × _MEDIAN_KERNEL``
        median filter.

        The median filter replaces each pixel with the median of its
        neighbourhood.  It is effective against isolated noise spikes (salt =
        isolated white pixels, pepper = isolated black pixels) while preserving
        edges — important for thin strokes in historical Polish typefaces.
        """
        return image.filter(ImageFilter.MedianFilter(size=_MEDIAN_KERNEL))

    # ------------------------------------------------------------------
    # Tesseract wrapper
    # ------------------------------------------------------------------

    def _run_tesseract(self, image: Image.Image, slice_obj: ContentSlice) -> OCREntry:
        """
        Run Tesseract on *image* and construct an ``OCREntry``.

        Uses ``pytesseract.image_to_data`` to obtain per-word bounding boxes
        and confidence scores in a single call (avoids running Tesseract twice).

        Confidence filtering
        --------------------
        Tesseract reports −1 for whitespace tokens and non-text regions.
        These are excluded from both the mean confidence calculation and the
        assembled output text.

        Bounding box format
        -------------------
        ``image_to_data`` returns boxes in the form (left, top, width, height)
        in *slice-local* pixel coordinates.  We convert to (x1, y1, x2, y2) to
        match the ``OCREntry.bboxes`` contract.
        """
        raw: dict[str, Any] = pytesseract.image_to_data(
            image,
            lang=self._tesseract_langs,
            config=_TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
        )

        words: list[str] = []
        confidences: list[float] = []
        bboxes: list[tuple[int, int, int, int]] = []

        n_boxes = len(raw["text"])
        for i in range(n_boxes):
            conf = int(raw["conf"][i])
            word = str(raw["text"][i]).strip()
            if conf < 0 or not word:  # skip whitespace / non-text tokens
                continue
            words.append(word)
            confidences.append(float(conf) / 100.0)  # normalise 0–100 → 0–1

            left = int(raw["left"][i])
            top = int(raw["top"][i])
            width = int(raw["width"][i])
            height = int(raw["height"][i])
            bboxes.append((left, top, left + width, top + height))

        full_text = " ".join(words)

        confidence_score: float = float(np.mean(confidences)) if confidences else 0.0
        needs_review = confidence_score < self._confidence_threshold

        if not words:
            self._log.warning(
                "Tesseract produced no words for slice",
                extra={"slice_id": slice_obj.slice_id},
            )

        return OCREntry(
            slice_id=slice_obj.slice_id,
            text=full_text,
            confidence_score=round(confidence_score, 4),
            bboxes=bboxes,
            needs_review=needs_review,
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _verify_tesseract(self) -> None:
        """
        Confirm that the ``tesseract`` binary is reachable on startup.

        Raises
        ------
        OCRError
            If ``pytesseract.get_tesseract_version()`` fails, which happens
            when Tesseract is not installed or not on PATH.
        """
        try:
            version = pytesseract.get_tesseract_version()
            self._log.info("Tesseract detected", extra={"version": str(version)})
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRError(
                phase_name=self.phase_name,
                doc_id="<startup>",
                message=(
                    "Tesseract OCR binary not found. "
                    "Install it with:  sudo apt-get install tesseract-ocr tesseract-ocr-pol  "
                    "and ensure it is on your PATH.  "
                    f"Original error: {exc}"
                ),
            ) from exc


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal smoke test — runs without any real PDF or DB.

    Creates a synthetic LayoutResult with two slices:
      - Slice A: a tiny white image with the word "Kowalski" rendered in black.
      - Slice B: no image_bytes (should be skipped gracefully).

    Requires Tesseract + the 'pol' language pack to be installed locally.
    Run with:
        python -m bio_extraction.phases.phase4_ocr
    """
    import sys
    from datetime import datetime, timezone
    from io import BytesIO

    from PIL import Image, ImageDraw

    from bio_extraction.contracts import (
        ContentSlice,
        DocumentType,
        LayoutResult,
    )

    print("=== Phase 4 smoke test ===\n")

    # ── Build a tiny synthetic image with legible text ─────────────────
    def _make_text_image(text: str, width: int = 300, height: int = 60) -> bytes:
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Use the default PIL font — no external font files required
        draw.text((10, 15), text, fill=(0, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    doc_id = "smoke_test_doc_001"
    now = datetime.now(tz=timezone.utc)

    slice_a = ContentSlice(
        slice_id=f"{doc_id}_p0_e0",
        page_num=0,
        entry_index=0,
        bbox=(0, 0, 300, 60),
        image_bytes=_make_text_image("Jan Kowalski ur. 1892 Lwów"),
    )
    slice_b = ContentSlice(
        slice_id=f"{doc_id}_p0_e1",
        page_num=0,
        entry_index=1,
        bbox=(0, 70, 300, 130),
        image_bytes=None,  # intentionally empty — should be skipped
    )

    layout = LayoutResult(
        doc_id=doc_id,
        doc_type=DocumentType.DIRECTORY,
        slices=[slice_a, slice_b],
        analyzed_at=now,
    )

    # ── Run the phase ──────────────────────────────────────────────────
    try:
        phase = OCRPhase()
        result = phase.run(layout)
    except OCRError as exc:
        print(f"[OCRError] {exc}")
        sys.exit(1)

    if result is None:
        print("Phase returned None (document discarded).")
        sys.exit(0)

    print(f"doc_id          : {result.doc_id}")
    print(f"processed_at    : {result.processed_at}")
    print(f"entries returned: {len(result.ocr_entries)}  (expected 1 — slice_b skipped)\n")

    for entry in result.ocr_entries:
        print(f"  slice_id        : {entry.slice_id}")
        print(f"  text            : {entry.text!r}")
        print(f"  confidence_score: {entry.confidence_score:.4f}")
        print(f"  needs_review    : {entry.needs_review}")
        print(f"  word bboxes     : {entry.bboxes[:3]} …")
        print()

    assert len(result.ocr_entries) == 1, "Expected exactly 1 entry (slice_b has no image)"
    print("✓ Smoke test passed.")

# Public alias used by test_e2e_pipeline.py
Phase4OCR = OCRPhase
