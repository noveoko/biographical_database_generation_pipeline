# REQUIRES: pymupdf, numpy, pillow

from __future__ import annotations

import io
from datetime import datetime
from typing import List, Optional, Tuple

import fitz  # pymupdf
import numpy as np
from PIL import Image

from bio_extraction.contracts import (
    ClassificationResult,
    LayoutResult,
    ContentSlice,
    AcquisitionResult,
    DocumentType,
)
from bio_extraction.protocol import PhaseProtocol
from bio_extraction.exceptions import PhaseError
from bio_extraction.logging_config import get_phase_logger
from bio_extraction.config import get_settings
from bio_extraction.checkpoint import CheckpointEngine


class LayoutPhase(PhaseProtocol[ClassificationResult, LayoutResult]):
    """Phase 3: Layout Analysis — splits directory PDFs into entry slices."""

    phase_name = "phase3_layout"

    def __init__(self) -> None:
        self.logger = get_phase_logger(self.phase_name)
        self.settings = get_settings()

    def run(self, doc: ClassificationResult) -> Optional[LayoutResult]:
        """Run layout analysis on a single classified document."""
        if doc.doc_type != DocumentType.DIRECTORY:
            self.logger.info(f"Skipping non-directory doc {doc.doc_id}")
            return None

        try:
            acquisition: AcquisitionResult = CheckpointEngine.load(
                "phase1_acquisition", doc.doc_id, AcquisitionResult
            )
        except Exception as e:
            raise PhaseError(self.phase_name, doc.doc_id, f"Checkpoint load failed: {e}") from e

        try:
            pdf = fitz.open(stream=acquisition.pdf_bytes, filetype="pdf")
        except Exception as e:
            raise PhaseError(self.phase_name, doc.doc_id, f"Failed to open PDF: {e}") from e

        slices: List[ContentSlice] = []

        for page_num in range(len(pdf)):
            try:
                page = pdf[page_num]
                img = self._render_page(page)
            except Exception as e:
                self.logger.warning(f"Page render failed (page {page_num}): {e}")
                continue

            h, w = img.shape

            # Remove header/footer (top/bottom 10%)
            top_cut = int(0.1 * h)
            bottom_cut = int(0.9 * h)
            img = img[top_cut:bottom_cut, :]

            columns = self._detect_columns(img)

            entry_index = 0

            for x1, x2 in columns:
                col_img = img[:, x1:x2]
                entries = self._segment_entries(col_img)

                for y1, y2 in entries:
                    if (y2 - y1) < 20:
                        continue

                    slice_img = col_img[y1:y2, :]
                    image_bytes = self._to_bytes(slice_img)

                    slice_id = f"{doc.doc_id}_p{page_num}_e{entry_index}"

                    slices.append(
                        ContentSlice(
                            slice_id=slice_id,
                            page_num=page_num,
                            entry_index=entry_index,
                            bbox=(x1, y1 + top_cut, x2, y2 + top_cut),
                            image_bytes=image_bytes,
                        )
                    )

                    entry_index += 1

        if not slices:
            raise PhaseError(self.phase_name, doc.doc_id, "No entries found")

        return LayoutResult(
            doc_id=doc.doc_id,
            doc_type=doc.doc_type,
            slices=slices,
            analyzed_at=datetime.utcnow(),
        )

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _render_page(self, page: fitz.Page) -> np.ndarray:
        """Render page to grayscale numpy array at 200 DPI."""
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        return img.reshape(pix.height, pix.width)

    def _detect_columns(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """Detect columns via vertical projection profile."""
        col_sum = np.sum(img, axis=0)
        threshold = np.percentile(col_sum, 20)

        is_valley = col_sum < threshold

        splits = []
        start = None

        for i, val in enumerate(is_valley):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if i - start > img.shape[0] * 0.8:
                    splits.append((start, i))
                start = None

        if not splits:
            return [(0, img.shape[1])]

        boundaries = [0] + [(s[0] + s[1]) // 2 for s in splits] + [img.shape[1]]

        return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    def _segment_entries(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """Segment entries via horizontal projection."""
        row_sum = np.sum(img, axis=1)
        threshold = np.percentile(row_sum, 10)

        is_gap = row_sum < threshold

        entries = []
        start = 0

        for i, gap in enumerate(is_gap):
            if gap:
                if i - start > 5:
                    entries.append((start, i))
                start = i + 1

        if start < len(img) - 1:
            entries.append((start, len(img)))

        return entries

    def _to_bytes(self, img: np.ndarray) -> bytes:
        """Convert numpy array to PNG bytes."""
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()


# ----------------------------
# Smoke test
# ----------------------------

if __name__ == "__main__":
    from bio_extraction.contracts import ClassificationResult, DocumentType
    from datetime import datetime

    fake = ClassificationResult(
        doc_id="test_doc",
        doc_type=DocumentType.DIRECTORY,
        confidence=1.0,
        sample_page_indices=[0],
        classified_at=datetime.utcnow(),
    )

    phase = LayoutPhase()

    try:
        result = phase.run(fake)
        print("Slices:", len(result.slices) if result else 0)
    except Exception as e:
        print("Error:", e)
