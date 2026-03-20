# REQUIRES: (none — stdlib only, uses sqlite3)

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

from pydantic import BaseModel


class ReviewItem(BaseModel):
    """Pydantic model representing a review queue item."""

    id: int | None = None
    doc_id: str
    slice_id: str | None = None
    phase_name: str
    reason: str
    data_snapshot: dict
    created_at: datetime
    status: str = "pending"
    resolved_at: datetime | None = None
    resolution_notes: str | None = None
    corrected_data: dict | None = None


class ManualReviewQueue:
    """SQLite-backed queue for manual review workflow."""

    def __init__(self, db_path: Path):
        """Open or create the review queue database."""
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    # ---------------- PUBLIC API ----------------

    def enqueue(self, item: ReviewItem) -> int:
        """Add an item for review. Returns the queue item ID."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO review_queue (
                doc_id, slice_id, phase_name, reason,
                data_snapshot, created_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.doc_id,
                item.slice_id,
                item.phase_name,
                item.reason,
                json.dumps(item.data_snapshot, ensure_ascii=False),
                item.created_at.isoformat(),
                item.status,
            ),
        )

        self.conn.commit()
        return cursor.lastrowid

    def dequeue(self, limit: int = 10) -> List[ReviewItem]:
        """Fetch the next N unreviewed items (FIFO)."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT * FROM review_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()
        return [self._row_to_model(row) for row in rows]

    def resolve(
        self,
        item_id: int,
        resolution: str,
        corrected_data: dict | None = None,
    ) -> None:
        """
        Mark an item as reviewed.

        Parameters
        ----------
        resolution : str
            One of: 'accepted', 'corrected', 'discarded'
        """
        if resolution not in {"accepted", "corrected", "discarded"}:
            raise ValueError("Invalid resolution status")

        cursor = self.conn.cursor()

        cursor.execute(
            """
            UPDATE review_queue
            SET status = ?, resolved_at = ?, corrected_data = ?
            WHERE id = ?
            """,
            (
                resolution,
                datetime.utcnow().isoformat(),
                json.dumps(corrected_data, ensure_ascii=False) if corrected_data else None,
                item_id,
            ),
        )

        self.conn.commit()

    def stats(self) -> dict:
        """Return queue statistics."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT status, COUNT(*) as count
            FROM review_queue
            GROUP BY status
            """
        )

        counts = {row["status"]: row["count"] for row in cursor.fetchall()}

        total = sum(counts.values())

        return {
            "pending": counts.get("pending", 0),
            "accepted": counts.get("accepted", 0),
            "corrected": counts.get("corrected", 0),
            "discarded": counts.get("discarded", 0),
            "total": total,
        }

    # ---------------- INTERNAL ----------------

    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self.conn.cursor()

        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                slice_id TEXT,
                phase_name TEXT NOT NULL,
                reason TEXT NOT NULL,
                data_snapshot TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                resolved_at TEXT,
                resolution_notes TEXT,
                corrected_data TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status);
            CREATE INDEX IF NOT EXISTS idx_review_phase ON review_queue(phase_name);
            """
        )

        self.conn.commit()

    def _row_to_model(self, row: sqlite3.Row) -> ReviewItem:
        """Convert DB row to ReviewItem model."""
        return ReviewItem(
            id=row["id"],
            doc_id=row["doc_id"],
            slice_id=row["slice_id"],
            phase_name=row["phase_name"],
            reason=row["reason"],
            data_snapshot=json.loads(row["data_snapshot"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            status=row["status"],
            resolved_at=datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None,
            resolution_notes=row["resolution_notes"],
            corrected_data=json.loads(row["corrected_data"]) if row["corrected_data"] else None,
        )


if __name__ == "__main__":
    # Minimal smoke test
    db_path = Path("review_queue_test.db")
    queue = ManualReviewQueue(db_path)

    item = ReviewItem(
        doc_id="doc1",
        slice_id="slice1",
        phase_name="phase4_ocr",
        reason="low_ocr_confidence",
        data_snapshot={"text": "Jna Kowalski"},
        created_at=datetime.utcnow(),
    )

    item_id = queue.enqueue(item)
    print("Enqueued ID:", item_id)

    items = queue.dequeue()
    print("Dequeued:", items)

    queue.resolve(item_id, "corrected", {"text": "Jan Kowalski"})

    print("Stats:", queue.stats())
