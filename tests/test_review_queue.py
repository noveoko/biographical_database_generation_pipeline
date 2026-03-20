import tempfile
from datetime import datetime
from pathlib import Path

from bio_extraction.utilities.review_queue import ManualReviewQueue, ReviewItem


def test_enqueue_and_dequeue():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queue.db"
        queue = ManualReviewQueue(db_path)

        item = ReviewItem(
            doc_id="doc1",
            slice_id="slice1",
            phase_name="phase4_ocr",
            reason="low_confidence",
            data_snapshot={"text": "abc"},
            created_at=datetime.utcnow(),
        )

        item_id = queue.enqueue(item)
        assert item_id is not None

        items = queue.dequeue()
        assert len(items) == 1
        assert items[0].doc_id == "doc1"


def test_fifo_order():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queue.db"
        queue = ManualReviewQueue(db_path)

        for i in range(3):
            queue.enqueue(
                ReviewItem(
                    doc_id=f"doc{i}",
                    slice_id=None,
                    phase_name="phase5",
                    reason="test",
                    data_snapshot={},
                    created_at=datetime.utcnow(),
                )
            )

        items = queue.dequeue(limit=3)
        assert [i.doc_id for i in items] == ["doc0", "doc1", "doc2"]


def test_resolve():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queue.db"
        queue = ManualReviewQueue(db_path)

        item_id = queue.enqueue(
            ReviewItem(
                doc_id="doc1",
                slice_id=None,
                phase_name="phase4",
                reason="test",
                data_snapshot={},
                created_at=datetime.utcnow(),
            )
        )

        queue.resolve(item_id, "corrected", {"fixed": True})

        items = queue.dequeue()
        assert len(items) == 0  # no longer pending


def test_stats():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queue.db"
        queue = ManualReviewQueue(db_path)

        id1 = queue.enqueue(
            ReviewItem(
                doc_id="doc1",
                slice_id=None,
                phase_name="phase4",
                reason="test",
                data_snapshot={},
                created_at=datetime.utcnow(),
            )
        )

        id2 = queue.enqueue(
            ReviewItem(
                doc_id="doc2",
                slice_id=None,
                phase_name="phase4",
                reason="test",
                data_snapshot={},
                created_at=datetime.utcnow(),
            )
        )

        queue.resolve(id1, "accepted")
        queue.resolve(id2, "discarded")

        stats = queue.stats()

        assert stats["accepted"] == 1
        assert stats["discarded"] == 1
        assert stats["pending"] == 0
        assert stats["total"] == 2
