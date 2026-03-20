# REQUIRES: (none — stdlib only)

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Pattern, Any


class PatternCache:
    """
    JSON-backed cache mapping structural fingerprints to validated regex patterns.

    Features:
    - Thread-safe via Lock
    - Auto-prunes invalid regex on load
    - Tracks usage statistics (hit_count, last_used)
    - Auto-save on every put() and every 100 record_hit() calls
    """

    def __init__(self, cache_path: Path):
        """Load existing cache from disk, or initialize empty."""
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._hit_counter = 0

        self._data: Dict[str, Any] = {
            "patterns": {},
            "metadata": {"version": 1, "total_patterns": 0},
        }

        if cache_path.exists():
            self._load()

    # ---------------- PUBLIC API ----------------

    def get(self, fingerprint: str) -> Optional[Pattern]:
        """Return compiled regex for this fingerprint, or None."""
        with self._lock:
            entry = self._data["patterns"].get(fingerprint)
            if not entry:
                return None

            try:
                pattern = re.compile(entry["regex"])
                return pattern
            except re.error:
                # corrupted pattern — remove it
                self._data["patterns"].pop(fingerprint, None)
                self._update_metadata()
                return None

    def put(
        self,
        fingerprint: str,
        pattern_str: str,
        metadata: dict | None = None,
    ) -> None:
        """Store a validated pattern."""
        try:
            re.compile(pattern_str)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        now = datetime.utcnow().isoformat()

        with self._lock:
            entry = {
                "regex": pattern_str,
                "created_at": now,
                "hit_count": 0,
                "last_used": None,
            }

            if metadata:
                entry.update(metadata)

            self._data["patterns"][fingerprint] = entry
            self._update_metadata()
            self.save()

    def record_hit(self, fingerprint: str) -> None:
        """Increment hit count for a fingerprint."""
        with self._lock:
            entry = self._data["patterns"].get(fingerprint)
            if not entry:
                return

            entry["hit_count"] = entry.get("hit_count", 0) + 1
            entry["last_used"] = datetime.utcnow().isoformat()

            self._hit_counter += 1

            if self._hit_counter % 100 == 0:
                self.save()

    def remove(self, fingerprint: str) -> bool:
        """Remove a pattern. Returns True if it existed."""
        with self._lock:
            existed = fingerprint in self._data["patterns"]
            if existed:
                self._data["patterns"].pop(fingerprint)
                self._update_metadata()
                self.save()
            return existed

    def stats(self) -> dict:
        """Return summary statistics."""
        with self._lock:
            patterns = self._data["patterns"]

            total = len(patterns)

            # most used top 10
            sorted_patterns = sorted(
                patterns.items(),
                key=lambda x: x[1].get("hit_count", 0),
                reverse=True,
            )[:10]

            most_used = [
                {
                    "fingerprint": fp,
                    "hit_count": data.get("hit_count", 0),
                }
                for fp, data in sorted_patterns
            ]

            # oldest & newest
            def parse_time(entry, key):
                val = entry.get(key)
                return datetime.fromisoformat(val) if val else None

            created_times = [
                parse_time(v, "created_at") for v in patterns.values() if v.get("created_at")
            ]

            oldest = min(created_times).isoformat() if created_times else None
            newest = max(created_times).isoformat() if created_times else None

            return {
                "total_patterns": total,
                "most_used_top_10": most_used,
                "oldest": oldest,
                "newest": newest,
            }

    def save(self) -> None:
        """Persist cache to disk."""
        with self._lock:
            self._update_metadata()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ---------------- INTERNAL ----------------

    def _load(self) -> None:
        """Load cache from disk and validate regex patterns."""
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return  # corrupted file → start fresh

        valid_patterns = {}

        for fp, entry in data.get("patterns", {}).items():
            try:
                re.compile(entry["regex"])
                valid_patterns[fp] = entry
            except re.error:
                continue  # skip invalid

        self._data = {
            "patterns": valid_patterns,
            "metadata": data.get("metadata", {"version": 1}),
        }
        self._update_metadata()

    def _update_metadata(self) -> None:
        """Update metadata counters."""
        self._data["metadata"]["total_patterns"] = len(self._data["patterns"])

    # ---------------- CLASS METHODS ----------------


if __name__ == "__main__":
    # Minimal smoke test
    cache = PatternCache(Path("pattern_cache_test.json"))

    fp = "abc123"
    regex = r"(?P<name>[A-Z][a-z]+)"

    cache.put(fp, regex, {"source_doc_id": "doc1"})
    pattern = cache.get(fp)

    if pattern:
        print(pattern.match("Jan"))

    for _ in range(5):
        cache.record_hit(fp)

    print(cache.stats())

    cache.remove(fp)
    cache.save()