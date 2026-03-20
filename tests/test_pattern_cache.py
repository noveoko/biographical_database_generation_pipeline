import tempfile
from pathlib import Path

from bio_extraction.utilities.pattern_cache import PatternCache


def test_put_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        cache = PatternCache(path)

        cache.put("fp1", r"\d+")
        pattern = cache.get("fp1")

        assert pattern is not None
        assert pattern.match("123")


def test_invalid_regex_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        cache = PatternCache(path)

        try:
            cache.put("bad", r"[unclosed")
            assert False, "Should raise"
        except ValueError:
            assert True


def test_record_hit_and_stats():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        cache = PatternCache(path)

        cache.put("fp1", r"\d+")
        cache.record_hit("fp1")
        cache.record_hit("fp1")

        stats = cache.stats()

        assert stats["total_patterns"] == 1
        assert stats["most_used_top_10"][0]["hit_count"] == 2


def test_remove():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"
        cache = PatternCache(path)

        cache.put("fp1", r"\d+")
        assert cache.remove("fp1") is True
        assert cache.get("fp1") is None


def test_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.json"

        cache = PatternCache(path)
        cache.put("fp1", r"\d+")

        cache2 = PatternCache(path)
        assert cache2.get("fp1") is not None
