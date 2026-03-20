# REQUIRES: mmh3, bitarray

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import List

import mmh3
from bitarray import bitarray


class SurnameBloomFilter:
    """
    Probabilistic Bloom filter for Polish surname validation.

    Design goals:
    - False positives are acceptable (~1% target).
    - False negatives must not occur.
    - Optimized for ~200,000 surnames.

    Default parameters:
    - m ≈ 1,917,011 bits (~234 KB)
    - k ≈ 7 hash functions

    Normalization:
    - Lowercase
    - Strip Polish diacritics (ą→a, ć→c, etc.)
    - Trim whitespace

    Suggested training data sources (user-provided):
    - "Słownik nazwisk współcześnie w Polsce używanych"
    - FamilySearch / GenoPro surname datasets
    - Extracted surnames from validated OCR pipeline outputs
    """

    DEFAULT_CAPACITY = 200_000
    DEFAULT_ERROR_RATE = 0.01

    def __init__(self, bloom_path: Path | None = None):
        """Load from disk if provided, otherwise initialize empty filter."""
        if bloom_path:
            loaded = self.from_file(bloom_path)
            self.size = loaded.size
            self.hash_count = loaded.hash_count
            self.bit_array = loaded.bit_array
        else:
            self.size, self.hash_count = self._optimal_params(
                self.DEFAULT_CAPACITY, self.DEFAULT_ERROR_RATE
            )
            self.bit_array = bitarray(self.size)
            self.bit_array.setall(0)

    # ---------------- PUBLIC API ----------------

    def train(self, surnames: List[str]) -> None:
        """Add surnames to the filter (normalized)."""
        for surname in surnames:
            norm = self._normalize(surname)
            if not norm:
                continue
            for i in range(self.hash_count):
                idx = mmh3.hash(norm, i) % self.size
                self.bit_array[idx] = 1

    def check(self, surname: str) -> bool:
        """Return True if POSSIBLY present, False if DEFINITELY NOT."""
        norm = self._normalize(surname)
        if not norm:
            return False

        for i in range(self.hash_count):
            idx = mmh3.hash(norm, i) % self.size
            if not self.bit_array[idx]:
                return False
        return True

    def check_batch(self, surnames: List[str]) -> float:
        """Return fraction of surnames that pass the filter."""
        if not surnames:
            return 0.0
        hits = sum(1 for s in surnames if self.check(s))
        return hits / len(surnames)

    def save(self, path: Path) -> None:
        """Persist filter to disk."""
        with path.open("wb") as f:
            header = f"{self.size},{self.hash_count}\n".encode()
            f.write(header)
            self.bit_array.tofile(f)

    @classmethod
    def from_file(cls, path: Path) -> "SurnameBloomFilter":
        """Load filter from disk."""
        with path.open("rb") as f:
            header = f.readline().decode().strip()
            size, hash_count = map(int, header.split(","))

            bf = cls.__new__(cls)
            bf.size = size
            bf.hash_count = hash_count
            bf.bit_array = bitarray()
            bf.bit_array.fromfile(f)

            # Ensure correct length (bitarray may over-read last byte)
            if len(bf.bit_array) > size:
                bf.bit_array = bf.bit_array[:size]

            return bf

    @classmethod
    def from_corpus_file(cls, corpus_path: Path) -> "SurnameBloomFilter":
        """Build filter from text file (one surname per line)."""
        bf = cls()
        with corpus_path.open("r", encoding="utf-8") as f:
            surnames = [line.strip() for line in f if line.strip()]
        bf.train(surnames)
        return bf

    # ---------------- INTERNAL ----------------

    def _normalize(self, text: str) -> str:
        """Normalize surname."""
        if not text:
            return ""

        text = text.lower().strip()

        # Strip diacritics using Unicode decomposition
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")

        return text

    def _optimal_params(self, n: int, p: float) -> tuple[int, int]:
        """Compute optimal Bloom filter size (m) and hash count (k)."""
        m = -int(n * math.log(p) / (math.log(2) ** 2))
        k = max(1, int((m / n) * math.log(2)))
        return m, k


if __name__ == "__main__":
    # Minimal smoke test
    bf = SurnameBloomFilter()

    surnames = ["Kowalski", "Nowak", "Wiśniewski", "Zieliński"]
    bf.train(surnames)

    print("Check Kowalski:", bf.check("Kowalski"))  # True
    print("Check Unknown:", bf.check("Xyzabc"))  # Likely False

    print("Batch hit rate:", bf.check_batch(["Kowalski", "Xyzabc"]))

    tmp_path = Path("bloom_test.bin")
    bf.save(tmp_path)

    bf2 = SurnameBloomFilter.from_file(tmp_path)
    print("Reload check Kowalski:", bf2.check("kowalski"))
