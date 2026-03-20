import tempfile
from pathlib import Path

from bio_extraction.utilities.bloom_filter import SurnameBloomFilter


def test_train_and_check_basic():
    bf = SurnameBloomFilter()
    bf.train(["Kowalski", "Nowak"])

    assert bf.check("Kowalski") is True
    assert bf.check("Nowak") is True
    assert bf.check("NonexistentSurname") is False


def test_normalization_diacritics():
    bf = SurnameBloomFilter()
    bf.train(["Wiśniewski"])

    assert bf.check("Wisniewski") is True  # diacritics stripped
    assert bf.check("WIŚNIEWSKI") is True


def test_check_batch():
    bf = SurnameBloomFilter()
    bf.train(["Kowalski", "Nowak"])

    rate = bf.check_batch(["Kowalski", "Nowak", "Fake"])
    assert 0.66 <= rate <= 1.0  # allow false positives


def test_save_and_load():
    bf = SurnameBloomFilter()
    bf.train(["Kowalski"])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bf.bin"
        bf.save(path)

        bf2 = SurnameBloomFilter.from_file(path)

        assert bf2.check("Kowalski") is True
        assert bf2.check("Fake") is False
