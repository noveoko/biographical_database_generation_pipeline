"""
config.py
=========
Loads ``config.yaml`` from the project root and exposes a typed ``Settings``
Pydantic model.

Usage
-----
    from bio_extraction.config import get_settings

    settings = get_settings()
    print(settings.source)           # "local"
    print(settings.ocr.confidence_threshold)  # 0.65

The settings object is cached after the first call.  Pass ``reload=True``
to force a re-read (useful in tests).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from bio_extraction.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class CommonCrawlSettings(BaseModel):
    index_url: str = Field(description="Base URL of the Common Crawl CDX/Index API endpoint.")
    query_domains: list[str] = Field(description="Domains to filter for when querying the CC index.")
    mime_filter: str = Field(description="MIME type filter passed to the CC index query (e.g. 'application/pdf').")


class OCRSettings(BaseModel):
    confidence_threshold: float = Field(ge=0.0, le=1.0, description="Minimum Tesseract confidence to accept a slice without flagging for review.")
    tesseract_langs: str = Field(description="Tesseract language string passed via -l (e.g. 'pol+pol_frak').")


class ExtractionSettings(BaseModel):
    ollama_model: str = Field(description="Ollama model tag used for LLM-assisted extraction (e.g. 'mistral:7b').")
    ollama_url: str = Field(description="Base URL of the local Ollama inference server.")
    bloom_filter_path: Path = Field(description="Path to the serialised SurnameBloomFilter file.")
    pattern_cache_path: Path = Field(description="Path to the JSON pattern cache file.")
    bloom_hit_rate_threshold: float = Field(ge=0.0, le=1.0, description="Minimum Bloom filter hit rate to prefer cached regex over LLM fallback.")


class ResolutionSettings(BaseModel):
    fuzzy_match_threshold: float = Field(ge=0.0, le=1.0, description="Minimum similarity score (0–1) to treat two PersonEntity records as the same person.")
    db_path: Path = Field(description="Path to the SQLite database file.")


# ---------------------------------------------------------------------------
# Root settings model
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    source: str = Field(description="Acquisition source mode: 'local' or 'commoncrawl'.")
    input_dir: Path = Field(description="Directory scanned for PDFs in local mode.")
    checkpoint_dir: Path = Field(description="Root directory for phase checkpoint files.")
    dead_letter_dir: Path = Field(description="Root directory for dead-letter failure records.")
    log_dir: Path = Field(description="Directory for pipeline log files.")
    commoncrawl: CommonCrawlSettings = Field(description="Settings specific to Common Crawl acquisition.")
    ocr: OCRSettings = Field(description="Settings for the OCR phase.")
    extraction: ExtractionSettings = Field(description="Settings for the extraction phase.")
    resolution: ResolutionSettings = Field(description="Settings for the resolution phase.")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_cached_settings: Settings | None = None


def get_settings(
    config_path: Path | None = None,
    *,
    reload: bool = False,
) -> Settings:
    """
    Load and return the pipeline settings.

    Parameters
    ----------
    config_path:
        Path to a ``config.yaml`` file.  Defaults to ``./config.yaml``
        relative to the project root.
    reload:
        If True, discard the cached instance and re-read from disk.
        Useful in tests that need isolated configs.

    Raises
    ------
    ConfigError
        If the file is missing, not valid YAML, or fails Pydantic validation.
    """
    global _cached_settings

    if _cached_settings is not None and not reload:
        return _cached_settings

    path = config_path or _DEFAULT_CONFIG_PATH
    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    try:
        _cached_settings = Settings.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"Config validation failed: {exc}") from exc

    return _cached_settings
