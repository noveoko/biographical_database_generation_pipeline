# bio_extraction

Biographical entity extraction pipeline for Polish historical PDFs —
directories (*spisy adresowe*), newspapers (*gazety*), and civil records
(*akta stanu cywilnego*).

The pipeline runs six sequential phases: **Acquisition → Classification →
Layout → OCR → Extraction → Resolution**. Each phase is an independent
module; all shared types live in `contracts.py`. The infrastructure
(runner, checkpoints, dead-letter queue) is production-ready. The phase
logic itself ships as documented stubs, ready for implementation.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | **3.11 or 3.12** | f-string type aliases; `tomllib` in stdlib |
| Tesseract OCR | **5.x** | System binary — see [below](#2-install-tesseract) |
| Polish Tesseract language packs | `pol` + `pol_frak` | `pol_frak` covers Fraktur typefaces in 19th-century records |
| Ollama | latest | Only needed when implementing Phase 5 extraction |

---

## Step-by-step setup

### 1. Clone / unpack the project

If you received the project as a tarball:

```bash
tar -xzf bio_extraction.tar.gz
cd bio_extraction
```

If you are working from a git repository:

```bash
git clone <your-repo-url>
cd bio_extraction
```

---

### 2. Install Tesseract

Tesseract is a **system-level binary**, not a Python package. Install it
before creating the virtual environment.

**Ubuntu / Debian (WSL included):**

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-pol
```

`pol_frak` (Fraktur) is distributed separately — check availability first:

```bash
apt-cache search tesseract | grep pol
```

If `tesseract-ocr-pol-frak` appears, install it:

```bash
sudo apt install -y tesseract-ocr-pol-frak
```

Otherwise download the trained data manually:

```bash
# tessdata must match your Tesseract version (use 'tessdata' for 5.x)
sudo wget -P /usr/share/tesseract-ocr/5/tessdata/ \
  https://github.com/tesseract-ocr/tessdata/raw/main/pol_frak.traineddata
```

Verify both packs are available:

```bash
tesseract --list-langs
# Expected output includes: pol  pol_frak
```

**macOS (Homebrew):**

```bash
brew install tesseract
# Polish pack
brew install tesseract-lang   # installs all language packs including pol
```

---

### 3. Create a virtual environment

The project requires Python 3.11+. Use whichever Python you have at that
version:

```bash
python3.11 -m venv .venv          # or python3.12, depending on your system
source .venv/bin/activate         # Linux / macOS / WSL
# .venv\Scripts\activate          # Windows PowerShell
```

Confirm the active Python:

```bash
python --version
# Python 3.11.x  (or 3.12.x)
```

---

### 4. Install Python dependencies

Install the package itself in **editable mode** together with all dev tools:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

`-e` (editable install) means changes to source files take effect
immediately without reinstalling. `[dev]` pulls in pytest, ruff, and mypy.

Verify the install:

```bash
python -c "import bio_extraction; print('OK')"
```

---

### 5. Verify the test suite

```bash
pytest tests/ -v
```

Expected output — all 20 tests green:

```
tests/test_e2e_pipeline.py::TestExceptions::test_phase_error_is_pipeline_error PASSED
tests/test_e2e_pipeline.py::TestExceptions::test_checkpoint_error_is_pipeline_error PASSED
tests/test_e2e_pipeline.py::TestCheckpointEngine::test_save_and_load_roundtrip PASSED
... (17 more) ...
20 passed in 0.35s
```

These tests cover the infrastructure only — they use fake phase doubles and
require no Tesseract, Ollama, or network access.

---

### 6. (Optional) Install and start Ollama

Ollama is required when Phase 5 (`phase5_extraction.py`) is implemented.
Skip this step until you are ready to work on that phase.

```bash
# Linux / WSL
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama
```

Pull the default model (configured in `config.yaml`):

```bash
ollama pull mistral:7b
```

Start the server (runs on port 11434 by default):

```bash
ollama serve
```

Confirm it is reachable:

```bash
curl http://localhost:11434/api/tags
```

---

## Project layout

```
bio_extraction/
├── pyproject.toml                    # build metadata + dependencies
├── config.yaml                       # runtime config (edit this for your run)
├── bio_extraction/
│   ├── contracts.py                  # ALL shared Pydantic models (CP1–CP5 + Phase 6)
│   ├── protocol.py                   # PhaseProtocol abstract base class
│   ├── runner.py                     # PipelineRunner — queues, checkpoints, dead-letter
│   ├── checkpoint.py                 # CheckpointEngine — atomic JSON serialization
│   ├── dead_letter.py                # DeadLetterQueue — structured failure records
│   ├── config.py                     # Settings model loaded from config.yaml
│   ├── exceptions.py                 # Custom exception hierarchy
│   ├── logging_config.py             # JSON-lines file + human stderr logging
│   ├── phases/
│   │   ├── phase1_acquisition.py     # STUB — fetch PDFs (local or CommonCrawl)
│   │   ├── phase2_classification.py  # STUB — predict document type
│   │   ├── phase3_layout.py          # STUB — segment pages into content slices
│   │   ├── phase4_ocr.py             # STUB — run Tesseract per slice
│   │   ├── phase5_extraction.py      # STUB — parse PersonEntity records
│   │   └── phase6_resolution.py      # STUB — deduplicate → SQLite
│   └── utilities/
│       ├── bloom_filter.py           # STUB — SurnameBloomFilter
│       ├── pattern_cache.py          # STUB — PatternCache (regex fingerprint store)
│       └── review_queue.py           # STUB — ManualReviewQueue
├── checkpoints/                      # created at runtime by CheckpointEngine
├── dead_letter/                      # created at runtime by DeadLetterQueue
├── input_pdfs/                       # place your PDFs here for local mode
├── data/                             # created at runtime (SQLite DB, bloom filter, cache)
└── tests/
    ├── conftest.py                   # shared pytest fixtures
    └── test_e2e_pipeline.py          # 20 infrastructure smoke tests
```

---

## Configuration

All runtime parameters live in `config.yaml`. The most common settings to
change before a first run:

```yaml
# Which source to read PDFs from: "local" or "commoncrawl"
source: local

# For local mode: put PDFs here
input_dir: ./input_pdfs

# OCR: lower threshold = fewer review flags, higher = more
ocr:
  confidence_threshold: 0.65
  tesseract_langs: "pol+pol_frak"

# Phase 5 extraction: change model if you have a different one in Ollama
extraction:
  ollama_model: "mistral:7b"
  ollama_url: "http://localhost:11434"
```

The config is loaded and validated at startup via `Settings.from_yaml()`.
Any missing required key raises `ConfigError` with a clear message.

---

## Running the pipeline

The pipeline does not have a CLI entry point yet (that comes in a later
task). To run it directly from Python:

```python
from bio_extraction.config import Settings
from bio_extraction.runner import PipelineRunner
from bio_extraction.phases.phase1_acquisition import Phase1Acquisition
# import the other phases once implemented ...

settings = Settings.from_yaml("config.yaml")

runner = PipelineRunner(
    phases=[
        Phase1Acquisition(),
        # Phase2Classification(),
        # Phase3Layout(),
        # Phase4OCR(),
        # Phase5Extraction(),
        # Phase6Resolution(),
    ],
    settings=settings,
)

runner.run_all()
```

Drop your PDFs in `./input_pdfs/` before running.

---

## Implementing a phase

Each stub in `bio_extraction/phases/` has a module-level docstring describing
its input contract, output contract, and key design decisions. The
implementation checklist for any phase:

1. Open the stub file and read the docstring fully.
2. Add constructor arguments if the phase needs config (e.g. `settings: Settings`).
3. Replace the `raise NotImplementedError(...)` in `run()` with real logic.
4. Return the correct Pydantic model on success, `None` to silently discard
   the document, or raise an exception to route it to the dead-letter queue.
5. Use `get_phase_logger(self.phase_name)` from `logging_config.py` for all
   log output — do **not** use `print()`.
6. **Do not import from another phase module.** All shared types are in
   `contracts.py`.

---

## Development tools

Run the linter:

```bash
ruff check bio_extraction/ tests/
```

Run the type checker:

```bash
mypy bio_extraction/
```

Run tests with coverage:

```bash
pytest tests/ -v --cov=bio_extraction --cov-report=term-missing
```

---

## Checkpoint and dead-letter directories

These are created automatically at runtime. Their structure:

```
checkpoints/
  phase1_acquisition/
    <doc_id>.json     ← AcquisitionResult for one document
  phase2_classification/
    <doc_id>.json     ← ClassificationResult
  ...

dead_letter/
  phase3_layout/
    <doc_id>.json     ← timestamp, error class, full traceback, input snapshot
```

To re-process a document that was previously checkpointed, delete its
checkpoint file:

```bash
rm checkpoints/phase3_layout/<doc_id>.json
```

To wipe all checkpoints and start fresh:

```bash
python -c "from bio_extraction.checkpoint import CheckpointEngine; CheckpointEngine().clear()"
```

---

## Logging

Every run appends structured JSON-lines to `./logs/pipeline.jsonl`. Each
line is a self-contained object:

```json
{"timestamp": "2024-03-15T12:00:01.123456", "level": "INFO",
 "phase_name": "phase1_acquisition", "doc_id": "deadbeef01234567",
 "message": "Processing."}
```

Human-readable output also streams to stderr during the run.

To tail the log during a run:

```bash
tail -f logs/pipeline.jsonl | python -m json.tool
```
