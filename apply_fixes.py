#!/usr/bin/env python3
"""
apply_fixes.py
==============
Applies every ruff / mypy / pre-commit error identified in snippets.txt.

Run from the project root:

    python apply_fixes.py            # applies all fixes
    python apply_fixes.py --dry-run  # previews changes, writes nothing

Each fix is labelled with the source file, error code, and a one-line reason.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

REPO = Path(".")
DRY = "--dry-run" in sys.argv
ERRORS: list[str] = []


def _read(rel: str) -> Optional[str]:
    p = REPO / rel
    if not p.exists():
        ERRORS.append(f"FILE NOT FOUND: {rel}")
        return None
    return p.read_text(encoding="utf-8")


def _write(rel: str, src: str) -> None:
    if DRY:
        print(f"  [DRY-RUN] would write {rel}")
        return
    (REPO / rel).write_text(src, encoding="utf-8")


def patch(rel: str, old: str, new: str, label: str) -> bool:
    """Replace the first occurrence of *old* with *new* in file *rel*."""
    src = _read(rel)
    if src is None:
        return False
    if old not in src:
        ERRORS.append(f"PATTERN NOT FOUND in {rel}: {label!r}")
        return False
    _write(rel, src.replace(old, new, 1))
    print(f"  ✓  {rel}  —  {label}")
    return True


def patch_re(rel: str, pattern: str, replacement: str, label: str, flags: int = 0) -> bool:
    """Replace first regex match of *pattern* in file *rel*."""
    src = _read(rel)
    if src is None:
        return False
    new_src, n = re.subn(pattern, replacement, src, count=1, flags=flags)
    if n == 0:
        ERRORS.append(f"REGEX NOT FOUND in {rel}: {label!r}")
        return False
    _write(rel, new_src)
    print(f"  ✓  {rel}  —  {label}")
    return True


def ensure_import(rel: str, import_line: str, after: str, label: str) -> bool:
    """Insert *import_line* after the first line matching *after* regex."""
    src = _read(rel)
    if src is None:
        return False
    if import_line.strip() in src:
        print(f"  ·  {rel}  —  already present: {label}")
        return True
    new_src, n = re.subn(
        rf"({re.escape(after)})",
        rf"\1\n{import_line}",
        src,
        count=1,
    )
    if n == 0:
        ERRORS.append(f"ANCHOR NOT FOUND in {rel}: {label!r}")
        return False
    _write(rel, new_src)
    print(f"  ✓  {rel}  —  {label}")
    return True


# ---------------------------------------------------------------------------
# FIX 1 — phase4_ocr.py  F821 / name-defined  ('arr' should be '_arr')
# The variable is assigned as `_arr = np.array(...)` two lines above,
# but the two subsequent lines accidentally reference the bare `arr`.
# ---------------------------------------------------------------------------
print("\n── FIX 1: phase4_ocr.py — arr → _arr ─────────────────────────────────")

patch(
    "bio_extraction/phases/phase4_ocr.py",
    "thresh = threshold_sauvola(arr,",
    "thresh = threshold_sauvola(_arr,",
    "F821: threshold_sauvola(arr → _arr)",
)
patch(
    "bio_extraction/phases/phase4_ocr.py",
    "binary = (arr > thresh)",
    "binary = (_arr > thresh)",
    "F821: (arr > thresh) → (_arr > thresh)",
)

# ---------------------------------------------------------------------------
# FIX 2 — phase3_layout.py  call-arg / arg-type
# CheckpointEngine.load() is an *instance* method; calling it as a class
# method passes the first positional arg ("phase1_acquisition") as `self`.
# We also need to widen the local type to Optional[AcquisitionResult] and
# add a None-guard so mypy is happy.
# ---------------------------------------------------------------------------
print("\n── FIX 2: phase3_layout.py — CheckpointEngine class-method call ───────")

# Dynamically discover the first constructor param name from checkpoint.py
# so the generated code uses the right keyword.
_ck_src = _read("bio_extraction/checkpoint.py") or ""
_ck_init = re.search(
    r"class CheckpointEngine[^:]*:.*?def __init__\(self(?:,\s*)?(.*?)\)",
    _ck_src,
    re.DOTALL,
)
_ck_first_param = "checkpoint_dir"   # safe default
if _ck_init:
    _raw = _ck_init.group(1).strip()
    _m = re.match(r"(\w+)", _raw)
    if _m:
        _ck_first_param = _m.group(1)
print(f"  detected CheckpointEngine first param: {_ck_first_param!r}")

# Discover whether settings exposes the checkpoint dir via acquisition sub-model
# or at the top level.  We look for the most likely attribute name.
_cfg_src = _read("bio_extraction/config.py") or ""
_possible_attrs = re.findall(r"checkpoint[_\w]*\s*[:=]", _cfg_src, re.IGNORECASE)
_settings_attr = "checkpoint_dir"
if _possible_attrs:
    _settings_attr = _possible_attrs[0].rstrip(" :=").strip()
print(f"  detected settings checkpoint attr:     {_settings_attr!r}")

patch(
    "bio_extraction/phases/phase3_layout.py",
    # ── old ──────────────────────────────────────────────────────────────────
    'acquisition: AcquisitionResult = CheckpointEngine.load(\n'
    '            "phase1_acquisition", doc.doc_id, AcquisitionResult\n'
    '        )',
    # ── new ──────────────────────────────────────────────────────────────────
    f'_ck_engine = CheckpointEngine(self.settings.{_settings_attr})\n'
    f'        acquisition: Optional[AcquisitionResult] = _ck_engine.load(\n'
    f'            "phase1_acquisition", doc.doc_id, AcquisitionResult\n'
    f'        )\n'
    f'        if acquisition is None:\n'
    f'            raise PhaseError(\n'
    f'                self.phase_name, doc.doc_id, "No acquisition checkpoint found"\n'
    f'            )',
    "call-arg/arg-type: instantiate engine, widen type, add None-guard",
)

# ---------------------------------------------------------------------------
# FIX 3 — bio_extraction/config.py  attr-defined
# ResolutionSettings is missing the `sqlite_path` field that phase6 reads.
# We also add `ollama_timeout_seconds` to ExtractionSettings here so that
# Fix 5c (phase5) is handled in one place.
# ---------------------------------------------------------------------------
print("\n── FIX 3: config.py — add missing settings fields ──────────────────────")

# 3a: ResolutionSettings.sqlite_path
if "sqlite_path" not in _cfg_src:
    patch_re(
        "bio_extraction/config.py",
        r"(class ResolutionSettings\s*\([^)]*\)\s*:\s*\n)",
        r'\1    sqlite_path: str = "data/resolution.db"\n',
        "attr-defined: add sqlite_path to ResolutionSettings",
    )
else:
    print("  ·  config.py — sqlite_path already present in ResolutionSettings")

# 3b: ExtractionSettings.ollama_timeout_seconds  (also fixes FIX 5c)
if "ollama_timeout_seconds" not in _cfg_src:
    patch_re(
        "bio_extraction/config.py",
        r"(class ExtractionSettings\s*\([^)]*\)\s*:\s*\n)",
        r'\1    ollama_timeout_seconds: int = 60\n',
        "attr-defined: add ollama_timeout_seconds to ExtractionSettings",
    )
else:
    print("  ·  config.py — ollama_timeout_seconds already present in ExtractionSettings")

# ---------------------------------------------------------------------------
# FIX 4 — phase6_resolution.py  assignment + return-value type errors
# ---------------------------------------------------------------------------
print("\n── FIX 4: phase6_resolution.py — type annotation fixes ────────────────")

# 4a: `best_id = None` assigned to an inferred-int variable.
#     Add an explicit Optional[int] annotation.
patch(
    "bio_extraction/phases/phase6_resolution.py",
    "        best_id = None\n",
    "        best_id: Optional[int] = None\n",
    "assignment: best_id needs Optional[int] annotation",
)

# 4b: `return cursor.lastrowid, True, None`
#     cursor.lastrowid is int | None; expected int.
#     After a successful INSERT lastrowid is never None, so assert.
patch(
    "bio_extraction/phases/phase6_resolution.py",
    "        return cursor.lastrowid, True, None",
    "        assert cursor.lastrowid is not None, \"INSERT returned no lastrowid\"\n"
    "        return cursor.lastrowid, True, None",
    "return-value: assert lastrowid not None before returning",
)

# 4c: `best_score = -1` inferred as int, later assigned float.
patch(
    "bio_extraction/phases/phase6_resolution.py",
    "        best_score = -1\n",
    "        best_score: float = -1.0\n",
    "assignment: best_score float annotation",
)

# 4d: Optional import — phase6 now uses Optional explicitly
_p6_src = _read("bio_extraction/phases/phase6_resolution.py") or ""
if "Optional" not in _p6_src:
    patch(
        "bio_extraction/phases/phase6_resolution.py",
        "from typing import List, Optional, Tuple",
        "from typing import List, Optional, Tuple",  # already there? check below
        "Optional already in typing import",
    )
# More safely: ensure Optional is in the typing import line
patch_re(
    "bio_extraction/phases/phase6_resolution.py",
    r"from typing import (List(?:, Optional)?(?:, Tuple)?)",
    lambda m: m.group(0) if "Optional" in m.group(0)  # type: ignore[arg-type]
              else m.group(0).replace("from typing import List", "from typing import List, Optional"),
    "ensure Optional is imported",
)

# ---------------------------------------------------------------------------
# FIX 5 — phase2_classification.py  no-redef + arg-type
# ---------------------------------------------------------------------------
print("\n── FIX 5: phase2_classification.py — DocumentType redef + mock arg ─────")

# 5a: Local `class DocumentType` in the __main__ demo block redefines the
#     imported name.  Rename the demo copy to avoid the collision.
patch(
    "bio_extraction/phases/phase2_classification.py",
    "# Enums matching contracts.py\n"
    "class DocumentType(enum.Enum):",
    "# Enums matching contracts.py\n"
    "class _DemoDocumentType(enum.Enum):  # renamed to avoid shadowing the import",
    "no-redef: rename local DocumentType to _DemoDocumentType",
)
# Update usages of the renamed class within the same demo block
# (only the __main__ section will use _DemoDocumentType after this)
# The real DocumentType from contracts is already imported and takes precedence
# outside __main__.

# 5b: MockAcquisitionResult is a plain dataclass, not an AcquisitionResult
#     (Pydantic model). Add a type: ignore to the call site.
patch(
    "bio_extraction/phases/phase2_classification.py",
    "    result = phase.run(acq_data)",
    "    result = phase.run(acq_data)  # type: ignore[arg-type]",
    "arg-type: suppress mock type mismatch in demo block",
)

# ---------------------------------------------------------------------------
# FIX 6 — phase5_extraction.py  call-arg + attr-defined
# ---------------------------------------------------------------------------
print("\n── FIX 6: phase5_extraction.py — PatternCache + ExtractionSettings ─────")

# 6a: PatternCache() called with no args but requires cache_path.
#     Inspect utilities/pattern_cache.py to find the actual param name.
_pc_src = _read("bio_extraction/utilities/pattern_cache.py") or ""
_pc_init = re.search(r"def __init__\(self(?:,\s*)?(.*?)\)", _pc_src)
_pc_first_param = "cache_path"  # default
if _pc_init:
    _raw = _pc_init.group(1).strip()
    _m = re.match(r"(\w+)", _raw)
    if _m:
        _pc_first_param = _m.group(1)
print(f"  detected PatternCache first param: {_pc_first_param!r}")

# Discover where ExtractionSettings stores the cache path
_ext_cache_attr = "pattern_cache_path"  # fallback
_attr_match = re.search(r"(cache_path|pattern_cache[\w]*)\s*:", _cfg_src)
if _attr_match:
    _ext_cache_attr = _attr_match.group(1)
print(f"  detected ExtractionSettings cache attr: {_ext_cache_attr!r}")

# ExtractionSettings may not have a cache_path field at all; add one if missing
_cfg_src_fresh = _read("bio_extraction/config.py") or ""
if _ext_cache_attr not in _cfg_src_fresh:
    patch_re(
        "bio_extraction/config.py",
        r"(class ExtractionSettings\s*\([^)]*\)\s*:\s*\n(?:    ollama_timeout_seconds[^\n]*\n)?)",
        r'\1    pattern_cache_path: str = "data/pattern_cache.pkl"\n',
        "add pattern_cache_path field to ExtractionSettings",
    )
    _ext_cache_attr = "pattern_cache_path"

patch(
    "bio_extraction/phases/phase5_extraction.py",
    "self._cache: PatternCache = pattern_cache or PatternCache()",
    f"self._cache: PatternCache = pattern_cache or PatternCache(\n"
    f"                Path(self._cfg.{_ext_cache_attr})\n"
    f"            )",
    "call-arg: pass cache_path to PatternCache()",
)

# Also ensure Path is imported in phase5
_p5_src = _read("bio_extraction/phases/phase5_extraction.py") or ""
if "from pathlib import Path" not in _p5_src:
    patch_re(
        "bio_extraction/phases/phase5_extraction.py",
        r"(^import hashlib\n)",
        r"from pathlib import Path\n\1",
        "add Path import",
        flags=re.MULTILINE,
    )

# 6b: self._cache.set() does not exist on PatternCache.
#     Determine the correct write method from the source.
_pc_methods = re.findall(r"def (\w+)\(self", _pc_src)
print(f"  PatternCache public methods found: {_pc_methods}")
_write_method = "set"  # optimistic default
for _candidate in ("put", "store", "add", "__setitem__", "set"):
    if _candidate in _pc_methods:
        _write_method = _candidate
        break
print(f"  using PatternCache write method: {_write_method!r}")

if _write_method != "set":
    patch(
        "bio_extraction/phases/phase5_extraction.py",
        "self._cache.set(fingerprint, compiled)",
        f"self._cache.{_write_method}(fingerprint, compiled)",
        f"attr-defined: .set() → .{_write_method}()",
    )
else:
    # .set() either already exists or nothing better was found.
    # Add it to PatternCache as a thin alias if it is genuinely missing.
    if "def set(" not in _pc_src and _pc_src:
        # Find the class body end and inject a .set() method
        patch_re(
            "bio_extraction/utilities/pattern_cache.py",
            r"(def put\(self, key[^:]*:[^,]*, value[^:]*:[^)]*\)[^:]*:\n(?:.*\n)*?.*return.*\n)",
            r'\1\n    # Alias expected by phase5\n    def set(self, key, value) -> None:\n        self.put(key, value)\n',
            "add .set() alias to PatternCache",
            flags=re.DOTALL,
        )

# ---------------------------------------------------------------------------
# FIX 7 — phase1_acquisition.py  assignment (_FakeSettings vs Settings)
# The lambda in the __main__ test block returns a _FakeSettings but the
# variable is typed as Callable[..., Settings].  A type: ignore suppresses
# this in test/demo code without touching production types.
# ---------------------------------------------------------------------------
print("\n── FIX 7: phase1_acquisition.py — _FakeSettings type mismatch ──────────")

patch(
    "bio_extraction/phases/phase1_acquisition.py",
    "_cfg.get_settings = lambda *args, **kwargs: _FakeSettings()",
    "_cfg.get_settings = lambda *args, **kwargs: _FakeSettings()  # type: ignore[assignment]",
    "assignment: suppress _FakeSettings vs Settings in demo block",
)

# ---------------------------------------------------------------------------
# FIX 8 — bloom_filter.py  has-type
# mypy cannot determine the types of self.size, self.hash_count, self.bit_array
# because they are set inside conditional branches of __init__ (one branch via
# from_file(), the other directly).  Adding class-level annotations resolves
# this without changing any runtime behaviour.
# ---------------------------------------------------------------------------
print("\n── FIX 8: bloom_filter.py — class-level type annotations ───────────────")

patch(
    "bio_extraction/utilities/bloom_filter.py",
    "    DEFAULT_CAPACITY = 200_000\n"
    "    DEFAULT_ERROR_RATE = 0.01\n",
    "    DEFAULT_CAPACITY = 200_000\n"
    "    DEFAULT_ERROR_RATE = 0.01\n"
    "\n"
    "    # mypy requires explicit annotations when attrs are set in branches\n"
    "    size: int\n"
    "    hash_count: int\n"
    "    bit_array: bitarray\n",
    "has-type: add class-level type annotations for size/hash_count/bit_array",
)

# ---------------------------------------------------------------------------
# FIX 9 — logging_config.py  override / LSP violation
# The inner _PhaseAdapter.process() uses dict[str, Any] for its kwargs param,
# but LoggerAdapter.process() declares MutableMapping[str, Any].  dict is a
# subtype of MutableMapping so this is safe, but mypy enforces Liskov at the
# call-site signature.  Widen to MutableMapping.
# ---------------------------------------------------------------------------
print("\n── FIX 9: logging_config.py — process() LSP signature fix ──────────────")

# Ensure MutableMapping is importable
_lc_src = _read("bio_extraction/logging_config.py") or ""
if "MutableMapping" not in _lc_src:
    patch(
        "bio_extraction/logging_config.py",
        "from typing import Any",
        "from typing import Any, MutableMapping",
        "add MutableMapping to typing imports",
    )

patch(
    "bio_extraction/logging_config.py",
    "    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:",
    "    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:",
    "override: widen process() kwargs to MutableMapping to satisfy LSP",
)

# ---------------------------------------------------------------------------
# FIX 10 — review_queue.py  return-value
# cursor.lastrowid is int | None (SQLite spec), but the method is annotated
# as returning int.  After a committed INSERT lastrowid is always non-None,
# so an assert is the right tool — it documents the invariant and satisfies
# mypy without hiding errors behind a silent `or 0` default.
# ---------------------------------------------------------------------------
print("\n── FIX 10: review_queue.py — lastrowid int | None ──────────────────────")

patch(
    "bio_extraction/utilities/review_queue.py",
    "        self.conn.commit()\n"
    "        return cursor.lastrowid",
    "        self.conn.commit()\n"
    "        assert cursor.lastrowid is not None, \"enqueue INSERT returned no lastrowid\"\n"
    "        return cursor.lastrowid",
    "return-value: assert lastrowid not None after INSERT commit",
)

# ---------------------------------------------------------------------------
# FIX 11 — tests/test_e2e_pipeline.py  F821 — all missing imports
# The test file uses PhaseError, PipelineError, CheckpointError,
# CheckpointEngine, AcquisitionResult, ClassificationResult, LayoutResult,
# PhaseOneInput, DocumentSource, and DeadLetterQueue without importing them.
# ---------------------------------------------------------------------------
print("\n── FIX 11: tests/test_e2e_pipeline.py — add missing imports ─────────────")

_E2E_IMPORTS = """\
from bio_extraction.checkpoint import CheckpointEngine
from bio_extraction.contracts import (
    AcquisitionResult,
    ClassificationResult,
    LayoutResult,
    PhaseOneInput,
    DocumentSource,
)
from bio_extraction.dead_letter import DeadLetterQueue
from bio_extraction.exceptions import CheckpointError, PhaseError, PipelineError"""

# Insert after the last stdlib import in the block (`import pytest`)
ensure_import(
    "tests/test_e2e_pipeline.py",
    _E2E_IMPORTS,
    "import pytest",
    "add all missing project imports",
)

# Fix Phase1Acquisition / Phase2Classification etc. not existing.
# The real classes are AcquisitionPhase, ClassificationPhase, etc.
# Add public aliases to each phase module so the test imports work.
_PHASE_ALIASES = [
    ("bio_extraction/phases/phase1_acquisition.py", "AcquisitionPhase",   "Phase1Acquisition"),
    ("bio_extraction/phases/phase2_classification.py", "ClassificationPhase", "Phase2Classification"),
    ("bio_extraction/phases/phase3_layout.py",         "LayoutPhase",         "Phase3Layout"),
    ("bio_extraction/phases/phase4_ocr.py",            None,                  "Phase4OCR"),
    ("bio_extraction/phases/phase5_extraction.py",     None,                  "Phase5Extraction"),
    ("bio_extraction/phases/phase6_resolution.py",     "ResolutionPhase",     "Phase6Resolution"),
]

print("\n── FIX 11b: add PhaseN aliases to each phase module ────────────────────")
for _rel, _known_name, _alias in _PHASE_ALIASES:
    _src = _read(_rel)
    if _src is None:
        continue
    if f"{_alias} = " in _src:
        print(f"  ·  {_rel} — {_alias} alias already present")
        continue
    # Discover class name if not given
    if _known_name is None:
        _candidates = re.findall(r"^class (\w+Phase)\b", _src, re.MULTILINE)
        _known_name = _candidates[0] if _candidates else None
    if _known_name is None:
        ERRORS.append(f"Cannot determine phase class in {_rel} for alias {_alias}")
        continue
    if _known_name not in _src:
        ERRORS.append(f"{_known_name} not found in {_rel}; cannot add alias {_alias}")
        continue
    # Append alias at end of file
    _new = _src.rstrip() + f"\n\n# Public alias used by test_e2e_pipeline.py\n{_alias} = {_known_name}\n"
    _write(_rel, _new)
    print(f"  ✓  {_rel} — added {_alias} = {_known_name}")

# ---------------------------------------------------------------------------
# FIX 12 — tests/integration/test_phase1_acquisition_integration.py
# enumerate_local_inputs is used at line 161 but never imported.
# ---------------------------------------------------------------------------
print("\n── FIX 12: test_phase1_acquisition_integration.py — missing import ──────")

patch(
    "tests/integration/test_phase1_acquisition_integration.py",
    "from bio_extraction.phases.phase1_acquisition import AcquisitionPhase",
    "from bio_extraction.phases.phase1_acquisition import AcquisitionPhase, enumerate_local_inputs",
    "F821: add enumerate_local_inputs to import",
)

# ---------------------------------------------------------------------------
# FIX 13 — tests/unit/test_layout_phase_unit.py  F811 + F821
# Line 4 imports ClassificationResult from contracts; line 5 re-imports it
# from phase2_classification (overwriting the first import — harmless at
# runtime but ruff F811).  LayoutPhase is used but never imported (F821).
# ---------------------------------------------------------------------------
print("\n── FIX 13: tests/unit/test_layout_phase_unit.py — F811 + F821 ───────────")

# Remove the duplicate import (line 5)
patch(
    "tests/unit/test_layout_phase_unit.py",
    "from bio_extraction.contracts import DocumentType, ClassificationResult\n"
    "from bio_extraction.phases.phase2_classification import ClassificationResult\n",
    "from bio_extraction.contracts import DocumentType, ClassificationResult\n",
    "F811: remove duplicate ClassificationResult import",
)

# Add LayoutPhase import right after the contracts import
ensure_import(
    "tests/unit/test_layout_phase_unit.py",
    "from bio_extraction.phases.phase3_layout import LayoutPhase",
    "from bio_extraction.contracts import DocumentType, ClassificationResult",
    "F821: add missing LayoutPhase import",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
if ERRORS:
    print(f"Completed with {len(ERRORS)} item(s) needing manual review:\n")
    for e in ERRORS:
        print(f"  ⚠  {e}")
else:
    print("All fixes applied successfully — no manual steps required.")
print("=" * 70)
print("\nNext steps:")
print("  1. git diff                      # review every change")
print("  2. pre-commit run --all-files    # verify hooks pass")
print("  3. pytest                        # run the test suite")
if DRY:
    print("\n  (DRY-RUN mode — no files were modified)")