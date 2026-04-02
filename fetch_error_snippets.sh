#!/usr/bin/env bash
# fetch_error_snippets.sh
# Extracts context windows around every line flagged by ruff/mypy/pre-commit.
# Usage: bash fetch_error_snippets.sh [REPO_ROOT]
# REPO_ROOT defaults to the current directory.

set -euo pipefail

REPO="${1:-.}"
CONTEXT=12   # lines of context above AND below each error line
DIVIDER="$(printf '=%.0s' {1..80})"
SEP="$(printf -- '-%.0s' {1..80})"

# Prints a labelled snippet from a file.
# Usage: snippet <file> <line> <label>
snippet() {
    local file="$REPO/$1"
    local line="$2"
    local label="$3"

    if [[ ! -f "$file" ]]; then
        echo "  !! FILE NOT FOUND: $file"
        return
    fi

    local total
    total=$(wc -l < "$file")
    local start=$(( line - CONTEXT < 1 ? 1 : line - CONTEXT ))
    local end=$(( line + CONTEXT > total ? total : line + CONTEXT ))

    echo "  FILE : $1"
    echo "  LABEL: $label"
    echo "  LINES: ${start}-${end}  (error @ ${line})"
    echo "$SEP"
    # nl prints line numbers; awk highlights the error line with >>>
    nl -ba -v"$start" -nrz "$file" \
        | sed -n "${start},${end}p" \
        | awk -v err="$line" '{
            lnum = $1 + 0
            line_text = substr($0, index($0,$2))
            if (lnum == err)
                printf ">>> %6d  %s\n", lnum, line_text
            else
                printf "    %6d  %s\n", lnum, line_text
        }'
    echo ""
}

# Also show the top-of-file imports for a given file (first 40 lines).
imports() {
    local file="$REPO/$1"
    if [[ ! -f "$file" ]]; then return; fi
    echo "  [imports block — first 40 lines of $1]"
    echo "$SEP"
    head -n 40 "$file" | nl -ba -nrz
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 1 — bio_extraction/phases/phase4_ocr.py"
echo "  F821 / name-defined: 'arr' undefined at lines 263-264"
echo "$DIVIDER"
snippet "bio_extraction/phases/phase4_ocr.py" 263 "F821 arr undefined"

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 2 — bio_extraction/phases/phase3_layout.py"
echo "  mypy: wrong CheckpointEngine.load() call signature at lines 43-44"
echo "$DIVIDER"
imports "bio_extraction/phases/phase3_layout.py"
snippet "bio_extraction/phases/phase3_layout.py" 43 "call-arg / arg-type on CheckpointEngine.load()"

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 3 — bio_extraction/phases/phase6_resolution.py"
echo "  attr-defined: ResolutionSettings.sqlite_path (line 39)"
echo "  assignment / return-value errors (lines 175, 204, 219)"
echo "$DIVIDER"
imports "bio_extraction/phases/phase6_resolution.py"
snippet "bio_extraction/phases/phase6_resolution.py"  39  "attr-defined: sqlite_path"
snippet "bio_extraction/phases/phase6_resolution.py" 175  "assignment: None -> int"
snippet "bio_extraction/phases/phase6_resolution.py" 204  "return-value: tuple mismatch"
snippet "bio_extraction/phases/phase6_resolution.py" 219  "assignment: float -> int"

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 4 — bio_extraction/phases/phase2_classification.py"
echo "  no-redef: DocumentType re-defined at line 270"
echo "  arg-type: MockAcquisitionResult passed where AcquisitionResult expected (line 351)"
echo "$DIVIDER"
imports "bio_extraction/phases/phase2_classification.py"
snippet "bio_extraction/phases/phase2_classification.py" 270 "no-redef: DocumentType"
snippet "bio_extraction/phases/phase2_classification.py" 351 "arg-type: MockAcquisitionResult"

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 5 — bio_extraction/phases/phase5_extraction.py"
echo "  call-arg: PatternCache() missing 'cache_path' (line 184)"
echo "  attr-defined: PatternCache.set missing (line 605)"
echo "  attr-defined: ExtractionSettings.ollama_timeout_seconds missing (lines 657, 675)"
echo "$DIVIDER"
imports "bio_extraction/phases/phase5_extraction.py"
snippet "bio_extraction/phases/phase5_extraction.py" 184 "call-arg: PatternCache missing cache_path"
snippet "bio_extraction/phases/phase5_extraction.py" 605 "attr-defined: PatternCache.set"
snippet "bio_extraction/phases/phase5_extraction.py" 657 "attr-defined: ollama_timeout_seconds"
snippet "bio_extraction/phases/phase5_extraction.py" 675 "attr-defined: ollama_timeout_seconds (2nd)"

# Also show PatternCache and ExtractionSettings definitions
echo "  [Searching for PatternCache class definition]"
echo "$SEP"
grep -n "class PatternCache" "$REPO/bio_extraction/phases/phase5_extraction.py" \
    "$REPO"/bio_extraction/**/*.py 2>/dev/null || true
echo ""

echo "  [Searching for ExtractionSettings class definition]"
echo "$SEP"
grep -rn "class ExtractionSettings" "$REPO/bio_extraction/" 2>/dev/null || true
echo ""

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 6 — bio_extraction/phases/phase1_acquisition.py"
echo "  assignment: _FakeSettings vs Settings at line 813"
echo "$DIVIDER"
imports "bio_extraction/phases/phase1_acquisition.py"
snippet "bio_extraction/phases/phase1_acquisition.py" 813 "assignment: _FakeSettings vs Settings"

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 7 — bio_extraction/utilities/bloom_filter.py"
echo "  has-type: Cannot determine type of size/hash_count/bit_array (lines 45-47)"
echo "$DIVIDER"
imports "bio_extraction/utilities/bloom_filter.py"
snippet "bio_extraction/utilities/bloom_filter.py" 45 "has-type: size/hash_count/bit_array"

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 8 — bio_extraction/logging_config.py"
echo "  override / LSP violation: LoggingAdapter.process() signature (line 148)"
echo "$DIVIDER"
imports "bio_extraction/logging_config.py"
snippet "bio_extraction/logging_config.py" 148 "override: process() signature mismatch"

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 9 — bio_extraction/utilities/review_queue.py"
echo "  return-value: got int | None, expected int (line 65)"
echo "$DIVIDER"
snippet "bio_extraction/utilities/review_queue.py" 65 "return-value: int | None"

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 10 — tests/test_e2e_pipeline.py"
echo "  F821 / name-defined: mass missing imports — showing top of file"
echo "$DIVIDER"
imports "tests/test_e2e_pipeline.py"
# Show the key error lines in one pass (lines 38-45, 55-60, 110-115, 140-155,
# 175-190, 195-220, 230-245)
for range_start in 35 53 108 138 173 197 228; do
    snippet "tests/test_e2e_pipeline.py" "$range_start" "missing names block"
done

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 11 — tests/integration/test_phase1_acquisition_integration.py"
echo "  F821: enumerate_local_inputs undefined (line 161)"
echo "$DIVIDER"
imports "tests/integration/test_phase1_acquisition_integration.py"
snippet "tests/integration/test_phase1_acquisition_integration.py" 161 "F821: enumerate_local_inputs"

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 12 — tests/unit/test_layout_phase_unit.py"
echo "  F811: ClassificationResult re-imported (lines 4-5)"
echo "  F821: LayoutPhase undefined (line 10)"
echo "$DIVIDER"
imports "tests/unit/test_layout_phase_unit.py"
snippet "tests/unit/test_layout_phase_unit.py" 4  "F811: duplicate ClassificationResult import"
snippet "tests/unit/test_layout_phase_unit.py" 10 "F821: LayoutPhase undefined"

# ─────────────────────────────────────────────────────────────────────────────
echo "$DIVIDER"
echo "SECTION 13 — CheckpointEngine / AcquisitionResult / DeadLetterQueue definitions"
echo "  (needed to understand correct call signatures)"
echo "$DIVIDER"
echo "  [grep: CheckpointEngine class + load method]"
echo "$SEP"
grep -rn "class CheckpointEngine\|def load" "$REPO/bio_extraction/" 2>/dev/null || true
echo ""
echo "  [grep: class AcquisitionResult]"
echo "$SEP"
grep -rn "class AcquisitionResult" "$REPO/bio_extraction/" 2>/dev/null || true
echo ""
echo "  [grep: class DeadLetterQueue]"
echo "$SEP"
grep -rn "class DeadLetterQueue" "$REPO/bio_extraction/" 2>/dev/null || true
echo ""
echo "  [grep: class ResolutionSettings]"
echo "$SEP"
grep -rn "class ResolutionSettings" "$REPO/bio_extraction/" 2>/dev/null || true
echo ""

echo "$DIVIDER"
echo "Done. Pipe output to a file: bash fetch_error_snippets.sh > snippets.txt"
echo "$DIVIDER"