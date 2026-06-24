#!/bin/bash
# Build HTML pages (with outputs) from marimo notebook sources
# Outputs are committed in rendered/ and served through GitHub Pages
#
# Usage:
#   ./build.sh                    # build all notebooks (full mode)
#   ./build.sh --fast             # build all notebooks in fast/dev mode
#   ./build.sh educator.py        # build a single notebook
#   ./build.sh --fast practitioner.py  # build one notebook in fast mode

usage() {
    cat <<EOF
Usage: ./build.sh [--fast] [notebook.py ...]

Build HTML pages (with outputs) from marimo notebook sources.
Outputs are committed in rendered/ and served through GitHub Pages.

Options:
  --fast    Use fast/development mode (1% rays, minimal iterations)
  --help    Show this help message

Examples:
  ./build.sh                       Build all notebooks (full mode)
  ./build.sh --fast                Build all notebooks in fast/dev mode
  ./build.sh educator.py           Build a single notebook
  ./build.sh --fast practitioner.py  Build one notebook in fast mode
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/rendered"
NOTEBOOK_DIR="$SCRIPT_DIR/notebooks"

NOTEBOOKS=()
MARIMO_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--fast" ]; then
        MARIMO_ARGS=(-- --mode fast)
    else
        NOTEBOOKS+=("$arg")
    fi
done

if [ ${#NOTEBOOKS[@]} -eq 0 ]; then
    NOTEBOOKS=(educator.py practitioner.py developer.py)
fi

cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_DIR"

TOTAL=${#NOTEBOOKS[@]}
MODE=$( [ ${#MARIMO_ARGS[@]} -gt 0 ] && echo "fast" || echo "full" )
MODE_SUFFIX=$( [ ${#MARIMO_ARGS[@]} -gt 0 ] && echo "_fast" || echo "" )

# ── Ephemeral virtualenv ─────────────────────────────────────────────────────
# A fresh venv is created for every build and deleted when the script exits —
# whether it succeeds, fails, or is interrupted (Ctrl+C / kill).
# If you already have a venv active, it will NOT be modified; this script runs
# in a subprocess and only affects its own environment.
VENV_DIR=""
cleanup() {
    if [ -n "${VENV_DIR:-}" ] && [ -d "$VENV_DIR" ]; then
        echo ""
        echo "Removing temporary virtualenv..."
        rm -rf "$VENV_DIR"
    fi
}
trap cleanup EXIT INT TERM

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install Python 3.10–3.13 and try again."
    exit 1
fi

VENV_DIR="$(mktemp -d)/cofi-build-venv"

echo "Creating temporary virtualenv..."
if ! python3 -m venv "$VENV_DIR"; then
    echo "Error: failed to create virtualenv."
    echo "  Make sure python3-venv is installed: sudo apt install python3-venv  (Debian/Ubuntu)"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Installing dependencies from requirements.txt..."
if ! pip install --quiet -r "$SCRIPT_DIR/requirements.txt"; then
    echo "Error: dependency installation failed."
    echo "  Check the output above for details."
    echo "  You may need build tools: sudo apt install build-essential  (Debian/Ubuntu)"
    exit 1
fi

# Patch arviz stats_utils for NumPy 2.0+ compatibility.
# NumPy 2.0 changed 0-d array indexing to return Python scalars; arviz's
# make_ufunc inner loop calls .item() on the result, which fails when the
# result is already a plain Python float.
python3 - <<'PYEOF' || true
import arviz, pathlib
hits = list(pathlib.Path(arviz.__file__).parent.rglob('stats_utils.py'))
if not hits:
    print("  arviz patch: stats_utils.py not found, skipping")
else:
    target = hits[0]
    src = target.read_text()
    patched = src.replace(
        'out_idx = out_idx.item()',
        'out_idx = out_idx.item() if hasattr(out_idx, "item") else out_idx'
    )
    if patched != src:
        target.write_text(patched)
        print(f"  Applied arviz NumPy 2.0 patch to {target}")
    else:
        print("  arviz patch: already applied or pattern not found")
PYEOF
echo ""
# ── End venv setup ───────────────────────────────────────────────────────────

# Force non-interactive matplotlib backend — prevents macOS Cocoa GUI operations
# in subprocess builds, which can hang without a proper AppKit event loop.
export MPLBACKEND=Agg

echo "Building $TOTAL notebook(s) [mode: $MODE]"
echo "Output directory: $OUTPUT_DIR"
echo ""

BUILT=0
FAILED=0
for i in "${!NOTEBOOKS[@]}"; do
    notebook="${NOTEBOOKS[$i]}"
    num=$((i + 1))
    notebook_path="$NOTEBOOK_DIR/$notebook"

    if [ ! -f "$notebook_path" ]; then
        echo "Error: $notebook not found in $NOTEBOOK_DIR"
        echo "Available notebooks:"
        ls "$NOTEBOOK_DIR"/*.py 2>/dev/null | xargs -n1 basename
        exit 1
    fi

    name="${notebook%.py}"
    output="$OUTPUT_DIR/${name}${MODE_SUFFIX}.html"
    LOG_FILE="$OUTPUT_DIR/${name}.log"

    echo "[$num/$TOTAL] Building $notebook ..."

    START_TIME=$SECONDS
    if marimo export html "$notebook_path" -o "$output" "${MARIMO_ARGS[@]}" 2>&1 | tee "$LOG_FILE"; then
        ELAPSED=$(( SECONDS - START_TIME ))
        if [ -f "$output" ]; then
            size=$(du -h "$output" | cut -f1)
            size_kb=$(du -k "$output" | cut -f1)
            echo "[$num/$TOTAL] Done: $output ($size, ${ELAPSED}s)"
            if [ "$size_kb" -lt 100 ]; then
                echo "  WARNING: output is only ${size_kb}KB — notebook may have failed silently"
                echo "  See log: $LOG_FILE"
            fi
        fi
        BUILT=$((BUILT + 1))

        # Full-mode only: also export a Jupyter notebook with outputs
        if [ ${#MARIMO_ARGS[@]} -eq 0 ]; then
            ipynb_output="$OUTPUT_DIR/${name}.ipynb"
            if marimo export ipynb "$notebook_path" -o "$ipynb_output" --include-outputs 2>/dev/null; then
                echo "[$num/$TOTAL] Jupyter: $ipynb_output"
            else
                echo "[$num/$TOTAL] WARNING: Jupyter export failed for $notebook"
            fi
        fi
    else
        ELAPSED=$(( SECONDS - START_TIME ))
        echo "[$num/$TOTAL] FAILED: $notebook (${ELAPSED}s)"
        echo "  See log: $LOG_FILE"
        FAILED=$((FAILED + 1))
        exit 1
    fi
    echo ""
done

echo "=========================================="
echo "Build complete: $BUILT/$TOTAL succeeded [mode: $MODE]"
echo "=========================================="
