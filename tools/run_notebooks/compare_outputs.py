"""
compare_outputs.py — CoFI notebook output change detector

Reads executed notebooks from the examples directory, extracts numerical
values from cell outputs, compares against a stored baseline, and writes
a Markdown report of significant changes.

Usage:
    python compare_outputs.py \
        --executed-dir /path/to/cofi-examples/examples \
        --baseline    /path/to/validation/baselines/output_baseline.json \
        --output      /path/to/validation/reports/output_changes.md \
        [--report     /path/to/notebook_execution_report.md]

Exit codes:
    0 — ran successfully (changes may or may not exist)
    1 — unrecoverable error (missing required argument, etc.)
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat

# ── Thresholds ──────────────────────────────────────────────────────────────
REL_THRESHOLD = 0.01   # 1 % relative change
ABS_THRESHOLD = 1e-3   # absolute change floor (for near-zero values)

# ── Regex helpers ────────────────────────────────────────────────────────────
# Matches floats / ints including scientific notation, optionally negative
_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")
# Strip ANSI escape codes from terminal output captured in notebooks
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def extract_numbers(text: str) -> list[float]:
    """Return all numbers found in *text* as floats."""
    return [float(m) for m in _NUM_RE.findall(_strip_ansi(text))]


# ── Notebook reading ─────────────────────────────────────────────────────────

def _cell_key(cell_index: int, output_type: str) -> str:
    return f"cell_{cell_index}_{output_type}"


def read_notebook_outputs(nb_path: str) -> dict:
    """
    Return a dict keyed by cell_key → {"numbers": [...], "has_output": bool}.
    Only considers code cells.
    """
    result = {}
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except Exception as e:
        print(f"  [warn] Could not read {nb_path}: {e}", file=sys.stderr)
        return result

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        outputs = cell.get("outputs", [])
        has_output = len(outputs) > 0

        all_numbers: list[float] = []
        for output in outputs:
            otype = output.get("output_type", "")
            if otype == "stream":
                all_numbers.extend(extract_numbers(output.get("text", "")))
            elif otype in ("execute_result", "display_data"):
                data = output.get("data", {})
                text = data.get("text/plain", "")
                if isinstance(text, list):
                    text = "".join(text)
                all_numbers.extend(extract_numbers(text))
            # skip image outputs

        key = f"cell_{idx}"
        result[key] = {"numbers": all_numbers, "has_output": has_output}

    return result


# ── Baseline I/O ─────────────────────────────────────────────────────────────

def load_baseline(path: str) -> dict:
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  [warn] Could not load baseline {path}: {e}", file=sys.stderr)
    return {}


def save_baseline(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Comparison logic ─────────────────────────────────────────────────────────

def is_significant(prev: float, curr: float) -> bool:
    """True when the change between prev and curr exceeds thresholds."""
    abs_diff = abs(curr - prev)
    if abs_diff < ABS_THRESHOLD:
        return False
    denom = max(abs(prev), ABS_THRESHOLD)
    return (abs_diff / denom) >= REL_THRESHOLD


def compare_notebook(nb_rel: str, prev_data: dict, curr_data: dict) -> dict:
    """
    Compare previous and current output snapshots for one notebook.
    Returns {"numerical": [...], "output_presence": [...]}.
    """
    numerical = []
    presence = []

    all_keys = set(prev_data) | set(curr_data)
    for key in sorted(all_keys):
        prev = prev_data.get(key, {})
        curr = curr_data.get(key, {})

        # Output presence change
        prev_has = prev.get("has_output", False)
        curr_has = curr.get("has_output", False)
        if prev_has != curr_has:
            presence.append({
                "cell": key,
                "change": "output added" if curr_has else "output removed",
            })

        # Numerical comparison (only when both have numbers)
        prev_nums = prev.get("numbers", [])
        curr_nums = curr.get("numbers", [])
        if not prev_nums or not curr_nums:
            continue

        # Compare element-wise up to the length of the shorter list
        for i, (p, c) in enumerate(zip(prev_nums, curr_nums)):
            if is_significant(p, c):
                rel = ((c - p) / max(abs(p), ABS_THRESHOLD)) * 100
                numerical.append({
                    "cell": key,
                    "index": i,
                    "previous": p,
                    "current": c,
                    "delta_pct": rel,
                })

    return {"numerical": numerical, "output_presence": presence}


# ── Regression detection from execution report ───────────────────────────────

def parse_regressions(report_path: str, baseline_path: str) -> list[dict]:
    """
    Compare FAILED/WARNING notebooks in the current report against those
    in the baseline report (stored as a sibling key in baseline JSON).
    Returns list of {"notebook": ..., "prev_status": ..., "curr_status": ...}.
    """
    if not report_path or not os.path.isfile(report_path):
        return []

    # Current statuses
    curr_statuses = _parse_statuses_from_report(report_path)

    # Previous statuses stored in baseline
    baseline = load_baseline(baseline_path)
    prev_statuses = baseline.get("_statuses", {})

    regressions = []
    for nb, curr_st in curr_statuses.items():
        prev_st = prev_statuses.get(nb, "UNKNOWN")
        if prev_st == curr_st:
            continue
        # Only flag if current state is worse
        if curr_st in ("FAILED", "WARNING") and prev_st != curr_st:
            regressions.append({"notebook": nb, "prev": prev_st, "curr": curr_st})

    return regressions


def _parse_statuses_from_report(report_path: str) -> dict[str, str]:
    """Extract {notebook_name: status} from a Markdown execution report."""
    statuses = {}
    status_re = re.compile(r"\|\s*(?:\[)?([^|\]]+\.ipynb)(?:\])?\([^)]*\)?\s*\|\s*!\[(PASSED|WARNING|FAILED)\]")
    try:
        with open(report_path) as f:
            for line in f:
                m = status_re.search(line)
                if m:
                    statuses[m.group(1).strip()] = m.group(2)
    except Exception:
        pass
    return statuses


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(output_path: str, all_changes: dict, regressions: list, run_ts: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = [f"# Output Changes Report\n", f"**Run:** {run_ts}\n"]

    # ── Regressions ──
    lines.append("\n## New Status Regressions\n")
    if regressions:
        lines.append("| Notebook | Previous | Current |")
        lines.append("|---|---|---|")
        for r in regressions:
            lines.append(f"| {r['notebook']} | {r['prev']} | {r['curr']} |")
    else:
        lines.append("- None")

    # ── Numerical changes ──
    lines.append("\n## Significant Numerical Changes\n")
    num_rows = []
    for nb_rel, changes in sorted(all_changes.items()):
        for ch in changes["numerical"]:
            sign = "+" if ch["delta_pct"] >= 0 else ""
            num_rows.append(
                f"| {nb_rel} | {ch['cell']} | {ch['index']} "
                f"| {ch['previous']:.6g} | {ch['current']:.6g} "
                f"| {sign}{ch['delta_pct']:.1f}% |"
            )
    if num_rows:
        lines.append("| Notebook | Cell | Value # | Previous | Current | Δ |")
        lines.append("|---|---|---|---|---|---|")
        lines.extend(num_rows)
    else:
        lines.append("- None")

    # ── Output presence changes ──
    lines.append("\n## Cell Output Added / Removed\n")
    presence_rows = []
    for nb_rel, changes in sorted(all_changes.items()):
        for ch in changes["output_presence"]:
            presence_rows.append(f"| {nb_rel} | {ch['cell']} | {ch['change']} |")
    if presence_rows:
        lines.append("| Notebook | Cell | Change |")
        lines.append("|---|---|---|")
        lines.extend(presence_rows)
    else:
        lines.append("- None")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def find_executed_notebooks(examples_dir: str) -> list[str]:
    """Walk examples_dir one level deep and return all .ipynb paths."""
    result = []
    base = Path(examples_dir)
    for subdir in sorted(base.iterdir()):
        if subdir.is_dir():
            result.extend(sorted(subdir.glob("*.ipynb")))
    return [str(p) for p in result]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CoFI notebook outputs to baseline.")
    parser.add_argument("--executed-dir", required=True, help="Path to cofi-examples/examples/")
    parser.add_argument("--baseline", required=True, help="Path to output_baseline.json")
    parser.add_argument("--output", required=True, help="Path to write output_changes.md")
    parser.add_argument("--report", default="", help="Path to notebook_execution_report.md (for regression detection)")
    args = parser.parse_args()

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Load previous baseline
    baseline = load_baseline(args.baseline)

    # Find and read all executed notebooks
    notebooks = find_executed_notebooks(args.executed_dir)
    if not notebooks:
        print(f"[warn] No notebooks found in {args.executed_dir}", file=sys.stderr)

    curr_snapshot: dict[str, dict] = {}
    all_changes: dict[str, dict] = {}

    for nb_path in notebooks:
        nb_rel = os.path.relpath(nb_path, args.executed_dir)
        print(f"  comparing: {nb_rel}")
        curr_data = read_notebook_outputs(nb_path)
        curr_snapshot[nb_rel] = curr_data

        prev_data = baseline.get(nb_rel, {})
        if prev_data:
            changes = compare_notebook(nb_rel, prev_data, curr_data)
            if changes["numerical"] or changes["output_presence"]:
                all_changes[nb_rel] = changes

    # Detect status regressions
    regressions = parse_regressions(args.report, args.baseline)

    # Write report
    write_report(args.output, all_changes, regressions, run_ts)
    print(f"\nOutput changes report written to {args.output}")

    # Update baseline (preserve _statuses from execution report)
    curr_snapshot["_statuses"] = _parse_statuses_from_report(args.report)
    save_baseline(args.baseline, curr_snapshot)
    print(f"Baseline updated at {args.baseline}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
