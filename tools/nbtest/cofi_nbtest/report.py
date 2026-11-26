"""Render tools/nbtest/REPORT.md from state.json."""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from cofi_nbtest.fileio import atomic_write_text

REPORT_RELATIVE = Path("tools") / "nbtest" / "REPORT.md"


def _escape(value: Any) -> str:
    return " ".join(str(value or "").split()).replace("|", "\\|")


def _duration(value: Any) -> str:
    if value is None:
        return ""
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}m {remainder:02d}s"


def _detail(state: Dict[str, Any], run_id: str, path: str, status: str) -> str:
    record = state["notebooks"].get(path)
    if not isinstance(record, dict) or record.get("run_id") != run_id:
        return ""
    if status == "skipped":
        return f"missing optional package: `{_escape(record.get('missing'))}`"
    error = record.get("error")
    if isinstance(error, dict):
        where = f"cell {error.get('cell')}: " if error.get("cell") else ""
        return f"{where}{_escape(error.get('ename'))}: {_escape(error.get('evalue'))}"
    if not record.get("staged_written", True):
        return "no staged copy written"
    return ""


def render_report(root: Path, state: Dict[str, Any]) -> str:
    run = state.get("last_run")
    if not isinstance(run, dict):
        return "# nbtest report\n\nNo run has been recorded.\n"

    entries: List[Dict[str, Any]] = run.get("notebooks", [])
    counts = Counter(entry.get("status", "not_run") for entry in entries)
    summary_order = ("ok", "failed", "timeout", "skipped", "interrupted", "not_run")
    summary = ", ".join(f"{counts[name]} {name}" for name in summary_order if counts[name])
    lines = [
        "# nbtest report",
        "",
        f"- Run: `{_escape(run.get('id'))}`",
        f"- Mode: `{_escape(run.get('mode'))}`",
        f"- State: `{_escape(run.get('status'))}`",
        f"- Started: {_escape(run.get('started'))}",
        f"- Finished: {_escape(run.get('finished')) or '—'}",
        f"- Timeout: {run.get('timeout_s')} s per notebook",
        f"- Exit code: {run.get('exit_code') if run.get('exit_code') is not None else '—'}",
        f"- Targets: {', '.join(f'`{_escape(target)}`' for target in run.get('targets', []))}",
        f"- Summary: {summary or 'no notebooks'}",
        "",
        "## Notebooks",
        "",
        "| Notebook | Role | Status | Duration | Details |",
        "| --- | --- | --- | ---: | --- |",
    ]
    run_id = run.get("id")
    for entry in entries:
        path = entry.get("path", "")
        status = entry.get("status", "not_run")
        record = state["notebooks"].get(path)
        record = record if isinstance(record, dict) and record.get("run_id") == run_id else {}
        staged = record.get("staged_path")
        link_target = f"../../{staged or path}"
        duration = _duration(record.get("duration_s"))
        detail = _detail(state, run_id, path, status)
        lines.append(
            f"| [{_escape(path)}]({link_target}) | {_escape(entry.get('role'))} | "
            f"**{_escape(status)}** | {duration} | {detail} |"
        )

    side_effects = run.get("side_effects", {})
    unavailable = set(run.get("side_effects_unavailable", []))
    if side_effects or unavailable:
        lines.extend(["", "## Files changed by notebooks", ""])
        for notebook in [entry.get("path") for entry in entries]:
            effects = side_effects.get(notebook, [])
            if not effects and notebook not in unavailable:
                continue
            lines.append(f"### `{_escape(notebook)}`")
            lines.append("")
            if notebook in unavailable:
                lines.append("- Side-effect information unavailable (Git metadata could not be read).")
            for effect in effects:
                tracked = " — **tracked file warning**" if effect.get("tracked") else ""
                lines.append(
                    f"- {effect.get('kind')}: `{_escape(effect.get('path'))}`{tracked}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(root: Path, state: Dict[str, Any]) -> Path:
    path = Path(root) / REPORT_RELATIVE
    atomic_write_text(path, render_report(root, state))
    return path

