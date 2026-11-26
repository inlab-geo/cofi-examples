"""Command line for local notebook execution and reporting."""

import argparse
import importlib.metadata
import importlib.util
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]  # cofi_nbtest -> nbtest -> tools -> repo root
REQUIRED = ("nbclient", "nbformat", "ipykernel", "jupyter_client", "jupyter_core")
DEFAULT_TIMEOUT_S = 10800  # 3 hours per notebook (maintainer decision 2026-11-25)


def _missing_requirements() -> List[str]:
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m {secs:02d}s"


def _one_line(text: Optional[str], limit: int = 160) -> str:
    line = " ".join((text or "").split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _persist(state: Dict[str, Any]) -> None:
    from cofi_nbtest.report import write_report
    from cofi_nbtest.state import save_state

    save_state(ROOT, state)
    write_report(ROOT, state)


def _print_side_effects(effects: List[Dict[str, Any]]) -> None:
    for effect in effects:
        warning = " [TRACKED FILE WARNING]" if effect["tracked"] else ""
        print(f"      {effect['kind']} {effect['path']}{warning}")


def _cmd_run(args: argparse.Namespace) -> int:
    from cofi_nbtest.discover import TargetError, resolve_targets, staged_path
    from cofi_nbtest.effects import EffectsError, compare_snapshots, snapshot_folder
    from cofi_nbtest.kernel import private_kernel
    from cofi_nbtest.runner import Interrupted, run_notebook
    from cofi_nbtest.state import (
        StateError,
        begin_run,
        finish_run,
        load_state,
        preflight_staged,
        record_interrupted,
        record_result,
        record_side_effects,
        record_side_effects_unavailable,
    )

    try:
        notebooks, prerequisites = resolve_targets(ROOT, args.targets)
        state = load_state(ROOT)
        expected_staged, forced = preflight_staged(ROOT, state, notebooks, args.force)
    except (TargetError, StateError) as exc:
        print(f"nbtest: {exc}", file=sys.stderr)
        return 2

    if forced:
        print("nbtest: --force will replace untrusted staged notebook(s):")
        for path in forced:
            print(f"  {path.relative_to(ROOT).as_posix()}")
        print()

    print(f"nbtest mode: local (Python {sys.version.split()[0]} at {sys.executable}, cofi {_version('cofi')})")
    print(f"{len(notebooks)} notebook(s), timeout {args.timeout} s per notebook\n")

    begin_run(state, args.targets, notebooks, prerequisites, ROOT, args.timeout)
    width = len(str(len(notebooks)))
    counts = {"ok": 0, "failed": 0, "timeout": 0, "skipped": 0}
    t0 = time.monotonic()
    current: Optional[Path] = None
    before: Optional[Dict[str, Dict[str, Any]]] = None
    final_status: Optional[str] = None
    exit_code = 2
    previous_sigint: Any = None

    try:
        # Save the complete plan before a kernel starts, then update it after every notebook.
        _persist(state)
        with private_kernel() as kernel_name:
            for i, nb in enumerate(notebooks, start=1):
                current = nb
                rel = nb.relative_to(ROOT).as_posix()
                try:
                    before = snapshot_folder(ROOT, nb.parent, exclude=staged_path(nb))
                except (EffectsError, OSError) as exc:
                    before = None
                    record_side_effects_unavailable(state, ROOT, nb)
                    print(f"nbtest: side effects unavailable for {rel}: {exc}", file=sys.stderr)
                tag = " (prerequisite)" if nb in prerequisites else ""
                print(f"[{i:>{width}}/{len(notebooks)}] {rel}{tag} ...", end="", flush=True)

                result = run_notebook(
                    nb,
                    ROOT,
                    kernel_name,
                    args.timeout,
                    expected_staged[nb],
                )
                # Trust the staged file as soon as execution returns. If side-effect inspection is
                # interrupted afterwards, the completed result is still recorded accurately.
                record_result(state, ROOT, result)
                current = None
                counts[result.status] += 1
                effects: List[Dict[str, Any]] = []
                if before is not None:
                    try:
                        after = snapshot_folder(ROOT, nb.parent, exclude=staged_path(nb))
                    except (EffectsError, OSError) as exc:
                        record_side_effects_unavailable(state, ROOT, nb)
                        print(f"nbtest: side effects unavailable for {rel}: {exc}", file=sys.stderr)
                    else:
                        effects = compare_snapshots(before, after)
                        record_side_effects(state, ROOT, nb, effects)
                before = None
                _persist(state)

                took = _fmt_duration(result.duration_s)
                if result.status == "ok":
                    print(f" ok ({took})")
                elif result.status == "skipped":
                    print(f" skipped (missing: {result.missing}) ({took})")
                else:
                    where = f"cell {result.cell}: " if result.cell else ""
                    print(f" {result.status} ({took})\n      {where}{result.ename}: {_one_line(result.evalue)}")
                    if result.kernel_log:
                        print("      kernel output (last lines):")
                        for line in result.kernel_log.splitlines():
                            print(f"        {line}")
                _print_side_effects(effects)

        final_status = "complete"
        exit_code = 1 if counts["failed"] or counts["timeout"] else 0
    except (Interrupted, KeyboardInterrupt):
        # A notebook may have changed files before it was interrupted. Capture those changes when
        # possible, but never let side-effect reporting prevent a clean Ctrl-C exit.
        try:
            previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (AttributeError, ValueError):
            previous_sigint = None
        record_interrupted(state, ROOT, current)
        effects: List[Dict[str, Any]] = []
        if current is not None and before is not None:
            try:
                after = snapshot_folder(ROOT, current.parent, exclude=staged_path(current))
                effects = compare_snapshots(before, after)
            except (EffectsError, OSError) as exc:
                record_side_effects_unavailable(state, ROOT, current)
                print(f"\nnbtest: could not inspect side effects after interruption: {exc}", file=sys.stderr)
            else:
                record_side_effects(state, ROOT, current, effects)
        if current is not None:
            print(" interrupted (staged copy left unchanged)")
        else:
            print("\ninterrupted while recording results")
        _print_side_effects(effects)
        final_status = "interrupted"
        exit_code = 130
    finally:
        # Unexpected exceptions propagate to the terminal, but state still records a safely
        # aborted run. Internal exception diagnostics deliberately do not enter state.json.
        if final_status is None:
            finish_run(state, "aborted", 2)
        else:
            finish_run(state, final_status, exit_code)
        try:
            _persist(state)
        finally:
            if previous_sigint is not None:
                signal.signal(signal.SIGINT, previous_sigint)

    total = sum(counts.values())
    summary = ", ".join(f"{n} {key}" for key, n in counts.items())
    print(f"\n{total} of {len(notebooks)} run: {summary} ({_fmt_duration(time.monotonic() - t0)})")
    if final_status == "interrupted":
        print("interrupted by Ctrl-C")
    return exit_code


def _cmd_report() -> int:
    from cofi_nbtest.report import write_report
    from cofi_nbtest.state import StateError, load_state

    try:
        state = load_state(ROOT, require_exists=True)
        path = write_report(ROOT, state)
    except StateError as exc:
        print(f"nbtest: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="nbtest", description="Run cofi-examples notebooks into staged copies.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="execute notebooks into <name>.nbtest.ipynb next to each original")
    run.add_argument(
        "targets",
        nargs="+",
        help="'all', example folders, notebook paths, or @list (tools/nbtest/<list>.txt)",
    )
    run.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"seconds per notebook (default {DEFAULT_TIMEOUT_S})",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="replace staged notebooks whose saved hash cannot be trusted",
    )
    sub.add_parser("report", help="rebuild tools/nbtest/REPORT.md from saved state")
    args = parser.parse_args(argv)

    try:
        if args.command == "report":
            return _cmd_report()

        missing = _missing_requirements()
        if missing:
            print(
                f"nbtest: missing Python packages in {sys.executable}: {', '.join(missing)}\n"
                f"        install them with: {sys.executable} -m pip install {' '.join(missing)}",
                file=sys.stderr,
            )
            return 2
        return _cmd_run(args)
    except Exception:
        # The run path has already marked a started command aborted in its finally block. Keep the
        # traceback in the terminal rather than duplicating internal diagnostics in state.json.
        traceback.print_exc()
        return 2
