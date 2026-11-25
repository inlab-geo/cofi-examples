"""Command line: ./nbtest run <targets>."""

import argparse
import importlib.metadata
import importlib.util
import sys
import time
from pathlib import Path
from typing import List, Optional

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


def _cmd_run(args: argparse.Namespace) -> int:
    from cofi_nbtest.discover import TargetError, resolve_targets
    from cofi_nbtest.kernel import private_kernel
    from cofi_nbtest.runner import Interrupted, run_notebook

    try:
        notebooks, prerequisites = resolve_targets(ROOT, args.targets)
    except TargetError as exc:
        print(f"nbtest: {exc}", file=sys.stderr)
        return 2

    print(f"nbtest mode: local (Python {sys.version.split()[0]} at {sys.executable}, cofi {_version('cofi')})")
    print(f"{len(notebooks)} notebook(s), timeout {args.timeout} s per notebook\n")

    width = len(str(len(notebooks)))
    counts = {"ok": 0, "failed": 0, "timeout": 0, "skipped": 0}
    t0 = time.monotonic()
    interrupted = False
    with private_kernel() as kernel_name:
        for i, nb in enumerate(notebooks, start=1):
            rel = nb.relative_to(ROOT).as_posix()
            tag = " (prerequisite)" if nb in prerequisites else ""
            print(f"[{i:>{width}}/{len(notebooks)}] {rel}{tag} ...", end="", flush=True)
            try:
                result = run_notebook(nb, ROOT, kernel_name, args.timeout)
            except (Interrupted, KeyboardInterrupt):
                # KeyboardInterrupt: Ctrl-C outside the kernel run (reading or writing files);
                # staged files are written atomically, so nothing is left half-written.
                print(" interrupted (no staged copy written)")
                interrupted = True
                break
            counts[result.status] += 1
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

    total = sum(counts.values())
    summary = ", ".join(f"{n} {k}" for k, n in counts.items())
    print(f"\n{total} of {len(notebooks)} run: {summary} ({_fmt_duration(time.monotonic() - t0)})")
    if interrupted:
        print("interrupted by Ctrl-C")
        return 130
    return 1 if counts["failed"] or counts["timeout"] else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="nbtest", description="Run cofi-examples notebooks into staged copies.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="execute notebooks into <name>.nbtest.ipynb next to each original")
    run.add_argument("targets", nargs="+", help="'all', example folders, notebook paths, or @list (tools/nbtest/<list>.txt)")
    run.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help=f"seconds per notebook (default {DEFAULT_TIMEOUT_S})")
    args = parser.parse_args(argv)

    missing = _missing_requirements()
    if missing:
        print(
            f"nbtest: missing Python packages in {sys.executable}: {', '.join(missing)}\n"
            f"        install them with: {sys.executable} -m pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    if args.command == "run":
        return _cmd_run(args)
    return 2
