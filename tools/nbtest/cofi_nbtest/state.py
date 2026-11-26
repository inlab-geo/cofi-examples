"""Local state for trusted staged notebooks and the latest run."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from cofi_nbtest import __version__
from cofi_nbtest.discover import staged_path
from cofi_nbtest.fileio import atomic_write_text

SCHEMA_VERSION = 1
STATE_RELATIVE = Path("tools") / "nbtest" / "state" / "state.json"
_RECOVERY = "preserve state and staged notebooks; restore this file or use a compatible nbtest version"


class StateError(Exception):
    """State is unreadable, unsupported or unsafe for the requested run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state_path(root: Path) -> Path:
    return Path(root) / STATE_RELATIVE


def empty_state() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "nbtest_version": __version__,
        "notebooks": {},
        "last_run": None,
    }


def load_state(root: Path, require_exists: bool = False) -> Dict[str, Any]:
    path = state_path(root)
    if not path.is_file():
        if require_exists:
            raise StateError(f"{path.relative_to(root)} does not exist; run a notebook first")
        return empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"cannot read {path.relative_to(root)}: {exc}; {_RECOVERY}") from None
    if not isinstance(data, dict):
        raise StateError(f"{path.relative_to(root)} is not a JSON object; {_RECOVERY}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise StateError(
            f"unsupported state schema {data.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}; {_RECOVERY}"
        )
    if not isinstance(data.get("notebooks"), dict):
        raise StateError(f"{path.relative_to(root)} has invalid notebook state; {_RECOVERY}")
    if data.get("last_run") is not None and not isinstance(data["last_run"], dict):
        raise StateError(f"{path.relative_to(root)} has invalid run state; {_RECOVERY}")
    return data


def save_state(root: Path, state: Dict[str, Any]) -> None:
    state["nbtest_version"] = __version__
    path = state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(state, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_staged(
    root: Path,
    state: Dict[str, Any],
    notebooks: Iterable[Path],
    force: bool,
) -> Tuple[Dict[Path, Optional[str]], List[Path]]:
    """Refuse staged files that nbtest cannot prove are unchanged.

    Returns the hashes to re-check immediately before replacement and staged paths that ``--force``
    will replace. Forced paths are for terminal output only.
    """

    conflicts: List[str] = []
    forced: List[Path] = []
    expected: Dict[Path, Optional[str]] = {}
    records = state["notebooks"]
    for original in notebooks:
        staged = staged_path(original)
        if not staged.is_file():
            expected[original] = None
            continue
        relative_original = original.relative_to(root).as_posix()
        relative_staged = staged.relative_to(root).as_posix()
        record = records.get(relative_original)
        recorded_hash = record.get("staged_sha256") if isinstance(record, dict) else None
        actual_hash = sha256_file(staged)
        expected[original] = actual_hash
        if recorded_hash == actual_hash:
            continue
        if force:
            forced.append(staged)
            continue
        if recorded_hash is None:
            reason = "was created before state tracking and may contain unpublished edits"
        else:
            reason = "has changed since nbtest wrote it"
        conflicts.append(f"  {relative_staged}: {reason}")

    if conflicts:
        joined = "\n".join(conflicts)
        raise StateError(
            f"refusing to overwrite untrusted staged notebook(s):\n{joined}\n"
            "inspect or copy out any edits first; --force permanently replaces these files"
        )
    return expected, forced


def begin_run(
    state: Dict[str, Any],
    targets: List[str],
    notebooks: Iterable[Path],
    prerequisites: Set[Path],
    root: Path,
    timeout_s: int,
) -> str:
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S.%fZ")
    state["last_run"] = {
        "id": run_id,
        "mode": "local",
        "status": "running",
        "started": started.isoformat(timespec="seconds"),
        "finished": None,
        "targets": list(targets),
        "timeout_s": timeout_s,
        "exit_code": None,
        "notebooks": [
            {
                "path": notebook.relative_to(root).as_posix(),
                "role": "prerequisite" if notebook in prerequisites else "target",
                "status": "not_run",
            }
            for notebook in notebooks
        ],
        "side_effects": {},
        "side_effects_unavailable": [],
    }
    return run_id


def _run_entry(state: Dict[str, Any], relative_path: str) -> Dict[str, Any]:
    for entry in state["last_run"]["notebooks"]:
        if entry["path"] == relative_path:
            return entry
    raise StateError(f"internal state error: {relative_path} is not in the latest run")


def record_result(
    state: Dict[str, Any],
    root: Path,
    result: Any,
) -> None:
    relative = result.path.relative_to(root).as_posix()
    _run_entry(state, relative)["status"] = result.status

    previous = state["notebooks"].get(relative)
    record = dict(previous) if isinstance(previous, dict) else {}
    record.update(
        {
            "run_id": state["last_run"]["id"],
            "staged_path": result.staged.relative_to(root).as_posix(),
            "staged_written": result.staged_written,
            "duration_s": round(result.duration_s, 1),
            "error": None
            if result.ename is None
            else {"cell": result.cell, "ename": result.ename, "evalue": result.evalue},
            "missing": result.missing,
        }
    )
    if result.staged_written:
        if result.staged_sha256 is None:
            raise StateError("internal state error: staged notebook was written without a hash")
        record["original_sha256"] = result.original_sha256
        record["staged_sha256"] = result.staged_sha256
    else:
        # Keep provenance paired: an older trusted staged file retains the hash of the original it
        # was actually built from. Latest-attempt details above may still be refreshed.
        record.setdefault("original_sha256", None)
        record.setdefault("staged_sha256", None)
    state["notebooks"][relative] = record


def record_side_effects(
    state: Dict[str, Any],
    root: Path,
    notebook: Path,
    side_effects: List[Dict[str, Any]],
) -> None:
    if side_effects:
        relative = notebook.relative_to(root).as_posix()
        state["last_run"]["side_effects"][relative] = side_effects


def record_side_effects_unavailable(state: Dict[str, Any], root: Path, notebook: Path) -> None:
    relative = notebook.relative_to(root).as_posix()
    unavailable = state["last_run"].setdefault("side_effects_unavailable", [])
    if relative not in unavailable:
        unavailable.append(relative)


def record_interrupted(
    state: Dict[str, Any],
    root: Path,
    notebook: Optional[Path],
    side_effects: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if notebook is None:
        return
    relative = notebook.relative_to(root).as_posix()
    _run_entry(state, relative)["status"] = "interrupted"
    record_side_effects(state, root, notebook, side_effects or [])


def finish_run(state: Dict[str, Any], status: str, exit_code: int) -> None:
    run = state["last_run"]
    run["status"] = status
    run["finished"] = utc_now()
    run["exit_code"] = exit_code

