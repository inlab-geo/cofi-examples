"""Detect files created, modified or deleted by one notebook run."""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class EffectsError(Exception):
    """The notebook folder could not be snapshotted."""


def _git_paths(root: Path, folder: Path, tracked_only: bool) -> Set[str]:
    relative_folder = folder.relative_to(root).as_posix()
    args = ["git", "ls-files", "-z", "--cached"]
    if not tracked_only:
        args.extend(["--others", "--exclude-standard"])
    args.extend(["--", relative_folder])
    try:
        completed = subprocess.run(args, cwd=root, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EffectsError(f"cannot list files below {relative_folder}: {exc}") from None
    return {
        item.decode("utf-8", errors="surrogateescape")
        for item in completed.stdout.split(b"\0")
        if item
    }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_symlink():
        digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_folder(
    root: Path,
    folder: Path,
    exclude: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Hash tracked and nonignored untracked files in one notebook folder."""

    root = Path(root).resolve()
    folder = Path(folder).resolve()
    exclude_relative = Path(exclude).resolve().relative_to(root).as_posix() if exclude else None
    tracked = _git_paths(root, folder, tracked_only=True)
    visible = _git_paths(root, folder, tracked_only=False)
    result: Dict[str, Dict[str, Any]] = {}
    for relative in sorted(visible):
        path = root / relative
        if relative == exclude_relative:
            continue
        if path.name.startswith(".") and path.name.endswith(".tmp"):
            continue
        if not path.exists() and not path.is_symlink():
            continue
        if not path.is_file() and not path.is_symlink():
            continue
        result[relative] = {"digest": _digest(path), "tracked": relative in tracked}
    return result


def compare_snapshots(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return compact side effects; snapshot hashes never leave this function."""

    effects: List[Dict[str, Any]] = []
    for relative in sorted(set(before) | set(after)):
        if relative not in before:
            kind = "created"
        elif relative not in after:
            kind = "deleted"
        elif before[relative]["digest"] != after[relative]["digest"]:
            kind = "modified"
        else:
            continue
        item = after.get(relative, before.get(relative, {}))
        effects.append(
            {"path": relative, "kind": kind, "tracked": bool(item.get("tracked"))}
        )
    return effects

