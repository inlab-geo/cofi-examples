"""Find notebooks, resolve command-line targets, apply dependencies.txt order and prerequisites.

Scope: notebooks exactly one folder below examples/ and tutorials/. Staged copies
(*.nbtest.ipynb) and hidden files or folders are never notebooks.
"""

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

SCOPES = ("examples", "tutorials")
STAGED_SUFFIX = ".nbtest.ipynb"
DEPENDENCIES_FILE = "dependencies.txt"
LIST_DIR = Path("tools") / "nbtest"


class TargetError(Exception):
    """A target or dependencies.txt cannot be resolved."""


def is_notebook_name(name: str) -> bool:
    return name.endswith(".ipynb") and not name.endswith(STAGED_SUFFIX) and not name.startswith(".")


def staged_path(original: Path) -> Path:
    """Return the disposable output path for an original notebook."""
    return original.with_name(original.name[: -len(".ipynb")] + STAGED_SUFFIX)


def _example_folders(root: Path) -> Iterable[Path]:
    for top in SCOPES:
        base = root / top
        if not base.is_dir():
            continue
        for sub in sorted(base.iterdir(), key=lambda p: p.name):
            if sub.is_dir() and not sub.name.startswith("."):
                yield sub


def read_dependencies(folder: Path) -> List[str]:
    """Notebook names listed in folder/dependencies.txt, in order (empty if there is none).

    Blank lines and "#" comments are ignored. Every other line must name an existing notebook in
    the folder; anything else (a typo such as "helpr.ipnyb", a missing file) raises TargetError, so
    a mistake can never silently change the order.
    """
    deps_file = folder / DEPENDENCIES_FILE
    if not deps_file.is_file():
        return []
    names: List[str] = []
    for number, line in enumerate(deps_file.read_text(encoding="utf-8").splitlines(), start=1):
        name = line.split("#", 1)[0].strip()
        if not name:
            continue
        if not is_notebook_name(name) or not (folder / name).is_file():
            raise TargetError(f"{deps_file}, line {number}: {name!r} is not a notebook in this folder")
        if name not in names:
            names.append(name)
    return names


def folder_order(folder: Path) -> List[Path]:
    """Notebooks of one folder: dependencies.txt entries first, in order, then the rest alphabetically."""
    deps = read_dependencies(folder)
    others = sorted(p.name for p in folder.iterdir() if p.is_file() and is_notebook_name(p.name) and p.name not in deps)
    return [folder / name for name in deps + others]


def discover_all(root: Path) -> List[Path]:
    root = Path(root).resolve()
    result: List[Path] = []
    for folder in _example_folders(root):
        result.extend(folder_order(folder))
    return result


def _scope_error(original: str) -> TargetError:
    return TargetError(
        f"{original}: outside nbtest's scope (only notebooks one folder below examples/ and tutorials/, "
        f"or such a folder)"
    )


def _resolve_path(root: Path, path: Path, original: str) -> List[Path]:
    path = path.resolve()
    if not path.exists():
        raise TargetError(f"{original}: no such file or folder")
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        raise _scope_error(original) from None
    if path.is_dir():
        if len(parts) == 2 and parts[0] in SCOPES and not parts[1].startswith("."):
            return folder_order(path)
        raise _scope_error(original)
    if len(parts) == 3 and parts[0] in SCOPES and not parts[1].startswith(".") and is_notebook_name(parts[2]):
        return [path]
    raise _scope_error(original)


def _read_list(root: Path, name: str) -> List[Path]:
    list_file = root / LIST_DIR / f"{name}.txt"
    if not name or not list_file.is_file():
        raise TargetError(f"@{name}: list file {list_file} not found")
    result: List[Path] = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            result.extend(_resolve_path(root, root / entry, f"{list_file.name}: {entry}"))
    return result


def resolve_targets(root: Path, targets: List[str], cwd: Optional[Path] = None) -> Tuple[List[Path], Set[Path]]:
    """Resolve command-line targets to an ordered list of notebooks to run.

    targets: "all", example folders, notebook paths (relative to cwd or absolute), or "@name" for
    the list file tools/nbtest/<name>.txt (one path per line relative to the repo root, "#" starts
    a comment).

    Returns (notebooks, prerequisites): notebooks grouped by folder in order of first appearance,
    each folder in folder_order(), with dependencies.txt prerequisites added; prerequisites is the
    subset that was added only as a prerequisite.
    """
    root = Path(root).resolve()
    cwd = Path(cwd) if cwd is not None else Path(os.getcwd())
    if not targets:
        raise TargetError("no targets given (use 'all', a folder, notebook paths or @list)")

    selected: List[Path] = []
    for target in targets:
        if target == "all":
            found = discover_all(root)
        elif target.startswith("@"):
            found = _read_list(root, target[1:])
        else:
            path = Path(target)
            found = _resolve_path(root, path if path.is_absolute() else cwd / path, target)
        selected.extend(found)

    # Remove duplicates, keep first occurrence; group by folder in order of first appearance.
    wanted: Dict[Path, List[Path]] = {}
    seen: Set[Path] = set()
    for nb in selected:
        if nb in seen:
            continue
        seen.add(nb)
        wanted.setdefault(nb.parent, []).append(nb)

    notebooks: List[Path] = []
    prerequisites: Set[Path] = set()
    for folder, chosen in wanted.items():
        deps = read_dependencies(folder)
        needed: Set[str] = {nb.name for nb in chosen}
        for nb in chosen:
            earlier = deps[: deps.index(nb.name)] if nb.name in deps else deps
            needed.update(earlier)
        chosen_names = {nb.name for nb in chosen}
        for nb in folder_order(folder):
            if nb.name in needed:
                notebooks.append(nb)
                if nb.name not in chosen_names:
                    prerequisites.add(nb)
    return notebooks, prerequisites
