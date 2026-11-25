"""Classify notebook errors: missing optional package (skipped) or real failure."""

import re
from pathlib import Path
from typing import Callable, Optional, Set, Tuple

_NO_MODULE = re.compile(r"""^No module named ['"]([^'"]+)['"]""")
OPTIONAL_PACKAGES_FILE = Path("tools") / "nbtest" / "optional_packages.txt"


def load_optional_packages(root: Path) -> Set[str]:
    """Top-level import names listed in tools/nbtest/optional_packages.txt (empty if the file is missing)."""
    path = Path(root) / OPTIONAL_PACKAGES_FILE
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.add(name)
    return names


def classify_error(
    ename: str, evalue: str, is_installed: Callable[[str], bool], optional: Set[str]
) -> Tuple[str, Optional[str]]:
    """Return ("skipped", top_level_module) or ("failed", None).

    A notebook is only "skipped" when it raised ModuleNotFoundError for a top-level module that is
    listed as optional AND is not installed at all in the kernel's Python. Anything else is a real
    failure: a module that is not on the optional list (a typo, a deleted local helper), a missing
    submodule of an installed package (e.g. cofi.xyz), any other ImportError, or a message that
    cannot be parsed.
    """
    if ename != "ModuleNotFoundError":
        return ("failed", None)
    match = _NO_MODULE.match((evalue or "").strip())
    if not match:
        return ("failed", None)
    top = match.group(1).split(".")[0]
    if not top or top not in optional or is_installed(top):
        return ("failed", None)
    return ("skipped", top)
