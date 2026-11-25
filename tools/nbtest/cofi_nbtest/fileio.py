"""Safe file writing."""

import os
import stat
import tempfile
from pathlib import Path
from typing import Optional


def _default_mode() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def atomic_write_text(path: Path, text: str, mode: Optional[int] = None) -> None:
    """Write text to path atomically.

    The text goes to a temporary file in the same folder, which is then renamed over path. The
    temporary name starts with "." and ends in ".tmp", so notebook discovery (and the old
    run_notebooks.py) never picks it up. Permissions: `mode` if given, else those of the file it
    replaces, else the normal permissions for new files (0666 minus the umask) -- never mkstemp's
    private 0600. On any error the temporary file is removed and an existing file at path is left
    unchanged.
    """
    path = Path(path)
    if mode is None:
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            mode = _default_mode()
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
