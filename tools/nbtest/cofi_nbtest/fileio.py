"""Safe file writing."""

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Optional

_NO_PRECONDITION = object()


class AtomicWriteConflict(Exception):
    """The target changed after the caller decided it was safe to replace."""


def _default_mode() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _sha256_file(path: Path) -> Optional[str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return None
    return digest.hexdigest()


def atomic_write_text(
    path: Path,
    text: str,
    mode: Optional[int] = None,
    *,
    expected_sha256: Any = _NO_PRECONDITION,
) -> str:
    """Write text to path atomically.

    The text goes to a temporary file in the same folder, which is then renamed over path. The
    temporary name starts with "." and ends in ".tmp", so notebook discovery (and the old
    run_notebooks.py) never picks it up. Permissions: `mode` if given, else those of the file it
    replaces, else the normal permissions for new files (0666 minus the umask) -- never mkstemp's
    private 0600. On any error the temporary file is removed and an existing file at path is left
    unchanged. When ``expected_sha256`` is supplied, it is the expected hash of the existing
    target, or ``None`` when the target should not exist. The target is checked immediately before
    replacement and a changed target raises ``AtomicWriteConflict``. Returns the SHA-256 of the
    bytes written.
    """
    path = Path(path)
    if mode is None:
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            mode = _default_mode()
    data = text.encode("utf-8")
    written_sha256 = hashlib.sha256(data).hexdigest()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        if expected_sha256 is not _NO_PRECONDITION:
            actual_sha256 = _sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise AtomicWriteConflict(f"{path} changed while the notebook was running")
        os.replace(tmp, path)
        return written_sha256
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
