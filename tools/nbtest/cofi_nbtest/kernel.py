"""A private Jupyter kernel that always uses nbtest's own Python.

All notebooks ask for a kernel named "python3", which Jupyter resolves through the user's kernel
settings and which may point at any Python. nbtest instead registers a temporary kernelspec with
a unique name whose argv starts with sys.executable, and runs every notebook with it.
"""

import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator

# Determinism settings passed to the kernel (DESIGN.md). MPLBACKEND is deliberately NOT set:
# with MPLBACKEND=Agg the kernel produces no inline figures at all.
KERNEL_ENV: Dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


@contextmanager
def private_kernel() -> Iterator[str]:
    """Register a temporary kernelspec for this Python and yield its kernel name.

    The kernelspec lives in a temporary folder that is put first on JUPYTER_PATH for the duration
    of the run and removed afterwards. MPLBACKEND is removed from the environment the kernel
    inherits.
    """
    name = f"nbtest-{os.getpid()}"
    tmp = Path(tempfile.mkdtemp(prefix="nbtest-kernel-"))
    spec_dir = tmp / "kernels" / name
    spec_dir.mkdir(parents=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"nbtest ({sys.executable})",
        "language": "python",
        "env": KERNEL_ENV,
    }
    (spec_dir / "kernel.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")

    saved = {key: os.environ.get(key) for key in ("JUPYTER_PATH", "MPLBACKEND")}
    os.environ["JUPYTER_PATH"] = os.pathsep.join(p for p in (str(tmp), saved["JUPYTER_PATH"]) if p)
    os.environ.pop("MPLBACKEND", None)
    try:
        yield name
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(tmp, ignore_errors=True)


def own_python() -> Dict[str, str]:
    """The Python nbtest runs in, as the kernel must report it."""
    return {"executable": sys.executable, "prefix": sys.prefix}
