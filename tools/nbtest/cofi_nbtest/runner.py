"""Execute one notebook into its staged copy (<name>.nbtest.ipynb next to the original).

Each notebook runs with nbclient in a worker thread with its own event loop. That keeps nbclient
from installing its own SIGINT handler (it cannot outside the main thread), so Ctrl-C reaches the
main thread, which cancels the worker; cancellation shuts the kernel down. The per-notebook
wall-clock timeout works the same way. nbclient's own `timeout` is per cell.
"""

import ast
import asyncio
import atexit
import hashlib
import importlib.util
import stat
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Dict, Optional

import nbformat
from jupyter_core.utils import ensure_async
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

from cofi_nbtest import __version__
from cofi_nbtest.discover import STAGED_SUFFIX
from cofi_nbtest.fileio import atomic_write_text
from cofi_nbtest.kernel import own_python
from cofi_nbtest.outcome import classify_error, load_optional_packages

CANCEL_GRACE_S = 60  # how long to wait for the kernel to shut down after cancelling


class Interrupted(Exception):
    """The user pressed Ctrl-C; the current notebook was abandoned and no staged file written."""


class KernelMismatch(Exception):
    """The kernel does not run in nbtest's own Python."""


@dataclass
class Result:
    path: Path
    staged: Path
    status: str  # ok | failed | timeout | skipped
    duration_s: float
    cell: Optional[int] = None  # 1-based position of the failing cell in the notebook
    ename: Optional[str] = None
    evalue: Optional[str] = None
    missing: Optional[str] = None
    kernel_python: Dict[str, str] = field(default_factory=dict)
    kernel_log: Optional[str] = None  # tail of the kernel process's own output, only if it died


def staged_path(original: Path) -> Path:
    return original.with_name(original.name[: -len(".ipynb")] + STAGED_SUFFIX)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


class _Progress:
    """Shared between the worker and the main thread."""

    def __init__(self) -> None:
        self.cell_index: Optional[int] = None
        self.kernel_python: Dict[str, str] = {}


async def _kernel_identity(client: NotebookClient) -> Dict[str, str]:
    expressions = {
        "executable": "__import__('sys').executable",
        "prefix": "__import__('sys').prefix",
    }
    msg_id = await ensure_async(
        client.kc.execute("", silent=True, store_history=False, user_expressions=expressions)
    )
    reply = await client.async_wait_for_reply(msg_id)
    values = (reply or {}).get("content", {}).get("user_expressions", {})
    identity = {}
    for key in expressions:
        item = values.get(key, {})
        if item.get("status") != "ok":
            raise KernelMismatch(f"could not read the kernel's sys.{key}")
        identity[key] = ast.literal_eval(item["data"]["text/plain"])
    return identity


async def _execute(client: NotebookClient, progress: _Progress, kernel_log: IO[bytes]) -> None:
    """Mirror of NotebookClient.async_execute with a kernel identity check before the first cell.

    The kernel process's own stdout/stderr (not the notebook's outputs, which ipykernel captures)
    go to kernel_log; it is shown only if the kernel dies.
    """
    client.reset_execution_trackers()
    async with client.async_setup_kernel(stdout=kernel_log, stderr=kernel_log):
        info = await client.async_wait_for_reply(await ensure_async(client.kc.kernel_info()))
        if info is not None and "language_info" in info["content"]:
            client.nb.metadata["language_info"] = info["content"]["language_info"]
        progress.kernel_python = await _kernel_identity(client)
        expected = own_python()
        if progress.kernel_python != expected:
            raise KernelMismatch(f"kernel runs {progress.kernel_python}, expected {expected}")
        for index, cell in enumerate(client.nb.cells):
            progress.cell_index = index
            await client.async_execute_cell(cell, index, execution_count=client.code_cells_executed + 1)
        progress.cell_index = None
        client.set_widgets_metadata()


class _Worker(threading.Thread):
    def __init__(self, client: NotebookClient, progress: _Progress, kernel_log: IO[bytes]) -> None:
        super().__init__(daemon=True)
        self.client = client
        self.progress = progress
        self.kernel_log = kernel_log
        self.error: Optional[BaseException] = None
        self.ready = threading.Event()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.task: Optional[asyncio.Task] = None

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        try:
            self.task = loop.create_task(_execute(self.client, self.progress, self.kernel_log))
            self.ready.set()
            loop.run_until_complete(self.task)
        except BaseException as exc:  # includes CancelledError
            self.error = exc
        finally:
            self.ready.set()
            try:
                # nbclient may leave helper tasks (e.g. its kernel-alive poller) after a cancellation;
                # cancel and await them so the loop closes cleanly.
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    def cancel(self) -> None:
        if self.loop is not None and self.task is not None and not self.loop.is_closed():
            try:
                self.loop.call_soon_threadsafe(self.task.cancel)
            except RuntimeError:
                pass  # loop already closed


def run_notebook(original: Path, root: Path, kernel_name: str, timeout_s: int) -> Result:
    """Execute original and write its staged copy. Raises Interrupted on Ctrl-C."""
    original_bytes = original.read_bytes()
    nb = nbformat.reads(original_bytes.decode("utf-8"), as_version=4)
    # Start from a clean slate: cells after a failure must not keep the original's committed outputs.
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(
        nb,
        kernel_name=kernel_name,
        timeout=timeout_s,
        allow_errors=False,
        shutdown_kernel="immediate",
        resources={"metadata": {"path": str(original.parent)}},
    )
    progress = _Progress()
    with tempfile.TemporaryFile() as kernel_log:
        return _run(client, nb, progress, kernel_log, original, root, original_bytes, timeout_s)


# Known harmless line ipykernel 7.3 prints for every local kernel; not worth showing.
_KERNEL_LOG_NOISE = ("Kernel is running over TCP without encryption",)


def _kernel_log_tail(kernel_log: IO[bytes], lines: int = 20) -> Optional[str]:
    """Last lines of the kernel process's own output, without known noise; None if nothing is left."""
    kernel_log.flush()
    kernel_log.seek(0)
    text = kernel_log.read().decode("utf-8", errors="replace")
    useful = [line for line in text.strip().splitlines() if line.strip() and not any(n in line for n in _KERNEL_LOG_NOISE)]
    return "\n".join(useful[-lines:]) or None


def _kill_kernel(client: NotebookClient) -> None:
    """Make sure no kernel process outlives this notebook.

    Normally nbclient has already shut the kernel down (and set client.km to None). If a
    cancellation hit before nbclient's own cleanup was in place (e.g. during kernel start-up), kill
    the kernel process here instead of leaving it to an exit handler.
    """
    km = client.km
    if km is None:
        return
    # Use the Popen object rather than a bare pid: Popen.kill() does nothing once the process has
    # been reaped, so an unrelated process that reused the pid can never be hit.
    process = getattr(getattr(km, "provisioner", None), "process", None)
    if process is not None and process.poll() is None:
        process.kill()
        try:
            process.wait(5)
        except Exception:
            pass
    try:
        atexit.unregister(client._cleanup_kernel)
    except Exception:
        pass


def _stop_worker(worker: "_Worker", client: NotebookClient) -> bool:
    """Cancel the worker and wait for it; further Ctrl-C presses are ignored meanwhile.

    Returns True if the worker finished, False if it was still running after the grace period
    (the kernel process is then killed).
    """
    worker.cancel()
    deadline = time.monotonic() + CANCEL_GRACE_S
    while worker.is_alive() and time.monotonic() < deadline:
        try:
            worker.join(0.5)
        except KeyboardInterrupt:
            print("\n      (stopping the kernel, please wait)", flush=True)
    if worker.is_alive():
        _kill_kernel(client)
        worker.join(5)
    return not worker.is_alive()


def _run(
    client: NotebookClient,
    nb: Any,
    progress: _Progress,
    kernel_log: IO[bytes],
    original: Path,
    root: Path,
    original_bytes: bytes,
    timeout_s: int,
) -> Result:
    worker = _Worker(client, progress, kernel_log)
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    timed_out = False

    stopped = True
    worker.start()
    try:
        worker.ready.wait()
        deadline = t0 + timeout_s
        while worker.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                stopped = _stop_worker(worker, client)
                break
            worker.join(min(0.5, remaining))
    except KeyboardInterrupt:
        _stop_worker(worker, client)
        _kill_kernel(client)
        raise Interrupted() from None
    _kill_kernel(client)

    duration = time.monotonic() - t0
    staged = staged_path(original)
    result = Result(path=original, staged=staged, status="ok", duration_s=duration, kernel_python=progress.kernel_python)
    cell_number = progress.cell_index + 1 if progress.cell_index is not None else None
    error = worker.error

    if not stopped:
        # The worker may still be touching the notebook; do not write a staged copy from it.
        result.status = "failed"
        result.cell = cell_number
        result.ename = "KernelShutdown"
        result.evalue = f"the kernel did not stop within {CANCEL_GRACE_S} s after the timeout and was killed; no staged copy written"
        return result
    if timed_out or isinstance(error, CellTimeoutError):
        result.status = "timeout"
        result.cell = cell_number
        result.ename = "Timeout"
        result.evalue = f"notebook exceeded {timeout_s} s" if timed_out else str(error).strip().splitlines()[0]
    elif isinstance(error, CellExecutionError):
        status, missing = classify_error(error.ename, error.evalue, _is_installed, load_optional_packages(root))
        result.status = status
        result.missing = missing
        result.cell = cell_number
        result.ename = error.ename
        result.evalue = error.evalue
    elif isinstance(error, DeadKernelError):
        result.status = "failed"
        result.cell = cell_number
        result.ename = "DeadKernelError"
        result.evalue = "the kernel died"
        result.kernel_log = _kernel_log_tail(kernel_log)
    elif isinstance(error, asyncio.CancelledError):
        # Only nbtest cancels the worker (timeout or Ctrl-C, both handled above).
        result.status = "failed"
        result.cell = cell_number
        result.ename = "Cancelled"
        result.evalue = "execution was cancelled"
    elif error is not None:
        result.status = "failed"
        result.cell = cell_number
        result.ename = type(error).__name__
        result.evalue = str(error)

    _write_staged(nb, result, root, original_bytes, started)
    return result


def _write_staged(nb: Any, result: Result, root: Path, original_bytes: bytes, started: datetime) -> None:
    stamp = {
        "nbtest_version": __version__,
        "status": result.status,
        "started": started.isoformat(timespec="seconds"),
        "duration_s": round(result.duration_s, 1),
        "python": {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "version": sys.version.split()[0],
            "kernel_executable": result.kernel_python.get("executable"),
            "kernel_prefix": result.kernel_python.get("prefix"),
        },
        "original": {
            "path": result.path.relative_to(root).as_posix(),
            "sha256": _sha256(original_bytes),
        },
        "error": None
        if result.status == "ok"
        else {"cell": result.cell, "ename": result.ename, "evalue": result.evalue, "kernel_log": result.kernel_log},
        "missing": result.missing,
    }
    nb.metadata["nbtest"] = stamp
    # The hash of the staged file as written is recorded outside the file (state.json, slice 2).
    # The staged copy gets the same permissions as its original notebook.
    atomic_write_text(result.staged, nbformat.writes(nb) + "\n", mode=stat.S_IMODE(result.path.stat().st_mode))
