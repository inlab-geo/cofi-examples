from collections import defaultdict
import nbformat
import os
import papermill as pm
from pathlib import Path
import re
import sys
import textwrap
import traceback
import shutil

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
EXAMPLES = "examples"
EXAMPLES_DIR = str(root_dir / EXAMPLES)
TUTORIALS = "tutorials"
TUTORIALS_DIR = str(root_dir / TUTORIALS)
FAILED_DIR = current_dir / "failed_notebooks"
# Regex for a typical Python warning line: "/some/path/file.py:123: UserWarning: message"
_WARN_LINE = re.compile(
    r"""
    (?P<path> /[^:]+ ) :          # absolute path up to colon
    (?P<lineno> \d+ ) :\s*        # line number
    (?P<wclass> \w*Warning ) :\s* # Warning class
    (?P<msg> .*?)\s*$             # the actual message
    """,
    re.VERBOSE,
)
# Regex for absolute paths (as fallback)
_PATH_RE = re.compile(r"(/[^/: ]+)+")
# For lines that start with just a filename
_FILE_AT_START_RE = re.compile(r"^\w+\.py\s*")

def save_failed_notebook(nb_path, status):
    """Copy failed notebooks for debugging"""
    if status != "FAILED":
        return
    #Preserve subdirectory structure
    nb_path= Path(nb_path)
    subdir = nb_path.parent.name
    dest_dir = FAILED_DIR / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = dest_dir / nb_path.name
    shutil.copy2(nb_path, dest_path)
    print(f" -> Saved to {dest_path.relative_to(current_dir)}")

def execute_notebook(input_path, output_path, cwd=None, params=None):
    """
    Executes a notebook and returns (status, message):
      - status: "PASSED", "WARNING", or "FAILED"
      - message: warnings (if any), else concise error, else empty string
    """
    try:
        pm.execute_notebook(input_path, output_path, cwd=cwd, parameters=params)
        warnings_in_nb = extract_warnings_from_notebook(output_path)
        if warnings_in_nb:
            msg = summarise_warnings(warnings_in_nb)
            return "WARNING", msg
        return "PASSED", ""
    except Exception as e:
        msg = summarise_exception_message(e)
        return "FAILED", msg


def extract_warnings_from_notebook(nb_path):
    """Extract warning-like messages from executed notebook outputs."""
    warnings_found = set()
    try:
        nb = nbformat.read(nb_path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code" and "outputs" in cell:
                for output in cell.outputs:
                    # Check output stream for 'Warning'
                    if output.output_type == "stream" and "Warning" in output.text:
                        for line in output.text.splitlines():
                            if "Warning" in line:
                                warnings_found.add(line.strip())
                    # Check error outputs
                    if output.output_type == "error":
                        for line in output.get("traceback", []):
                            if "Warning" in line:
                                warnings_found.add(line.strip())
    except Exception:
        pass
    return sorted(warnings_found)


def summarise_warnings(warnings_in_nb, *, maxlen: int = 150):
    """
    Return a compact string of unique warning messages (all noise removed).
    """
    cleaned = [_strip_noise(w) for w in warnings_in_nb]
    uniq = list(dict.fromkeys(cleaned))  # preserves order, removes duplicates
    msg = "; ".join(uniq)
    if len(msg) > maxlen:
        msg = msg[: maxlen - 3] + "..."
    return msg


def _strip_noise(msg: str) -> str:
    """Clean up a message, removing noise from warnings and paths."""
    # Try to match and extract message from a typical warning line
    m = _WARN_LINE.match(msg.strip())
    if m:
        return m.group("msg")
    # Fallback: remove :123: UserWarning: from the middle (rare)
    msg = re.sub(r":\d+:\s*\w*Warning:\s*", "", msg)
    # Replace absolute paths with basenames
    msg = _PATH_RE.sub(lambda m: Path(m.group(0)).name, msg)
    # Remove filename at start, e.g. "foo.py: message"
    msg = _FILE_AT_START_RE.sub("", msg)
    return " ".join(msg.split())


def _deepest_relevant(exc: BaseException) -> BaseException:
    """Follow __cause__/__context__ to the root exception."""
    while True:
        if exc.__cause__ is not None:
            exc = exc.__cause__
        elif exc.__context__ is not None and not exc.__suppress_context__:
            exc = exc.__context__
        else:
            return exc


def summarise_exception_message(exc: BaseException, *, maxlen: int = 150) -> str:
    """
    One-line summary of *exc*, truncated to *maxlen*.
    Works for ordinary exceptions, Papermill wrappers, SQLAlchemy wrappers, etc.
    """
    exc = _deepest_relevant(exc)

    # Generic “wrapper carries strings” case (Papermill, SQLAlchemy, etc)
    for a, b in (("ename", "evalue"), ("orig", "statement")):
        if hasattr(exc, a) and hasattr(exc, b):
            msg = f"{getattr(exc, a)}: {getattr(exc, b)}"
            break
    else:
        # Fall back to traceback’s own single-line rendering
        tb = traceback.TracebackException.from_exception(exc, capture_locals=False)
        msg = "".join(tb.format_exception_only()).strip()

    msg = _strip_noise(msg)
    return textwrap.shorten(msg, width=maxlen, placeholder="…")


def get_ordered_ipynbs_in_dir(dirpath):
    """
    Returns a list of .ipynb files in dirpath, with those listed in dependencies.txt first (if present).
    """
    result = []
    dependencies_path = os.path.join(dirpath, "dependencies.txt")
    dependencies = []
    if os.path.isfile(dependencies_path):
        with open(dependencies_path) as f:
            dependencies = [line.strip() for line in f if line.strip()]
        for dep in dependencies:
            dep_path = os.path.join(dirpath, dep)
            if dep.endswith(".ipynb") and os.path.isfile(dep_path):
                result.append(dep_path)
    already = set(dependencies)
    for f in os.listdir(dirpath):
        if f.endswith(".ipynb") and f not in already:
            f_path = os.path.join(dirpath, f)
            if os.path.isfile(f_path):
                result.append(f_path)
    return result


def find_ipynb_files(parent_dir):
    """
    Returns a list of .ipynb files exactly one directory below parent_dir,
    with dependencies.txt ordering applied within each subdirectory.
    """
    result = []
    parent_dir = os.path.abspath(parent_dir)
    base_depth = parent_dir.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(parent_dir):
        current_depth = dirpath.rstrip(os.sep).count(os.sep)
        if current_depth - base_depth == 1:
            result.extend(get_ordered_ipynbs_in_dir(dirpath))
        if current_depth - base_depth >= 1:
            dirnames[:] = []
    return result


def status_badge(status):
    """Return a shields.io Markdown badge for the given status."""
    if status == "PASSED":
        return "![PASSED](https://img.shields.io/badge/PASSED-brightgreen)"
    elif status == "WARNING":
        return "![WARNING](https://img.shields.io/badge/WARNING-orange)"
    elif status == "FAILED":
        return "![FAILED](https://img.shields.io/badge/FAILED-red)"
    else:
        return status


def write_markdown_report(report_path, rows):
    """
    Write the notebook execution summary as a Markdown table.
    """
    lines = []
    lines.append("# Notebook Execution Report\n")
    lines.append("| Notebook | Status | Message |")
    lines.append("|---|:---:|---|")
    lines.extend(rows)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    # Collect all examples & tutorials
    if sys.argv[-1] == "all":
        all_examples = find_ipynb_files(EXAMPLES_DIR)
        all_examples.extend(find_ipynb_files(TUTORIALS_DIR))
    else:
        all_examples = sys.argv[1:]
        all_examples = [file for file in all_examples if file.endswith(".ipynb")]

    print(all_examples)
    print("Executing examples...")

    # Dictionary: {dir_name: [(nb_name, badge, msg), ...]}
    report_rows_by_dir = defaultdict(list)
    if FAILED_DIR.exists():
        shutil.rmtree(FAILED_DIR)

    for nb in all_examples:
        print(f"file: {nb}")
        path = nb[: nb.rfind("/")]
        status, msg = execute_notebook(nb, nb, cwd=path)
        save_failed_notebook(nb, status)
        badge = status_badge(status)
        dir_name = os.path.basename(path)
        nb_name = os.path.basename(nb)
        report_rows_by_dir[dir_name].append((nb_name, badge, msg, status))

    report_path = os.path.join(current_dir, "notebook_execution_report.md")
    with open(report_path, "w") as f:
        f.write("# CoFI-Examples Execution Report\n\n")
        for dir_name in sorted(report_rows_by_dir):
            f.write(f"## {dir_name}\n\n")
            f.write("| Notebook | Status | Message |\n")
            f.write("|---|:---:|---|\n")
            for nb_name, badge, msg, status in report_rows_by_dir[dir_name]:
                if status == 'FAILED':
                    nb_link = f"[{nb_name}](failed_notebooks/{dir_name}/{nb_name})"
                else:
                    nb_link = nb_name
                f.write(f"| {nb_link} | {badge} | {msg} |\n")
            f.write("\n")
    print(f"\nExecution complete. Report written to {report_path}")
