from collections import defaultdict
import os
from pathlib import Path
import papermill as pm
import sys
import warnings

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
EXAMPLES = "examples"
EXAMPLES_DIR = str(root_dir / EXAMPLES)
TUTORIALS = "tutorials"
TUTORIALS_DIR = str(root_dir / TUTORIALS)


def execute_notebook(input_path, output_path, cwd=None, params=None):
    """
    Executes a notebook and returns (status, message):
      - status: "PASSED", "WARNING", or "FAILED"
      - message: warnings (if any), else concise error, else empty string
    """
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pm.execute_notebook(input_path, output_path, cwd=cwd, parameters=params)
            if w:
                warning_msgs = set(str(warn.message) for warn in w)
                sorted_msgs = sorted(warning_msgs)
                msg = "; ".join(sorted_msgs)
                # Truncate if overly long
                # if len(msg) > 50:
                #     msg = msg[:47] + "..."
                return "WARNING", msg
        return "PASSED", ""
    except Exception as e:
        msg = str(e).splitlines()[0]
        if len(msg) > 50:
            msg = msg[:47] + "..."
        return "FAILED", msg


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

    for nb in all_examples:
        print(f"file: {nb}")
        path = nb[: nb.rfind("/")]
        status, msg = execute_notebook(nb, nb, cwd=path)
        badge = status_badge(status)
        dir_name = os.path.basename(path)
        nb_name = os.path.basename(nb)
        report_rows_by_dir[dir_name].append((nb_name, badge, msg))

    report_path = os.path.join(current_dir, "notebook_execution_report.md")
    with open(report_path, "w") as f:
        f.write("# Notebook Execution Report\n\n")
        for dir_name in sorted(report_rows_by_dir):
            f.write(f"## {dir_name}\n\n")
            f.write("| Notebook | Status | Message |\n")
            f.write("|---|:---:|---|\n")
            for nb_name, badge, msg in report_rows_by_dir[dir_name]:
                f.write(f"| `{nb_name}` | {badge} | {msg} |\n")
            f.write("\n")
    print(f"\nExecution complete. Report written to {report_path}")
