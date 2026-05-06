"""Validate CoFI example metadata for the InLab Explorer pipeline.

This script is intentionally read-only: it only reads ``meta.yml`` files and
CoFI method declarations, prints diagnostics, and exits non-zero on errors.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by missing environment deps
    print(
        "ERROR: PyYAML is required. Install it with `python -m pip install pyyaml`.",
        file=sys.stderr,
    )
    sys.exit(2)


REQUIRED_TOP_LEVEL_KEYS = ("title", "application domain", "description", "method")
ALLOWED_TOP_LEVEL_KEYS = set(REQUIRED_TOP_LEVEL_KEYS)
ALLOWED_METHOD_ENTRY_KEYS = {"description", "methods"}
SUPPORTED_EXAMPLE_SUFFIXES = (".ipynb", ".py")


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_mapping_without_duplicate_keys(
    loader: UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_without_duplicate_keys,
)


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    path: Path
    message: str
    line: int | None = None
    col: int | None = None


class Reporter:
    def __init__(self, root: Path, annotations: bool) -> None:
        self.root = root
        self.annotations = annotations
        self.diagnostics: list[Diagnostic] = []

    def error(
        self, path: Path, message: str, line: int | None = None, col: int | None = None
    ) -> None:
        self.diagnostics.append(Diagnostic("error", path, message, line, col))

    def warning(
        self, path: Path, message: str, line: int | None = None, col: int | None = None
    ) -> None:
        self.diagnostics.append(Diagnostic("warning", path, message, line, col))

    def emit(self) -> None:
        for diagnostic in self.diagnostics:
            print(self._format_human(diagnostic))
            if self.annotations:
                print(self._format_annotation(diagnostic))

        error_count = sum(1 for item in self.diagnostics if item.severity == "error")
        warning_count = sum(1 for item in self.diagnostics if item.severity == "warning")
        print(f"\nValidation finished with {error_count} error(s), {warning_count} warning(s).")

    def has_errors(self) -> bool:
        return any(item.severity == "error" for item in self.diagnostics)

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def _format_human(self, diagnostic: Diagnostic) -> str:
        location = self._relative_path(diagnostic.path)
        if diagnostic.line is not None:
            location += f":{diagnostic.line}"
            if diagnostic.col is not None:
                location += f":{diagnostic.col}"
        return f"{diagnostic.severity.upper()} {location}: {diagnostic.message}"

    def _format_annotation(self, diagnostic: Diagnostic) -> str:
        properties = [f"file={_escape_annotation_property(self._relative_path(diagnostic.path))}"]
        if diagnostic.line is not None:
            properties.append(f"line={diagnostic.line}")
        if diagnostic.col is not None:
            properties.append(f"col={diagnostic.col}")
        message = _escape_annotation_message(diagnostic.message)
        return f"::{diagnostic.severity} {','.join(properties)}::{message}"


def _escape_annotation_property(value: str) -> str:
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def _escape_annotation_message(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _normalize_tree_path(value: str) -> str:
    return " -> ".join(part.strip() for part in re.split(r"\s*->\s*", value.strip()))


def _find_top_level_key_line(lines: list[str], key: str) -> int | None:
    pattern = re.compile(rf"^{re.escape(key)}\s*:")
    for index, line in enumerate(lines, start=1):
        if pattern.match(line):
            return index
    return None


def _find_method_entry_lines(lines: list[str]) -> dict[str, int]:
    method_line = _find_top_level_key_line(lines, "method")
    if method_line is None:
        return {}

    entry_lines: dict[str, int] = {}
    for index, line in enumerate(lines[method_line:], start=method_line + 1):
        if line and not line.startswith((" ", "\t")):
            break
        match = re.match(r"^ {2}([^:#][^:]*):\s*(?:#.*)?$", line)
        if match:
            entry_lines.setdefault(match.group(1).strip(), index)
    return entry_lines


def _yaml_mark_location(error: yaml.YAMLError) -> tuple[int | None, int | None]:
    mark = getattr(error, "problem_mark", None) or getattr(error, "context_mark", None)
    if mark is None:
        return None, None
    return mark.line + 1, mark.column + 1


def _read_yaml(meta_path: Path, reporter: Reporter) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        text = meta_path.read_text(encoding="utf-8")
    except OSError as error:
        reporter.error(meta_path, f"cannot read file: {error}")
        return None, []

    lines = text.splitlines()
    try:
        data = yaml.load(text, Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        line, col = _yaml_mark_location(error)
        message = getattr(error, "problem", None) or str(error)
        reporter.error(meta_path, f"YAML syntax is invalid: {message}", line, col)
        return None, lines

    if data is None:
        reporter.error(meta_path, "file is empty")
        return None, lines
    if not isinstance(data, dict):
        reporter.error(meta_path, "top-level YAML value must be a mapping")
        return None, lines
    return data, lines


def _load_cofi_methods(cofi_tools: Path, reporter: Reporter) -> set[str]:
    if not cofi_tools.exists():
        reporter.error(cofi_tools, "CoFI tools path does not exist")
        return set()
    if not cofi_tools.is_dir():
        reporter.error(cofi_tools, "CoFI tools path must be a directory")
        return set()

    methods: set[str] = set()
    for python_file in sorted(cofi_tools.rglob("*.py")):
        try:
            lines = python_file.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            reporter.error(python_file, f"cannot read CoFI method file: {error}")
            continue
        for line in lines:
            match = re.match(r"^\s*#\s*(CoFI\s*->.*)$", line)
            if match:
                methods.add(_normalize_tree_path(match.group(1)))

    if not methods:
        reporter.error(cofi_tools, "no `# CoFI -> ...` method declarations found")
    return methods


def _iter_meta_files(root: Path) -> Iterable[Path]:
    for folder_name in ("examples", "tutorials"):
        folder = root / folder_name
        if not folder.is_dir():
            continue
        yield from sorted(folder.glob("*/meta.yml"))


def _validate_required_string(
    meta_path: Path,
    data: dict[str, Any],
    lines: list[str],
    key: str,
    reporter: Reporter,
) -> None:
    line = _find_top_level_key_line(lines, key)
    if key not in data:
        reporter.error(meta_path, f"missing required top-level key `{key}`", line)
        return
    if not isinstance(data[key], str) or not data[key].strip():
        reporter.error(meta_path, f"`{key}` must be a non-empty string", line)


def _validate_application_domain(
    meta_path: Path, data: dict[str, Any], lines: list[str], reporter: Reporter
) -> None:
    key = "application domain"
    line = _find_top_level_key_line(lines, key)
    value = data.get(key)
    if not isinstance(value, str):
        return

    parts = value.split(" -> ")
    normalized = _normalize_tree_path(value)
    if normalized != value.strip():
        reporter.warning(
            meta_path,
            "`application domain` has inconsistent whitespace around `->` separators",
            line,
        )
    if not parts or parts[0] != "CoFI Examples":
        reporter.error(meta_path, "`application domain` must start with `CoFI Examples`", line)


def _validate_method_entry(
    meta_path: Path,
    referenced_name: Any,
    entry: Any,
    line: int | None,
    declared_methods: set[str],
    reporter: Reporter,
) -> None:
    if not isinstance(referenced_name, str) or not referenced_name.strip():
        reporter.error(meta_path, "`method` entry names must be non-empty strings", line)
        return

    if not referenced_name.endswith(SUPPORTED_EXAMPLE_SUFFIXES):
        reporter.error(
            meta_path,
            f"`{referenced_name}` must reference a .ipynb or .py file",
            line,
        )

    referenced_file = meta_path.parent / referenced_name
    if not referenced_file.is_file():
        reporter.error(meta_path, f"referenced file does not exist: {referenced_name}", line)

    method_values = entry
    if isinstance(entry, dict):
        unknown_keys = sorted(set(entry) - ALLOWED_METHOD_ENTRY_KEYS)
        for unknown_key in unknown_keys:
            reporter.warning(
                meta_path,
                f"`{referenced_name}` has unknown method-entry key `{unknown_key}`",
                line,
            )

        description = entry.get("description")
        if not isinstance(description, str) or not description.strip():
            reporter.error(
                meta_path,
                f"`{referenced_name}` requires a non-empty `description` string",
                line,
            )
        if "methods" not in entry:
            reporter.error(meta_path, f"`{referenced_name}` is missing `methods`", line)
            return
        method_values = entry["methods"]
    elif not isinstance(entry, list):
        reporter.error(
            meta_path,
            f"`{referenced_name}` must be a list or a mapping with `description` and `methods`",
            line,
        )
        return

    if not isinstance(method_values, list) or not method_values:
        reporter.error(meta_path, f"`{referenced_name}` methods must be a non-empty list", line)
        return

    for method_index, method_value in enumerate(method_values, start=1):
        if not isinstance(method_value, str) or not method_value.strip():
            reporter.error(
                meta_path,
                f"`{referenced_name}` method #{method_index} must be a non-empty string",
                line,
            )
            continue

        normalized_method = _normalize_tree_path(method_value)
        if normalized_method != method_value.strip():
            reporter.warning(
                meta_path,
                f"`{referenced_name}` method #{method_index} has inconsistent whitespace around `->` separators",
                line,
            )
        if normalized_method not in declared_methods:
            reporter.error(
                meta_path,
                f"`{referenced_name}` method #{method_index} does not match a CoFI method declaration: {method_value}",
                line,
            )


def _validate_meta_file(
    meta_path: Path, declared_methods: set[str], reporter: Reporter
) -> None:
    data, lines = _read_yaml(meta_path, reporter)
    if data is None:
        return

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key == "method":
            continue
        _validate_required_string(meta_path, data, lines, key, reporter)
    _validate_application_domain(meta_path, data, lines, reporter)

    unknown_top_level_keys = sorted(set(data) - ALLOWED_TOP_LEVEL_KEYS)
    for key in unknown_top_level_keys:
        reporter.warning(
            meta_path,
            f"unknown top-level key `{key}` will be ignored by the explorer backend",
            _find_top_level_key_line(lines, str(key)),
        )

    method_line = _find_top_level_key_line(lines, "method")
    if "method" not in data:
        reporter.error(meta_path, "missing required top-level key `method`", method_line)
        return

    methods = data["method"]
    if not isinstance(methods, dict) or not methods:
        reporter.error(meta_path, "`method` must be a non-empty mapping", method_line)
        return

    method_entry_lines = _find_method_entry_lines(lines)
    for referenced_name, entry in methods.items():
        line = method_entry_lines.get(str(referenced_name), method_line)
        _validate_method_entry(
            meta_path,
            referenced_name,
            entry,
            line,
            declared_methods,
            reporter,
        )


def _default_cofi_tools(root: Path) -> Path | None:
    for candidate in (
        root / "cofi" / "src" / "cofi" / "tools",
        root.parent / "cofi" / "src" / "cofi" / "tools",
    ):
        if candidate.is_dir():
            return candidate
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate examples/*/meta.yml and tutorials/*/meta.yml for InLab Explorer."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to the cofi-examples repository root.",
    )
    parser.add_argument(
        "--cofi-tools",
        type=Path,
        default=None,
        help="Path to cofi/src/cofi/tools for method taxonomy validation.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Disable GitHub Actions annotation output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = args.root.resolve()
    cofi_tools = args.cofi_tools.resolve() if args.cofi_tools else _default_cofi_tools(root)

    reporter = Reporter(root=root, annotations=not args.no_annotations)

    if cofi_tools is None:
        reporter.error(
            root,
            "CoFI tools path was not found; pass `--cofi-tools ./cofi/src/cofi/tools`",
        )
        reporter.emit()
        return 1

    declared_methods = _load_cofi_methods(cofi_tools, reporter)
    meta_files = list(_iter_meta_files(root))
    if not meta_files:
        reporter.error(root, "no `examples/*/meta.yml` or `tutorials/*/meta.yml` files found")
    for meta_path in meta_files:
        _validate_meta_file(meta_path, declared_methods, reporter)

    reporter.emit()
    return 1 if reporter.has_errors() else 0


if __name__ == "__main__":
    sys.exit(main())
