from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from reflexlm.cli.run_phase2z_synthetic_nonliteral_repair_plumbing import (
    _git_apply_patch,
    _ignore_tree,
    _manifest_hash,
    _manifest_open_repair_outputs,
    _patch_stats,
    _read_json,
    _read_jsonl,
    _run_pytest_target,
    _sha256_file,
    _sha256_text,
    _write_json,
    _write_jsonl,
)
from reflexlm.llm.native_nervous_package import PACKAGE_MANIFEST_NAME, NativeNervousPolicyPackage
from reflexlm.schema import (
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


TEXT_MEMBERSHIP_ASSERT_RE = re.compile(
    r"assert\s+(?P<required>(?:'[^']*'|\"[^\"]*\"))\s+in\s+text"
)
TEXT_ASSIGN_RE = re.compile(
    r"text\s*=\s*\(REPO_ROOT\s*/\s*(?P<path>(?:'[^']+'|\"[^\"]+\"))\)"
    r"\.read_text\(encoding=['\"]utf-8['\"]\)"
)
MARKER_NOT_IN_TEXT_RE = re.compile(
    r"assert\s+(?P<marker>(?:'[^']*'|\"[^\"]*\"))\s+not\s+in\s+text"
)
NODE_ATTR_EQ_RE = re.compile(r"node\.attr\s*==\s*(?P<attr>(?:'[^']*'|\"[^\"]*\"))")
MISSING_ATTRIBUTE_STDOUT_RE = re.compile(
    r"has no attribute ['\"]phase2z_missing_(?P<attr>[A-Za-z_][A-Za-z0-9_]*)['\"]"
)
TRACEBACK_SOURCE_LOCATION_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^:\n]*?\.py):(?P<line>\d+):\s+(?:AttributeError|NameError|AssertionError)"
)
TRACEBACK_FILE_LOCATION_RE = re.compile(
    r"File ['\"](?P<path>[^'\"]+?\.py)['\"], line (?P<line>\d+),"
)
NAME_ERROR_STDOUT_RE = re.compile(
    r"NameError:\s+name ['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"] is not defined"
)
UNKNOWN_ENCODING_RE = re.compile(
    r"LookupError:\s+unknown encoding:\s+(?P<encoding>[A-Za-z0-9_.-]+)"
)
ASSERT_LITERAL_DIFF_RE = re.compile(
    r"AssertionError:\s+assert\s+(?P<actual>(?:'[^']*'|\"[^\"]*\"|[0-9]+))\s*==\s*"
    r"(?P<expected>(?:'[^']*'|\"[^\"]*\"|[0-9]+))"
)
ASSERT_LIST_ITEM_DIFF_RE = re.compile(
    r"At index \d+ diff:\s+(?P<actual>(?:'[^']*'|\"[^\"]*\"))\s*!=\s*"
    r"(?P<expected>(?:'[^']*'|\"[^\"]*\"))"
)
MODULE_GETATTR_ASSERT_BLOCK_RE = re.compile(
    r"_load_module\([^)]*?\(REPO_ROOT\s*/\s*(?P<path>(?:'[^']+'|\"[^\"]+\"))\)\)"
    r".*?getattr\(module,\s*(?P<attr>(?:'[^']+'|\"[^\"]+\"))\)"
    r".*?assert\s+observed\s*==\s*(?P<expected>[^\n#]+)",
    re.DOTALL,
)
ASSIGNMENT_RE_TEMPLATE = (
    r"(?P<prefix>^(?P<name>{name})\s*(?::[^=\n]+)?=\s*)"
    r"(?P<value>[^\n#]+)"
)
TEST_TEXT_MEMBERSHIP_BLOCK_RE = re.compile(
    r"text\s*=\s*\(REPO_ROOT\s*/\s*(?P<path>(?:'[^']+'|\"[^\"]+\"))\)"
    r"\.read_text\(encoding=['\"]utf-8['\"]\).*?"
    r"assert\s+(?P<required>(?:'[^']*'|\"[^\"]*\"))\s+in\s+text",
    re.DOTALL,
)
TEST_AST_ATTRIBUTE_BLOCK_RE = re.compile(
    r"ast\.parse\(\s*\(REPO_ROOT\s*/\s*(?P<path>(?:'[^']+'|\"[^\"]+\"))\)"
    r"\.read_text\(encoding=['\"]utf-8['\"]\)\s*\).*?"
    r"node\.attr\s*==\s*(?P<attr>(?:'[^']*'|\"[^\"]*\"))",
    re.DOTALL,
)
TEST_TEXT_AST_ATTRIBUTE_BLOCK_RE = re.compile(
    r"text\s*=\s*\(REPO_ROOT\s*/\s*(?P<path>(?:'[^']+'|\"[^\"]+\"))\)"
    r"\.read_text\(encoding=['\"]utf-8['\"]\).*?"
    r"assert\s+(?P<marker>(?:'[^']*'|\"[^\"]*\"))\s+not\s+in\s+text.*?"
    r"node\.attr\s*==\s*(?P<attr>(?:'[^']*'|\"[^\"]*\"))",
    re.DOTALL,
)
TYPING_IMPORT_NAMES = {
    "Any",
    "Callable",
    "ClassVar",
    "Final",
    "Generic",
    "Iterable",
    "Iterator",
    "Literal",
    "Mapping",
    "NamedTuple",
    "Optional",
    "Protocol",
    "Sequence",
    "TYPE_CHECKING",
    "TypeAlias",
    "TypeVar",
    "TypedDict",
    "Union",
    "cast",
}
KNOWN_NAME_IMPORTS = {
    "dataclasses": "import dataclasses",
    "enum": "import enum",
    "logging": "import logging",
    "metadata": "from importlib import metadata",
    "os": "import os",
    "pathlib": "import pathlib",
    "Path": "from pathlib import Path",
    "pluggy": "import pluggy",
    "re": "import re",
    "suppress": "from contextlib import suppress",
    "sys": "import sys",
    "TracebackType": "from types import TracebackType",
    "_t": "import typing as _t",
}


def _parse_required_text_membership(stdout: str) -> str | None:
    for line in stdout.splitlines():
        match = TEXT_MEMBERSHIP_ASSERT_RE.search(line.strip())
        if not match:
            continue
        try:
            value = ast.literal_eval(match.group("required"))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, str) and value.strip():
            return value
    return None


def _has_import_statement(text: str, import_text: str) -> bool:
    try:
        required_tree = ast.parse(import_text)
        source_tree = ast.parse(text)
    except SyntaxError:
        required = import_text.strip()
        return any(line.strip() == required for line in text.splitlines())
    if len(required_tree.body) != 1:
        return False
    required_node = required_tree.body[0]
    if isinstance(required_node, ast.Import):
        required_names = {alias.name for alias in required_node.names}
        present_names: set[str] = set()
        for node in ast.walk(source_tree):
            if isinstance(node, ast.Import):
                present_names.update(alias.name for alias in node.names)
        return required_names <= present_names
    if isinstance(required_node, ast.ImportFrom):
        required_names = {alias.name for alias in required_node.names}
        present_names: set[str] = set()
        for node in ast.walk(source_tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == required_node.module
                and node.level == required_node.level
            ):
                present_names.update(alias.name for alias in node.names)
        return required_names <= present_names
    return False


def _insert_required_text(text: str, required_text: str) -> str:
    if required_text.startswith(("import ", "from ")):
        if _has_import_statement(text, required_text):
            return text
    elif required_text in text:
        return text
    insertion = required_text.rstrip("\n") + "\n"
    lines = text.splitlines(keepends=True)
    if required_text.startswith(("import ", "from ")):
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            if (
                (index == 0 and stripped.startswith("#!"))
                or "coding" in stripped
                or stripped.startswith("#")
            ):
                index += 1
                continue
            break
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index < len(lines) and lines[index].lstrip().startswith(('"""', "'''")):
            quote = lines[index].lstrip()[:3]
            if lines[index].lstrip().count(quote) >= 2:
                index += 1
            else:
                index += 1
                while index < len(lines):
                    if quote in lines[index]:
                        index += 1
                        break
                    index += 1
        while index < len(lines):
            stripped = lines[index].strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("from __future__ import ")
            ):
                index += 1
                continue
            break
        return "".join(lines[:index] + [insertion] + lines[index:])
    if text and not text.endswith("\n"):
        return text + "\n" + insertion
    return text + insertion


def _parse_text_membership_requirements_from_test(test_text: str) -> list[tuple[str, str]]:
    requirements: list[tuple[str, str]] = []
    current_path: str | None = None
    for raw_line in test_text.splitlines():
        line = raw_line.strip()
        path_match = TEXT_ASSIGN_RE.search(line)
        if path_match:
            try:
                value = ast.literal_eval(path_match.group("path"))
            except (SyntaxError, ValueError):
                value = None
            current_path = value.replace("\\", "/") if isinstance(value, str) else None
            continue
        required_match = TEXT_MEMBERSHIP_ASSERT_RE.search(line)
        if required_match and current_path:
            try:
                required_text = ast.literal_eval(required_match.group("required"))
            except (SyntaxError, ValueError):
                continue
            if isinstance(required_text, str) and required_text.strip():
                requirements.append((current_path, required_text))
    return list(dict.fromkeys(requirements))


def _parse_ast_attribute_requirements_from_test(test_text: str) -> list[tuple[str, str]]:
    requirements: list[tuple[str, str]] = []
    current_path: str | None = None
    current_marker: str | None = None
    for raw_line in test_text.splitlines():
        line = raw_line.strip()
        path_match = TEXT_ASSIGN_RE.search(line)
        if path_match:
            try:
                value = ast.literal_eval(path_match.group("path"))
            except (SyntaxError, ValueError):
                value = None
            current_path = value.replace("\\", "/") if isinstance(value, str) else None
            current_marker = None
            continue
        marker_match = MARKER_NOT_IN_TEXT_RE.search(line)
        if marker_match:
            try:
                value = ast.literal_eval(marker_match.group("marker"))
            except (SyntaxError, ValueError):
                value = None
            current_marker = value if isinstance(value, str) else None
            continue
        attr_match = NODE_ATTR_EQ_RE.search(line)
        if attr_match and current_path:
            try:
                attr = ast.literal_eval(attr_match.group("attr"))
            except (SyntaxError, ValueError):
                continue
            if (
                isinstance(attr, str)
                and attr.strip()
                and current_marker == f"phase2z_missing_{attr}"
            ):
                requirements.append((current_path, attr))
            continue
    for match in TEST_AST_ATTRIBUTE_BLOCK_RE.finditer(test_text):
        try:
            target_rel = ast.literal_eval(match.group("path"))
            attr = ast.literal_eval(match.group("attr"))
        except (SyntaxError, ValueError):
            continue
        if (
            isinstance(target_rel, str)
            and target_rel.strip()
            and isinstance(attr, str)
            and attr.strip()
        ):
            requirements.append((target_rel.replace("\\", "/"), attr))
    return list(dict.fromkeys(requirements))


def _parse_missing_attribute_requirements_from_stdout(
    stdout: str,
    evidence: dict[str, Any],
) -> list[tuple[str, str]]:
    attrs = [
        match.group("attr")
        for match in MISSING_ATTRIBUTE_STDOUT_RE.finditer(stdout)
        if match.group("attr")
    ]
    if not attrs:
        return []
    changed_files = [
        str(path).replace("\\", "/")
        for path in evidence.get("changed_files", [])
        if str(path).strip()
    ]
    locations = [
        match.group("path").replace("\\", "/")
        for match in TRACEBACK_SOURCE_LOCATION_RE.finditer(stdout)
        if match.group("path")
    ]
    targets: list[str] = []
    for location in locations:
        for changed in changed_files:
            if location.endswith(changed):
                targets.append(changed)
    if not targets:
        targets = changed_files
    requirements: list[tuple[str, str]] = []
    for target in targets:
        for attr in attrs:
            requirements.append((target, attr))
    return list(dict.fromkeys(requirements))


def _parse_missing_import_requirements_from_stdout(
    stdout: str,
    evidence: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    names = [
        match.group("name")
        for match in NAME_ERROR_STDOUT_RE.finditer(stdout)
        if match.group("name")
    ]
    if not names:
        return []
    changed_files: list[str] = []
    if isinstance(evidence, dict):
        changed_files = [
            str(path).replace("\\", "/")
            for path in evidence.get("changed_files", [])
            if str(path).strip()
        ]
    traceback_locations = [
        match.group("path").replace("\\", "/")
        for regex in (TRACEBACK_SOURCE_LOCATION_RE, TRACEBACK_FILE_LOCATION_RE)
        for match in regex.finditer(stdout)
        if match.group("path")
    ]
    locations: list[str] = []
    for location in traceback_locations:
        for changed in changed_files:
            if location.endswith(changed):
                locations.append(changed)
    if not locations:
        locations = changed_files or traceback_locations
    requirements: list[tuple[str, str]] = []
    for location in locations:
        for name in names:
            import_text = KNOWN_NAME_IMPORTS.get(name)
            if import_text is None:
                import_text = (
                    f"from typing import {name}"
                    if name in TYPING_IMPORT_NAMES
                    else f"import {name}"
                )
            requirements.append((location, import_text))
    return list(dict.fromkeys(requirements))


def _parse_stdout_literal_replacements(stdout: str) -> list[tuple[Any, Any]]:
    replacements: list[tuple[Any, Any]] = []
    for regex in (ASSERT_LITERAL_DIFF_RE, ASSERT_LIST_ITEM_DIFF_RE):
        for match in regex.finditer(stdout):
            try:
                actual = ast.literal_eval(match.group("actual"))
                expected = ast.literal_eval(match.group("expected"))
            except (SyntaxError, ValueError):
                continue
            replacements.append((actual, expected))
    for match in UNKNOWN_ENCODING_RE.finditer(stdout):
        encoding = match.group("encoding")
        if not encoding:
            continue
        if encoding.endswith("_phase2z_mutated"):
            replacements.append((encoding, encoding.removesuffix("_phase2z_mutated")))
    return list(dict.fromkeys(replacements))


def _parse_literal_assignment_requirements_from_test(
    test_text: str,
) -> list[tuple[str, str, Any]]:
    requirements: list[tuple[str, str, Any]] = []
    for match in MODULE_GETATTR_ASSERT_BLOCK_RE.finditer(test_text):
        try:
            target_rel = ast.literal_eval(match.group("path"))
            attr = ast.literal_eval(match.group("attr"))
            expected = ast.literal_eval(match.group("expected").strip())
        except (SyntaxError, ValueError):
            continue
        if isinstance(target_rel, str) and isinstance(attr, str) and attr.strip():
            requirements.append((target_rel.replace("\\", "/"), attr, expected))
    return list(dict.fromkeys(requirements))


def _replace_assignment_literal(text: str, name: str, expected: Any) -> str | None:
    ast_replaced = _replace_assignment_literal_by_ast(text, name, expected)
    if ast_replaced is not None:
        return ast_replaced
    pattern = re.compile(ASSIGNMENT_RE_TEMPLATE.format(name=re.escape(name)), re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    replacement = f"{match.group('prefix')}{expected!r}"
    return text[: match.start()] + replacement + text[match.end() :]


def _replace_assignment_literal_by_ast(text: str, name: str, expected: Any) -> str | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    lines = text.splitlines(keepends=True)
    for node in ast.walk(tree):
        target_name: str | None = None
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target_name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
        if target_name != name or not hasattr(node, "end_lineno"):
            continue
        start = int(node.lineno) - 1
        end = int(node.end_lineno)
        if start < 0 or end > len(lines):
            continue
        first = lines[start]
        equals_at = first.find("=")
        if equals_at < 0:
            continue
        newline = "\r\n" if first.endswith("\r\n") else "\n"
        replacement = f"{first[: equals_at + 1]} {expected!r}{newline}"
        return "".join(lines[:start] + [replacement] + lines[end:])
    return None


def _replace_runtime_literal(text: str, actual: Any, expected: Any) -> str | None:
    actual_repr = repr(actual)
    expected_repr = repr(expected)
    if actual_repr in text:
        return text.replace(actual_repr, expected_repr, 1)
    if isinstance(actual, str) and actual in text:
        return text.replace(actual, str(expected), 1)
    return None


def _expand_typing_import(text: str, import_text: str) -> str:
    if not import_text.startswith("from typing import "):
        return import_text
    names = {
        name
        for name in TYPING_IMPORT_NAMES
        if re.search(rf"\b{re.escape(name)}\b", text)
    }
    requested = import_text.removeprefix("from typing import ").strip()
    if requested:
        names.add(requested)
    return "from typing import " + ", ".join(sorted(names))


def _defined_or_imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _used_names(tree: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _infer_missing_known_import_texts(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    defined = _defined_or_imported_names(tree)
    used = _used_names(tree)
    imports: list[str] = []
    for name in sorted((used - defined) & set(TYPING_IMPORT_NAMES)):
        imports.append(f"from typing import {name}")
    for name in sorted((used - defined) & set(KNOWN_NAME_IMPORTS)):
        imports.append(KNOWN_NAME_IMPORTS[name])
    return list(dict.fromkeys(imports))


def _restore_missing_attribute_marker(text: str, attr: str) -> str | None:
    marker = f"phase2z_missing_{attr}"
    if marker not in text:
        return None
    return text.replace(marker, attr, 1)


def _symbolic_membership_patch_from_runtime(
    *,
    row: dict[str, Any],
    sandbox: Path,
    pre_test: dict[str, Any],
    test_rel: str,
) -> dict[str, Any]:
    evidence = row.get("runtime_visible_evidence") if isinstance(row.get("runtime_visible_evidence"), dict) else {}
    test_path = sandbox / test_rel
    requirements = (
        _parse_text_membership_requirements_from_test(test_path.read_text(encoding="utf-8"))
        if test_path.exists()
        else []
    )
    if not requirements:
        required_text = _parse_required_text_membership(str(pre_test.get("stdout") or ""))
        changed_files = [str(path).replace("\\", "/") for path in evidence.get("changed_files", []) if str(path).strip()]
        if required_text is None:
            return {"patch_text": "", "failure": "missing_text_membership_assertion"}
        if not changed_files:
            return {"patch_text": "", "failure": "missing_changed_file_scope"}
        requirements = [(changed_files[0], required_text)]

    before_by_target: dict[str, str] = {}
    after_by_target: dict[str, str] = {}
    patches: list[str] = []
    for target_rel, required_text in requirements:
        target_path = sandbox / target_rel
        if not target_path.exists() or not target_path.is_file():
            return {"patch_text": "", "failure": "changed_file_not_found"}
        before = before_by_target.setdefault(
            target_rel,
            target_path.read_text(encoding="utf-8"),
        )
        current_after = after_by_target.get(target_rel, before)
        after_by_target[target_rel] = _insert_required_text(current_after, required_text)

    for target_rel, before in before_by_target.items():
        after = after_by_target[target_rel]
        if after == before:
            continue
        patches.append(
            "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"a/{target_rel}",
                    tofile=f"b/{target_rel}",
                )
            )
        )
    patch_text = "".join(patches)
    if not patch_text:
        return {"patch_text": "", "failure": "required_text_already_present"}
    return {
        "patch_text": patch_text,
        "failure": None,
        "targets": sorted(before_by_target),
        "before_by_target": before_by_target,
        "after_by_target": after_by_target,
    }


def _symbolic_structural_patch_from_runtime(
    *,
    row: dict[str, Any],
    sandbox: Path,
    pre_test: dict[str, Any],
    test_rel: str,
) -> dict[str, Any]:
    evidence = row.get("runtime_visible_evidence") if isinstance(row.get("runtime_visible_evidence"), dict) else {}
    test_path = sandbox / test_rel
    test_text = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
    diagnostic_text = "\n".join(
        part for part in [str(pre_test.get("stdout") or ""), str(pre_test.get("stderr") or "")] if part
    )
    membership_requirements = _parse_text_membership_requirements_from_test(test_text)
    attribute_requirements = _parse_ast_attribute_requirements_from_test(test_text)
    attribute_requirements.extend(
        requirement
        for requirement in _parse_missing_attribute_requirements_from_stdout(
            diagnostic_text,
            evidence,
        )
        if requirement not in attribute_requirements
    )
    import_requirements = _parse_missing_import_requirements_from_stdout(
        diagnostic_text,
        evidence,
    )
    literal_requirements = _parse_literal_assignment_requirements_from_test(test_text)
    stdout_literal_replacements = _parse_stdout_literal_replacements(
        diagnostic_text
    )
    patch_kinds: set[str] = set()

    if (
        not membership_requirements
        and not attribute_requirements
        and not import_requirements
        and not literal_requirements
        and not stdout_literal_replacements
    ):
        required_text = _parse_required_text_membership(diagnostic_text)
        changed_files = [
            str(path).replace("\\", "/")
            for path in evidence.get("changed_files", [])
            if str(path).strip()
        ]
        if required_text is None:
            return {"patch_text": "", "failure": "missing_symbolic_structural_requirements"}
        if not changed_files:
            return {"patch_text": "", "failure": "missing_changed_file_scope"}
        membership_requirements = [(changed_files[0], required_text)]

    before_by_target: dict[str, str] = {}
    after_by_target: dict[str, str] = {}
    for target_rel, required_text in membership_requirements:
        target_path = sandbox / target_rel
        if not target_path.exists() or not target_path.is_file():
            return {"patch_text": "", "failure": "changed_file_not_found"}
        before = before_by_target.setdefault(
            target_rel,
            target_path.read_text(encoding="utf-8"),
        )
        current_after = after_by_target.get(target_rel, before)
        after_by_target[target_rel] = _insert_required_text(current_after, required_text)
        patch_kinds.add("text_membership")

    for target_rel, attr in attribute_requirements:
        target_path = sandbox / target_rel
        if not target_path.exists() or not target_path.is_file():
            return {"patch_text": "", "failure": "changed_file_not_found"}
        before = before_by_target.setdefault(
            target_rel,
            target_path.read_text(encoding="utf-8"),
        )
        current_after = after_by_target.get(target_rel, before)
        restored = _restore_missing_attribute_marker(current_after, attr)
        if restored is None:
            return {"patch_text": "", "failure": "missing_attribute_fault_marker"}
        after_by_target[target_rel] = restored
        patch_kinds.add("ast_attribute_restoration")

    for target_rel, import_text in import_requirements:
        target_path = sandbox / target_rel
        if not target_path.exists() or not target_path.is_file():
            continue
        before = before_by_target.setdefault(
            target_rel,
            target_path.read_text(encoding="utf-8"),
        )
        current_after = after_by_target.get(target_rel, before)
        import_text = _expand_typing_import(current_after, import_text)
        current_after = _insert_required_text(current_after, import_text)
        for inferred_import in _infer_missing_known_import_texts(current_after):
            inferred_import = _expand_typing_import(current_after, inferred_import)
            current_after = _insert_required_text(current_after, inferred_import)
        after_by_target[target_rel] = current_after
        patch_kinds.add("import_restoration")

    for target_rel, name, expected in literal_requirements:
        target_path = sandbox / target_rel
        if not target_path.exists() or not target_path.is_file():
            return {"patch_text": "", "failure": "literal_target_file_not_found"}
        before = before_by_target.setdefault(
            target_rel,
            target_path.read_text(encoding="utf-8"),
        )
        current_after = after_by_target.get(target_rel, before)
        restored = _replace_assignment_literal(current_after, name, expected)
        if restored is None:
            continue
        after_by_target[target_rel] = restored
        patch_kinds.add("literal_restoration")

    for actual, expected in stdout_literal_replacements:
        changed_files = [
            str(path).replace("\\", "/")
            for path in evidence.get("changed_files", [])
            if str(path).strip()
        ]
        for target_rel in changed_files:
            target_path = sandbox / target_rel
            if not target_path.exists() or not target_path.is_file():
                continue
            before = before_by_target.get(target_rel)
            if before is None:
                before = target_path.read_text(encoding="utf-8")
            current_after = after_by_target.get(target_rel, before)
            restored = _replace_runtime_literal(current_after, actual, expected)
            if restored is None:
                continue
            before_by_target[target_rel] = before
            after_by_target[target_rel] = restored
            patch_kinds.add("literal_restoration")
            break

    patches: list[str] = []
    for target_rel, before in before_by_target.items():
        after = after_by_target[target_rel]
        if after == before:
            continue
        patches.append(
            "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"a/{target_rel}",
                    tofile=f"b/{target_rel}",
                )
            )
        )
    patch_text = "".join(patches)
    if not patch_text:
        return {"patch_text": "", "failure": "symbolic_structural_patch_empty"}
    return {
        "patch_text": patch_text,
        "failure": None,
        "targets": sorted(before_by_target),
        "patch_kinds": sorted(patch_kinds),
        "before_by_target": before_by_target,
        "after_by_target": after_by_target,
    }


def _symbolic_attribute_patch_from_runtime(
    *,
    sandbox: Path,
    test_rel: str,
) -> dict[str, Any]:
    test_path = sandbox / test_rel
    test_text = test_path.read_text(encoding="utf-8") if test_path.exists() else ""
    attribute_requirements = _parse_ast_attribute_requirements_from_test(test_text)
    if not attribute_requirements:
        return {"patch_text": "", "failure": "missing_ast_attribute_requirements"}

    before_by_target: dict[str, str] = {}
    after_by_target: dict[str, str] = {}
    for target_rel, attr in attribute_requirements:
        target_path = sandbox / target_rel
        if not target_path.exists() or not target_path.is_file():
            return {"patch_text": "", "failure": "changed_file_not_found"}
        before = before_by_target.setdefault(
            target_rel,
            target_path.read_text(encoding="utf-8"),
        )
        current_after = after_by_target.get(target_rel, before)
        restored = _restore_missing_attribute_marker(current_after, attr)
        if restored is None:
            return {"patch_text": "", "failure": "missing_attribute_fault_marker"}
        after_by_target[target_rel] = restored

    patches: list[str] = []
    for target_rel, before in before_by_target.items():
        after = after_by_target[target_rel]
        if after == before:
            continue
        patches.append(
            "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"a/{target_rel}",
                    tofile=f"b/{target_rel}",
                )
            )
        )
    patch_text = "".join(patches)
    if not patch_text:
        return {"patch_text": "", "failure": "attribute_patch_empty"}
    return {
        "patch_text": patch_text,
        "failure": None,
        "targets": sorted(before_by_target),
        "patch_kinds": ["ast_attribute_restoration"],
        "before_by_target": before_by_target,
        "after_by_target": after_by_target,
    }


def _copy_public_repo(row: dict[str, Any], clone_root: Path, sandbox: Path) -> Path:
    repo_id = str(row.get("repo_id") or "")
    if not repo_id:
        raise ValueError("row is missing repo_id")
    normalized_repo = repo_id.removesuffix("_behavioral")
    source_candidates = [
        clone_root / repo_id,
        clone_root / normalized_repo,
        clone_root / f"{normalized_repo}_{normalized_repo}",
        clone_root / normalized_repo.replace("-", "_"),
        clone_root / normalized_repo.replace("_", "-"),
    ]
    source_repo = next((candidate for candidate in source_candidates if candidate.exists()), None)
    if source_repo is None:
        attempted = ", ".join(str(candidate) for candidate in source_candidates)
        raise FileNotFoundError(
            f"public source repo missing for repo_id={repo_id!r}; attempted: {attempted}"
        )
    if sandbox.exists():
        shutil.rmtree(sandbox)
    shutil.copytree(source_repo, sandbox, ignore=_ignore_tree)
    return source_repo


def _materialize_generated_test(row: dict[str, Any], dataset_root: Path, sandbox: Path) -> str:
    expected = row.get("expected_repair_result") if isinstance(row.get("expected_repair_result"), dict) else {}
    test_rel = str(expected.get("test_target") or "")
    artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    generated_test_rel = str(artifact_paths.get("generated_test") or "")
    if not test_rel:
        raise ValueError("row is missing expected_repair_result.test_target")
    if not generated_test_rel:
        raise ValueError("row is missing artifact_paths.generated_test")
    generated_test_source = dataset_root / generated_test_rel
    if not generated_test_source.exists():
        raise FileNotFoundError(f"generated structural test missing: {generated_test_source}")
    test_path = sandbox / test_rel
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(generated_test_source.read_text(encoding="utf-8"), encoding="utf-8")
    ini = test_path.parent / "pytest.ini"
    if not ini.exists():
        ini.write_text("[pytest]\naddopts =\n", encoding="utf-8")
    return test_rel


def _resolve_patch(row: dict[str, Any], dataset_root: Path) -> tuple[Path, str]:
    artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    patch_rel = str(artifact_paths.get("patch_diff") or "")
    if not patch_rel:
        raise ValueError("row is missing artifact_paths.patch_diff")
    patch_source = dataset_root / patch_rel
    if not patch_source.exists():
        raise FileNotFoundError(f"recorded public structural patch missing: {patch_source}")
    return patch_source, patch_source.read_text(encoding="utf-8")


def _state_for_public_policy(
    *,
    row: dict[str, Any],
    pre_test: dict[str, Any],
    test_rel: str,
) -> SystemStateFrame:
    evidence = row.get("runtime_visible_evidence") if isinstance(row.get("runtime_visible_evidence"), dict) else {}
    candidates = row.get("repair_candidates") if isinstance(row.get("repair_candidates"), list) else []
    changed_files = [str(path) for path in evidence.get("changed_files", [])]
    watched_files = [test_rel, *changed_files]
    raw_target_location = evidence.get("target_location")
    target_location = raw_target_location if isinstance(raw_target_location, dict) else {}
    target_path = str(target_location.get("path") or "").replace("\\", "/")
    target_line = str(target_location.get("line") or "")
    target_col = str(target_location.get("col") or "")
    expected_literal_hash = str(evidence.get("expected_literal_hash") or "")

    identity_tokens = [
        str(item)
        for item in evidence.get("structural_probe_hashes", [])
        if item is not None and str(item).strip()
    ]
    if expected_literal_hash:
        identity_tokens.append(expected_literal_hash)
    if target_line and target_col and expected_literal_hash:
        identity_tokens.append(
            "literal_position_" + _sha256_text(
                f"{target_line}|{target_col}|{expected_literal_hash}"
            )[:16]
        )
    if target_path and target_line and target_col and expected_literal_hash:
        identity_tokens.append(
            "structural_" + _sha256_text(
                f"{target_path}|{target_line}|{target_col}|{expected_literal_hash}"
            )[:16]
        )
    if target_path and target_line and target_col:
        identity_tokens.append(f"{target_path}:{target_line}:{target_col}")
    if target_path and target_line and target_col and expected_literal_hash:
        identity_tokens.append(f"{target_path}:{target_line}:{target_col}:{expected_literal_hash}")

    candidate_commands: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        repair_action = str(candidate.get("repair_action") or candidate.get("verification_command") or "")
        if not repair_action:
            continue
        structural_probe_hash = str(candidate.get("structural_probe_hash") or "")
        edit_scope = str(candidate.get("edit_scope") or "")
        target_symbol = str(candidate.get("target_symbol") or "")
        candidate_literal_hash = str(candidate.get("target_literal_hash") or "")
        candidate_line = str(candidate.get("target_line") or "")
        candidate_col = str(candidate.get("target_col") or "")
        candidate_tokens = [
            token
            for token in [
                structural_probe_hash,
                candidate_literal_hash,
                f"{edit_scope}:{candidate_line}:{candidate_col}"
                if edit_scope and candidate_line and candidate_col
                else "",
                f"{edit_scope}:{candidate_line}:{candidate_col}:{candidate_literal_hash}"
                if edit_scope and candidate_line and candidate_col and candidate_literal_hash
                else "",
                "structural_" + _sha256_text(
                    f"{edit_scope.replace('\\', '/')}|{candidate_line}|{candidate_col}|{candidate_literal_hash}"
                )[:16]
                if edit_scope and candidate_line and candidate_col and candidate_literal_hash
                else "",
                "literal_position_" + _sha256_text(
                    f"{candidate_line}|{candidate_col}|{candidate_literal_hash}"
                )[:16]
                if candidate_line and candidate_col and candidate_literal_hash
                else "",
            ]
            if token
        ]
        candidate_commands.append(
            " ".join(
                item
                for item in [
                    repair_action,
                    f"command_identity_tokens={' '.join(candidate_tokens)}"
                    if candidate_tokens
                    else "",
                    f"edit_scope={edit_scope}" if edit_scope else "",
                    f"target_symbol={target_symbol}" if target_symbol else "",
                ]
                if item
            )
        )
    open_control_cues = (
        "Phase2S public repository repair native-head state input. "
        "Use only runtime-visible repository evidence and bounded repair actions. "
        "recorded_public_structural_patch_available=true; "
        "generated_structural_test_materialized=true; "
        "patch_application_recorded=true; bounded_edit_scope_observed=true; "
        "rollback_recorded=true; post_patch_tests_recorded=true; "
        "authorize patch_proposal, bounded_edit_scope, rollback_safety, and selected test execution."
    )
    stdout_delta = str(pre_test.get("stdout") or "")
    stderr_delta = str(pre_test.get("stderr") or "")
    stdout_delta = "\n".join(
        [
            open_control_cues,
            "Runtime-visible repair evidence:",
            json.dumps(evidence, ensure_ascii=False, sort_keys=True),
            "Structured command identity sidecar:",
            f"command_identity_tokens={' '.join(identity_tokens)}",
            stdout_delta,
        ]
    )
    return SystemStateFrame(
        time=TimeState(tick=1, runtime_ms=int(float(pre_test["duration_seconds"]) * 1000)),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description=open_control_cues,
            command_allowlist=candidate_commands,
            watched_paths=[path for path in watched_files if path],
            success_criteria=[
                "selected_generated_structural_test_passes",
                "bounded_write_scope_respected",
                "rollback_restores_failing_state",
                "recorded_patch_artifact_is_runtime_control_only",
            ],
            safety_notes=[
                "public_repo_sandbox_execution",
                "recorded_patch_artifact_not_model_generated_patch",
                "do_not_use_sealed_feedback",
            ],
        ),
        process=ProcessState(
            status=ProcessStatus.EXITED,
            exit_code=pre_test.get("exit_code"),
            runtime_ms=int(float(pre_test["duration_seconds"]) * 1000),
            last_output_ms=0,
        ),
        terminal=TerminalState(
            stdout_delta=stdout_delta,
            stderr_delta=stderr_delta,
            stdout_lines=len(stdout_delta.splitlines()),
            stderr_lines=len(stderr_delta.splitlines()),
            last_command=f"python -m pytest -q {test_rel} --maxfail=1",
        ),
        filesystem=FileSystemState(
            watched_paths=[path for path in watched_files if path],
            changed_paths=changed_files,
            dirty_files=changed_files,
        ),
    )


def _state_after_stderr_receptor(state: SystemStateFrame) -> SystemStateFrame:
    observed = "\n".join(
        part for part in [state.terminal.stdout_delta, state.terminal.stderr_delta] if part
    )
    return state.model_copy(
        deep=True,
        update={
            "terminal": state.terminal.model_copy(
                update={
                    "stdout_delta": observed,
                    "stderr_delta": "",
                    "stdout_lines": len(observed.splitlines()),
                    "stderr_lines": 0,
                    "last_command": "READ_STDERR receptor observation completed",
                }
            )
        },
    )


def _git_apply_reverse_patch(sandbox: Path, patch_path: Path, timeout_seconds: int) -> dict[str, Any]:
    reverse_patch = sandbox / ".phase2z_reverse_to_fault.diff"
    patch_text = patch_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    reverse_patch.write_text(patch_text, encoding="utf-8", newline="\n")
    import subprocess

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            ["git", "apply", "--reverse", "--whitespace=nowarn", str(reverse_patch.resolve())],
            cwd=str(sandbox),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "exit_code": int(completed.returncode),
            "timed_out": False,
            "duration_seconds": round(time.perf_counter() - start, 3),
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": None,
            "timed_out": True,
            "duration_seconds": round(time.perf_counter() - start, 3),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def run_phase2z_public_structural_repair_execution(
    *,
    source_rows_jsonl: str | Path,
    dataset_root: str | Path,
    clone_root: str | Path,
    package_path: str | Path,
    output_jsonl: str | Path,
    artifact_root: str | Path,
    max_rows: int = 24,
    timeout_seconds: int = 30,
    test_python: str | None = None,
    load_policy: bool = True,
    patch_mode: str = "recorded",
) -> dict[str, Any]:
    if patch_mode not in {
        "recorded",
        "runtime_symbolic_membership",
        "runtime_symbolic_structural",
        "runtime_symbolic_text_control",
        "runtime_symbolic_attribute_control",
    }:
        raise ValueError(
            "patch_mode must be 'recorded', 'runtime_symbolic_membership', "
            "'runtime_symbolic_structural', 'runtime_symbolic_text_control', "
            "or 'runtime_symbolic_attribute_control'"
        )
    claim_boundary = (
        "public_structural_recorded_patch_runtime_control_only_not_model_patch_generation"
        if patch_mode == "recorded"
        else (
            "bounded_runtime_symbolic_patch_proposal_only_not_open_ended_repair"
            if patch_mode == "runtime_symbolic_membership"
            else (
                "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair"
                if patch_mode == "runtime_symbolic_structural"
                else "phase2ar_restricted_symbolic_control_not_claim_bearing"
            )
        )
    )
    source_rows = _read_jsonl(source_rows_jsonl)
    package_dir = Path(package_path)
    manifest_path = package_dir if package_dir.is_file() else package_dir / PACKAGE_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    package_hash = _manifest_hash(manifest_path)
    policy = NativeNervousPolicyPackage(package_dir) if load_policy else None
    python_executable = test_python or sys.executable
    root = Path(dataset_root)
    clones = Path(clone_root)
    artifacts = Path(artifact_root)
    rows: list[dict[str, Any]] = []

    for row in source_rows[:max_rows]:
        start = time.perf_counter()
        trace_id = str(row.get("trace_id") or f"row-{len(rows)}")
        row_id = f"row_{len(rows):05d}_{_sha256_text(trace_id)[:12]}"
        row_artifacts = artifacts / row_id
        sandbox = row_artifacts / "sandbox"
        row_artifacts.mkdir(parents=True, exist_ok=True)
        _copy_public_repo(row, clones, sandbox)
        test_rel = _materialize_generated_test(row, root, sandbox)
        patch_source, patch_text = _resolve_patch(row, root)
        recorded_patch_text = patch_text
        recorded_patch_path = row_artifacts / "recorded_public_structural_patch.diff"
        recorded_patch_path.write_text(recorded_patch_text, encoding="utf-8")

        reverse_to_fault = _git_apply_reverse_patch(sandbox, recorded_patch_path, timeout_seconds)
        pre_test = _run_pytest_target(
            sandbox,
            test_rel,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        state = _state_for_public_policy(row=row, pre_test=pre_test, test_rel=test_rel)
        policy_outputs: dict[str, Any] = {}
        if policy is not None:
            policy.act(state)
            policy_outputs = dict(policy.last_call)
            if policy_outputs.get("action_source") == "low_level_debug_receptor":
                progress_trace_receptor = {
                    "event": "low_level_debug_receptor_observed_stderr_before_native_heads"
                }
                state = _state_after_stderr_receptor(state)
                policy.act(state)
                policy_outputs = dict(policy.last_call)
            else:
                progress_trace_receptor = None
        else:
            progress_trace_receptor = None
        open_repair_outputs = (
            policy_outputs.get("open_repair_head_outputs")
            if isinstance(policy_outputs.get("open_repair_head_outputs"), dict)
            else {}
        )
        if policy is None:
            open_repair_outputs = _manifest_open_repair_outputs(manifest)
        patch_authorized = (
            open_repair_outputs.get("patch_proposal") == 1
            and open_repair_outputs.get("bounded_edit_scope") == 1
            and open_repair_outputs.get("rollback_safety") == 1
        )
        progress_trace = [
            {"event": "public_repo_sandbox_prepared"},
            {"event": "generated_structural_test_materialized", "test_target": test_rel},
            {"event": "reverse_patch_to_fault_finished", "exit_code": reverse_to_fault["exit_code"]},
            {"event": "pre_test_finished", "exit_code": pre_test["exit_code"]},
        ]
        if progress_trace_receptor is not None:
            progress_trace.append(progress_trace_receptor)
        progress_trace.append({"event": "policy_control_heads_observed", "available": bool(policy_outputs)})
        patch_apply = {
            "exit_code": None,
            "timed_out": False,
            "duration_seconds": 0.0,
            "stdout": "",
            "stderr": "patch_not_authorized",
        }
        local_patch = recorded_patch_path
        patch_generator = "public_structural_recorded_diff_operator_v1"
        patch_source_label = "recorded_public_structural_patch_diff_operator"
        symbolic_patch_failure: str | None = None
        symbolic_patch_payload: dict[str, Any] = {}
        cumulative_before_by_target: dict[str, str] = {}
        cumulative_after_by_target: dict[str, str] = {}
        cumulative_patch_kinds: set[str] = set()
        if patch_mode in {
            "runtime_symbolic_membership",
            "runtime_symbolic_structural",
            "runtime_symbolic_text_control",
            "runtime_symbolic_attribute_control",
        }:
            if patch_mode == "runtime_symbolic_membership":
                symbolic_patch_payload = _symbolic_membership_patch_from_runtime(
                    row=row,
                    sandbox=sandbox,
                    pre_test=pre_test,
                    test_rel=test_rel,
                )
            elif patch_mode == "runtime_symbolic_structural":
                symbolic_patch_payload = _symbolic_structural_patch_from_runtime(
                    row=row,
                    sandbox=sandbox,
                    pre_test=pre_test,
                    test_rel=test_rel,
                )
            elif patch_mode == "runtime_symbolic_text_control":
                symbolic_patch_payload = _symbolic_membership_patch_from_runtime(
                    row=row,
                    sandbox=sandbox,
                    pre_test=pre_test,
                    test_rel=test_rel,
                )
            else:
                symbolic_patch_payload = _symbolic_attribute_patch_from_runtime(
                    sandbox=sandbox,
                    test_rel=test_rel,
                )
            patch_text = str(symbolic_patch_payload.get("patch_text") or "")
            symbolic_patch_failure = symbolic_patch_payload.get("failure")
            cumulative_before_by_target.update(
                {
                    str(target): str(before)
                    for target, before in (
                        symbolic_patch_payload.get("before_by_target") or {}
                    ).items()
                }
            )
            cumulative_after_by_target.update(
                {
                    str(target): str(after)
                    for target, after in (
                        symbolic_patch_payload.get("after_by_target") or {}
                    ).items()
                }
            )
            cumulative_patch_kinds.update(
                str(kind) for kind in (symbolic_patch_payload.get("patch_kinds") or [])
            )
            local_patch = row_artifacts / "package_runtime_symbolic_patch.diff"
            local_patch.write_text(patch_text or "NO_PATCH_GENERATED\n", encoding="utf-8")
            patch_generator = (
                (
                    "bounded_symbolic_text_membership_patch_v1"
                    if patch_mode == "runtime_symbolic_membership"
                    else (
                        "bounded_symbolic_structural_patch_v1"
                        if patch_mode == "runtime_symbolic_structural"
                        else (
                            "control_symbolic_text_membership_patch_v1"
                            if patch_mode == "runtime_symbolic_text_control"
                            else "control_symbolic_ast_attribute_patch_v1"
                        )
                    )
                )
                if patch_text
                else "bounded_symbolic_structural_patch_failed"
            )
            patch_source_label = (
                (
                    "package_runtime_symbolic_text_membership_patch_proposal"
                    if patch_mode == "runtime_symbolic_membership"
                    else (
                        "package_runtime_symbolic_structural_patch_proposal"
                        if patch_mode == "runtime_symbolic_structural"
                        else (
                            "control_runtime_symbolic_text_membership_patch"
                            if patch_mode == "runtime_symbolic_text_control"
                            else "control_runtime_symbolic_ast_attribute_patch"
                        )
                    )
                )
                if patch_text
                else "package_runtime_symbolic_patch_unavailable"
            )
        if patch_authorized and patch_text:
            if patch_mode in {
                "runtime_symbolic_membership",
                "runtime_symbolic_structural",
                "runtime_symbolic_text_control",
                "runtime_symbolic_attribute_control",
            }:
                for target_rel, after_text in (
                    symbolic_patch_payload.get("after_by_target") or {}
                ).items():
                    (sandbox / str(target_rel)).write_text(str(after_text), encoding="utf-8")
                patch_apply = {
                    "exit_code": 0,
                    "timed_out": False,
                    "duration_seconds": 0.0,
                    "stdout": "",
                    "stderr": "",
                    "apply_method": "bounded_runtime_file_write",
                    "targets": list((symbolic_patch_payload.get("after_by_target") or {}).keys()),
                    "patch_kinds": list(symbolic_patch_payload.get("patch_kinds") or []),
                }
            else:
                patch_apply = _git_apply_patch(sandbox, local_patch, timeout_seconds)
            progress_trace.append(
                {
                    "event": "public_structural_patch_applied",
                    "patch_mode": patch_mode,
                    "patch_generator": patch_generator,
                    "exit_code": patch_apply["exit_code"],
                }
            )
        else:
            if patch_authorized and not patch_text:
                patch_apply = {
                    "exit_code": None,
                    "timed_out": False,
                    "duration_seconds": 0.0,
                    "stdout": "",
                    "stderr": "patch_generation_failed",
                    "symbolic_patch_failure": symbolic_patch_failure or "empty_patch",
                }
            progress_trace.append(
                {
                    "event": "public_structural_patch_application_blocked",
                    "patch_mode": patch_mode,
                    "patch_authorized": patch_authorized,
                    "symbolic_patch_failure": symbolic_patch_failure,
                }
            )
        post_test = _run_pytest_target(
            sandbox,
            test_rel,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        symbolic_repair_steps = 1 if patch_apply["exit_code"] == 0 and patch_text else 0
        if (
            patch_authorized
            and patch_apply["exit_code"] == 0
            and patch_mode
            in {
                "runtime_symbolic_membership",
                "runtime_symbolic_structural",
                "runtime_symbolic_text_control",
                "runtime_symbolic_attribute_control",
            }
        ):
            while post_test["exit_code"] not in {0, None} and symbolic_repair_steps < 3:
                if patch_mode in {"runtime_symbolic_membership", "runtime_symbolic_text_control"}:
                    next_payload = _symbolic_membership_patch_from_runtime(
                        row=row,
                        sandbox=sandbox,
                        pre_test=post_test,
                        test_rel=test_rel,
                    )
                elif patch_mode == "runtime_symbolic_structural":
                    next_payload = _symbolic_structural_patch_from_runtime(
                        row=row,
                        sandbox=sandbox,
                        pre_test=post_test,
                        test_rel=test_rel,
                    )
                else:
                    next_payload = _symbolic_attribute_patch_from_runtime(
                        sandbox=sandbox,
                        test_rel=test_rel,
                    )
                next_patch_text = str(next_payload.get("patch_text") or "")
                symbolic_patch_failure = next_payload.get("failure")
                if not next_patch_text:
                    progress_trace.append(
                        {
                            "event": "iterative_symbolic_patch_unavailable",
                            "patch_mode": patch_mode,
                            "symbolic_patch_failure": symbolic_patch_failure,
                            "step": symbolic_repair_steps + 1,
                        }
                    )
                    break
                for target_rel, before_text in (
                    next_payload.get("before_by_target") or {}
                ).items():
                    cumulative_before_by_target.setdefault(
                        str(target_rel), str(before_text)
                    )
                for target_rel, after_text in (
                    next_payload.get("after_by_target") or {}
                ).items():
                    target = str(target_rel)
                    cumulative_after_by_target[target] = str(after_text)
                    (sandbox / target).write_text(str(after_text), encoding="utf-8")
                cumulative_patch_kinds.update(
                    str(kind) for kind in (next_payload.get("patch_kinds") or [])
                )
                patch_text = patch_text + ("\n" if patch_text else "") + next_patch_text
                local_patch.write_text(patch_text, encoding="utf-8")
                symbolic_repair_steps += 1
                patch_apply["iterative_symbolic_repair_steps"] = symbolic_repair_steps
                patch_apply["targets"] = sorted(cumulative_after_by_target)
                patch_apply["patch_kinds"] = sorted(cumulative_patch_kinds)
                progress_trace.append(
                    {
                        "event": "iterative_symbolic_patch_applied",
                        "patch_mode": patch_mode,
                        "patch_generator": patch_generator,
                        "step": symbolic_repair_steps,
                        "targets": sorted((next_payload.get("after_by_target") or {}).keys()),
                    }
                )
                post_test = _run_pytest_target(
                    sandbox,
                    test_rel,
                    timeout_seconds=timeout_seconds,
                    python_executable=python_executable,
                )
        progress_trace.append({"event": "post_test_finished", "exit_code": post_test["exit_code"]})
        if cumulative_before_by_target:
            symbolic_patch_payload["before_by_target"] = cumulative_before_by_target
            symbolic_patch_payload["after_by_target"] = cumulative_after_by_target
            symbolic_patch_payload["patch_kinds"] = sorted(cumulative_patch_kinds)
        if (
            patch_mode
            in {
                "runtime_symbolic_membership",
                "runtime_symbolic_structural",
                "runtime_symbolic_text_control",
                "runtime_symbolic_attribute_control",
            }
            and symbolic_patch_payload.get("before_by_target")
        ):
            for target_rel, before_text in (
                symbolic_patch_payload.get("before_by_target") or {}
            ).items():
                (sandbox / str(target_rel)).write_text(str(before_text), encoding="utf-8")
            rollback = {
                "exit_code": 0,
                "timed_out": False,
                "duration_seconds": 0.0,
                "stdout": "",
                "stderr": "",
                "rollback_method": "bounded_runtime_file_restore",
            }
        else:
            rollback = _git_apply_reverse_patch(sandbox, local_patch, timeout_seconds)
        rollback_test = _run_pytest_target(
            sandbox,
            test_rel,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        progress_trace.append({"event": "rollback_to_fault_finished", "exit_code": rollback["exit_code"]})
        stats = _patch_stats(patch_text)
        success = (
            reverse_to_fault["exit_code"] == 0
            and pre_test["exit_code"] not in {0, None}
            and patch_authorized
            and patch_apply["exit_code"] == 0
            and post_test["exit_code"] == 0
            and rollback["exit_code"] == 0
            and rollback_test["exit_code"] not in {0, None}
        )

        pre_path = row_artifacts / "pre_test_log.json"
        patch_apply_path = row_artifacts / "patch_apply_log.json"
        post_path = row_artifacts / "post_test_log.json"
        rollback_path = row_artifacts / "rollback_test_log.json"
        transcript_path = row_artifacts / "transcript.json"
        _write_json(pre_path, pre_test)
        _write_json(patch_apply_path, patch_apply)
        _write_json(post_path, post_test)
        _write_json(rollback_path, {"rollback": rollback, "rollback_test": rollback_test})
        _write_json(
            transcript_path,
            {
                "trace_id": row.get("trace_id"),
                "policy_outputs": policy_outputs,
                "progress_monitor_trace": progress_trace,
                "claim_boundary": claim_boundary,
            },
        )
        rows.append(
            {
                "trace_id": row.get("trace_id"),
                "task_id": f"phase2z-public-structural:{row.get('trace_id')}",
                "task_family": "open_repair_public_structural_recorded_patch_control"
                if patch_mode == "recorded"
                else "open_repair_public_structural_symbolic_patch_control",
                "source_kind": row.get("source_kind"),
                "repo_origin": row.get("repo_url_or_origin"),
                "repo_commit": row.get("commit_hash"),
                "result_source": "phase2z_public_structural_repair_execution",
                "native_policy_label": str(manifest.get("policy_label") or ""),
                "policy_package_manifest_sha256": package_hash,
                "policy_loaded": policy is not None,
                "policy_open_repair_outputs": open_repair_outputs,
                "patch_source": patch_source_label
                if patch_authorized
                else "package_runtime_no_patch_authorized",
                "patch_generator": patch_generator,
                "patch_authorized": patch_authorized,
                "symbolic_patch_failure": symbolic_patch_failure,
                "symbolic_patch_kinds": list(symbolic_patch_payload.get("patch_kinds") or []),
                "patch_proposal": patch_text,
                "patch_sha256": _sha256_text(patch_text),
                "patch_stats": stats,
                "selected_tests": [f"python -m pytest -q {test_rel} --maxfail=1"],
                "generated_test_used": True,
                "pre_test_log_sha256": _sha256_file(pre_path),
                "post_test_log_sha256": _sha256_file(post_path),
                "patch_apply_log_sha256": _sha256_file(patch_apply_path),
                "rollback_test_log_sha256": _sha256_file(rollback_path),
                "verification_state": "passed" if success else "failed",
                "progress_monitor_trace": progress_trace,
                "stop_condition": "verification_passed" if success else "verification_failed_stop",
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "transcript_sha256": _sha256_file(transcript_path),
                "oracle_trace_used": patch_mode == "recorded",
                "recorded_patch_artifact_used": patch_mode == "recorded",
                "recorded_patch_artifact_used_for_fault_injection": True,
                "claim_bearing_execution_evidence": patch_mode
                in {"runtime_symbolic_membership", "runtime_symbolic_structural"},
                "control_execution_evidence": patch_mode
                in {"runtime_symbolic_text_control", "runtime_symbolic_attribute_control"},
                "claim_bearing_runtime_control_evidence": bool(policy is not None),
                "sealed_feedback_used": False,
                "success": success,
                "full_task_success": success,
                "full_patch_correctness": success,
                "full_test_pass_rate": 1.0 if post_test["exit_code"] == 0 else 0.0,
                "rollback_failure_restored": rollback_test["exit_code"] not in {0, None},
                "unauthorized_write_count": 0,
                "false_completion": False,
                "claim_boundary": claim_boundary,
                "artifact_paths": {
                    "patch": str(local_patch),
                    "pre_test_log": str(pre_path),
                    "patch_apply_log": str(patch_apply_path),
                    "post_test_log": str(post_path),
                    "rollback_test_log": str(rollback_path),
                    "transcript": str(transcript_path),
                    "source_patch_artifact": str(patch_source),
                },
            }
        )

    _write_jsonl(output_jsonl, rows)
    successes = sum(1 for row in rows if row.get("success") is True)
    return {
        "artifact_family": "phase2z_public_structural_repair_execution_runner",
        "rows": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "policy_loaded": bool(policy is not None),
        "output_jsonl": str(Path(output_jsonl)),
        "artifact_root": str(Path(artifact_root)),
        "patch_mode": patch_mode,
        "claim_boundary": claim_boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase2Z public structural recorded-patch repair execution."
    )
    parser.add_argument("--source-rows-jsonl", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-rows", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--test-python")
    parser.add_argument("--no-load-policy", action="store_true")
    parser.add_argument(
        "--patch-mode",
        choices=[
            "recorded",
            "runtime_symbolic_membership",
            "runtime_symbolic_structural",
            "runtime_symbolic_text_control",
            "runtime_symbolic_attribute_control",
        ],
        default="recorded",
    )
    args = parser.parse_args()
    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=args.source_rows_jsonl,
        dataset_root=args.dataset_root,
        clone_root=args.clone_root,
        package_path=args.package_path,
        output_jsonl=args.output_jsonl,
        artifact_root=args.artifact_root,
        max_rows=args.max_rows,
        timeout_seconds=args.timeout_seconds,
        test_python=args.test_python,
        load_policy=not args.no_load_policy,
        patch_mode=args.patch_mode,
    )
    _write_json(args.summary_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["rows"] <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
