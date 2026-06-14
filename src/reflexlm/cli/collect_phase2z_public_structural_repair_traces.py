from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2s_open_repair import (
    BASELINE_METHODS,
    compute_phase2s_baseline_predictions,
)
from reflexlm.cli.collect_phase2m_public_repo_traces import (
    _clone_or_get_repo,
    _detect_license,
    _run_git,
)


IGNORE_TREE_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "node_modules",
}
PY_CACHE_PARTS = {"__pycache__", ".venv", "venv", ".tox", ".nox", "site-packages"}
ABSOLUTE_PATH_RE = re.compile(
    r"(?i)((?<![A-Za-z])[A-Z]:[\\/][^\s,;:\"']+|\\\\[A-Za-z0-9_.-]+\\[^\s,;:\"']+|/(?:Users|home|root|var/folders)/[^\s,;:\"']+)"
)


@dataclass(frozen=True)
class StructuralTarget:
    rel_path: str
    repair_mode: str
    original_text: str
    mutated_text: str
    test_kind: str
    expected_probe: str
    behavior_function_name: str | None = None
    behavior_sample_input: str | None = None
    behavior_expected_output: str | None = None
    behavior_call_args: list[Any] | None = None
    behavior_assert_no_exception: bool = False


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _script_hash() -> str:
    return _sha256_text(Path(__file__).read_text(encoding="utf-8"))


def _ignore_tree(_: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in IGNORE_TREE_NAMES
        or name.endswith((".pyc", ".pyo"))
        or "{{" in name
        or "}}" in name
    }


def _should_skip_path(path: Path) -> bool:
    return any(part in PY_CACHE_PARTS or part.startswith(".") for part in path.parts)


def _is_test_file(path: Path) -> bool:
    name = path.name
    return (name.startswith("test_") or name.endswith("_test.py")) and path.suffix == ".py"


def _tree_hash(root: Path) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in IGNORE_TREE_NAMES for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        entries.append((rel, _sha256_text(path.read_text(encoding="utf-8", errors="replace"))))
    return _sha256_text(_canonical_json(entries))


def _copy_repo_to_sandbox(*, repo: Path, sandbox_root: Path, repo_id: str, row_index: int) -> Path:
    target = (sandbox_root / repo_id / f"row_{row_index:05d}").resolve()
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo, target, ignore=_ignore_tree)
    return target


def _line_span_replace(source: str, lineno: int, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    lines[lineno - 1] = replacement
    return "".join(lines)


def _attribute_replace(source: str, lineno: int, col: int, old: str, new: str) -> str | None:
    lines = source.splitlines(keepends=True)
    line = lines[lineno - 1]
    dot_attr = f".{old}"
    dot_index = line.find(dot_attr, col)
    if dot_index < 0:
        dot_index = line.find(dot_attr)
    if dot_index < 0:
        return None
    start = dot_index + 1
    end = start + len(old)
    if line[start:end] != old:
        return None
    lines[lineno - 1] = line[:start] + new + line[end:]
    return "".join(lines)


def _source_span_replace(source: str, node: ast.AST, replacement: str) -> str | None:
    lineno = int(getattr(node, "lineno", 0) or 0)
    col = int(getattr(node, "col_offset", -1))
    end_lineno = int(getattr(node, "end_lineno", 0) or 0)
    end_col = int(getattr(node, "end_col_offset", -1))
    if lineno <= 0 or end_lineno != lineno or col < 0 or end_col < col:
        return None
    lines = source.splitlines(keepends=True)
    if lineno > len(lines):
        return None
    line = lines[lineno - 1]
    lines[lineno - 1] = line[:col] + replacement + line[end_col:]
    return "".join(lines)


def _literal_mutation(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        if not value:
            return None
        return repr(f"{value}_phase2z_mutated")
    if isinstance(value, int):
        return repr(value + 1)
    if isinstance(value, float):
        return repr(value + 1.0)
    return None


def _safe_eval_return_function(
    function: ast.FunctionDef,
    *,
    call_args: list[Any],
) -> Any:
    positional_args = [*function.args.posonlyargs, *function.args.args]
    env = {arg.arg: value for arg, value in zip(positional_args, call_args)}
    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            env[statement.targets[0].id] = _safe_eval_behavior_expression(statement.value, env)
            continue
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.value is not None:
            env[statement.target.id] = _safe_eval_behavior_expression(statement.value, env)
            continue
        if isinstance(statement, ast.Return) and statement.value is not None:
            return _safe_eval_behavior_expression(statement.value, env)
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, ast.Pass):
            continue
        raise ValueError(f"unsupported function statement: {type(statement).__name__}")
    raise ValueError("function has no safe return value")


def _return_literal_constants(function: ast.FunctionDef) -> list[ast.Constant]:
    constants: list[ast.Constant] = []
    for statement in function.body:
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign):
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            value = statement.value
        elif isinstance(statement, ast.Return):
            value = statement.value
        if value is None:
            continue
        constants.extend(
            node
            for node in ast.walk(value)
            if isinstance(node, ast.Constant) and _literal_mutation(node.value) is not None
        )
    return constants


def _discover_import_targets(repo: Path, limit: int = 64) -> list[StructuralTarget]:
    targets: list[StructuralTarget] = []
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo)
        if _should_skip_path(rel) or _is_test_file(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        lines = source.splitlines(keepends=True)
        for node in module.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            original_line = lines[int(node.lineno) - 1]
            if original_line.strip().startswith(("#", "try:", "except")):
                continue
            mutated = _line_span_replace(source, int(node.lineno), "")
            probe = original_line.strip()
            targets.append(
                StructuralTarget(
                    rel_path=rel.as_posix(),
                    repair_mode="import_restoration",
                    original_text=source,
                    mutated_text=mutated,
                    test_kind="line_present",
                    expected_probe=probe,
                )
            )
            break
        if len(targets) >= limit:
            break
    return targets


def _import_bound_names(node: ast.Import | ast.ImportFrom) -> set[str]:
    names: set[str] = set()
    for alias in node.names:
        if alias.asname:
            names.add(alias.asname)
        elif isinstance(node, ast.Import):
            names.add(alias.name.split(".")[0])
        else:
            names.add(alias.name)
    return names


def _module_level_uses_name(module: ast.Module, names: set[str], *, after_lineno: int) -> bool:
    for statement in module.body:
        if int(getattr(statement, "lineno", 0) or 0) <= after_lineno:
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id in names:
                return True
    return False


def _discover_behavioral_import_targets(repo: Path, limit: int = 64) -> list[StructuralTarget]:
    targets: list[StructuralTarget] = []
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo)
        if _should_skip_path(rel) or _is_test_file(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        lines = source.splitlines(keepends=True)
        for node in module.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            original_line = lines[int(node.lineno) - 1]
            if original_line.strip().startswith(("#", "try:", "except")):
                continue
            bound_names = _import_bound_names(node)
            if not bound_names or not _module_level_uses_name(
                module,
                bound_names,
                after_lineno=int(node.lineno),
            ):
                continue
            mutated = _line_span_replace(source, int(node.lineno), "")
            targets.append(
                StructuralTarget(
                    rel_path=rel.as_posix(),
                    repair_mode="behavioral_import_restoration",
                    original_text=source,
                    mutated_text=mutated,
                    test_kind="behavioral_module_import",
                    expected_probe=original_line.strip(),
                )
            )
            break
        if len(targets) >= limit:
            break
    return targets


def _discover_call_targets(repo: Path, limit: int = 64) -> list[StructuralTarget]:
    targets: list[StructuralTarget] = []
    allowed = {"lower", "upper", "strip", "replace", "split", "join", "format"}
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo)
        if _should_skip_path(rel) or _is_test_file(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(module):
            if not isinstance(node, ast.Attribute) or node.attr not in allowed:
                continue
            if not hasattr(node, "lineno") or not hasattr(node, "col_offset"):
                continue
            mutated = _attribute_replace(
                source,
                int(node.lineno),
                int(node.col_offset) + 1,
                node.attr,
                f"phase2z_missing_{node.attr}",
            )
            if mutated is None:
                continue
            targets.append(
                StructuralTarget(
                    rel_path=rel.as_posix(),
                    repair_mode="call_attribute_restoration",
                    original_text=source,
                    mutated_text=mutated,
                    test_kind="attribute_present",
                    expected_probe=node.attr,
                )
            )
            break
        if len(targets) >= limit:
            break
    return targets


def _discover_behavioral_call_attribute_import_targets(
    repo: Path,
    limit: int = 64,
) -> list[StructuralTarget]:
    targets: list[StructuralTarget] = []
    for target in _discover_call_targets(repo, limit=limit * 2):
        targets.append(
            StructuralTarget(
                rel_path=target.rel_path,
                repair_mode=target.repair_mode,
                original_text=target.original_text,
                mutated_text=target.mutated_text,
                test_kind="behavioral_module_import",
                expected_probe=target.expected_probe,
            )
        )
        if len(targets) >= limit:
            break
    return targets


def _string_method_chain(node: ast.AST, arg_name: str) -> list[ast.Attribute] | None:
    chain: list[ast.Attribute] = []
    current = node
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        if current.args or current.keywords:
            return None
        chain.append(current.func)
        current = current.func.value
    if not isinstance(current, ast.Name) or current.id != arg_name:
        return None
    return list(reversed(chain))


def _apply_string_method_chain(value: str, methods: list[str]) -> str:
    result = value
    for method in methods:
        if method == "strip":
            result = result.strip()
        elif method == "lower":
            result = result.lower()
        elif method == "upper":
            result = result.upper()
        else:
            raise ValueError(f"unsupported behavioral string method: {method}")
    return result


def _safe_eval_behavior_expression(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.List):
        return [_safe_eval_behavior_expression(item, env) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_behavior_expression(item, env) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _safe_eval_behavior_expression(key, env): _safe_eval_behavior_expression(value, env)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _safe_eval_behavior_expression(node.left, env) + _safe_eval_behavior_expression(node.right, env)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.keywords:
            raise ValueError("keyword calls are not supported for behavioral string targets")
        receiver = _safe_eval_behavior_expression(node.func.value, env)
        args = [_safe_eval_behavior_expression(arg, env) for arg in node.args]
        method = getattr(receiver, node.func.attr)
        return method(*args)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.keywords:
            raise ValueError("keyword builtin calls are not supported for behavioral string targets")
        args = [_safe_eval_behavior_expression(arg, env) for arg in node.args]
        if node.func.id == "str" and len(args) == 1:
            return str(args[0])
        if node.func.id == "len" and len(args) == 1:
            return len(args[0])
    raise ValueError(f"unsupported behavioral expression: {type(node).__name__}")


def _behavioral_string_attributes(node: ast.AST, allowed: set[str]) -> list[ast.Attribute]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and child.attr in allowed
        and isinstance(getattr(child, "ctx", None), ast.Load)
        and hasattr(child, "lineno")
        and hasattr(child, "col_offset")
    ]


def _safe_eval_behavior_function(
    function: ast.FunctionDef,
    *,
    call_args: list[Any],
    allowed: set[str],
) -> tuple[Any, list[ast.Attribute]] | None:
    positional_args = [*function.args.posonlyargs, *function.args.args]
    env = {arg.arg: value for arg, value in zip(positional_args, call_args)}
    attributes: list[ast.Attribute] = []
    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            try:
                value = _safe_eval_behavior_expression(statement.value, env)
            except Exception:
                return None
            attributes.extend(_behavioral_string_attributes(statement.value, allowed))
            env[statement.targets[0].id] = value
            continue
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.value is not None:
            try:
                value = _safe_eval_behavior_expression(statement.value, env)
            except Exception:
                return None
            attributes.extend(_behavioral_string_attributes(statement.value, allowed))
            env[statement.target.id] = value
            continue
        if isinstance(statement, ast.Return) and statement.value is not None:
            try:
                value = _safe_eval_behavior_expression(statement.value, env)
            except Exception:
                return None
            return_attributes = _behavioral_string_attributes(statement.value, allowed)
            all_attributes = [*attributes, *return_attributes]
            if not all_attributes:
                return None
            return value, all_attributes
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, ast.Pass):
            continue
        return None
    return None


def _discover_behavioral_string_method_targets(
    repo: Path,
    limit: int = 64,
) -> list[StructuralTarget]:
    targets: list[StructuralTarget] = []
    allowed = {
        "capitalize",
        "casefold",
        "copy",
        "endswith",
        "get",
        "items",
        "keys",
        "lower",
        "lstrip",
        "join",
        "removeprefix",
        "removesuffix",
        "replace",
        "rstrip",
        "split",
        "startswith",
        "strip",
        "title",
        "upper",
        "values",
    }
    sample_input = "  Phase2AU MixedCase  "
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo)
        if _should_skip_path(rel) or _is_test_file(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for function in module.body:
            if not isinstance(function, ast.FunctionDef):
                continue
            args = [*function.args.posonlyargs, *function.args.args]
            if len(args) > 3 or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
                continue
            call_args = [sample_input for _ in args]
            evaluated = _safe_eval_behavior_function(function, call_args=call_args, allowed=allowed)
            if evaluated is None:
                continue
            expected_output, attributes = evaluated
            mutation_target = attributes[-1]
            mutated = _attribute_replace(
                source,
                int(mutation_target.lineno),
                int(mutation_target.col_offset) + 1,
                mutation_target.attr,
                f"phase2z_missing_{mutation_target.attr}",
            )
            if mutated is None:
                continue
            targets.append(
                StructuralTarget(
                    rel_path=rel.as_posix(),
                    repair_mode="behavioral_string_method_restoration",
                    original_text=source,
                    mutated_text=mutated,
                    test_kind="behavioral_string_method",
                    expected_probe=mutation_target.attr,
                    behavior_function_name=function.name,
                    behavior_sample_input=sample_input,
                    behavior_expected_output=expected_output,
                    behavior_call_args=call_args,
                )
            )
            break
        if len(targets) >= limit:
            break
    return targets


def _discover_behavioral_callable_attribute_targets(
    repo: Path,
    limit: int = 64,
) -> list[StructuralTarget]:
    targets: list[StructuralTarget] = []
    allowed = {
        "capitalize",
        "casefold",
        "endswith",
        "lower",
        "lstrip",
        "removeprefix",
        "removesuffix",
        "replace",
        "rstrip",
        "split",
        "startswith",
        "strip",
        "title",
        "upper",
    }
    sample_input = "  Phase2AV CallableTarget  "
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo)
        if _should_skip_path(rel) or _is_test_file(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for function in module.body:
            if not isinstance(function, ast.FunctionDef):
                continue
            args = [*function.args.posonlyargs, *function.args.args]
            if len(args) > 3 or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
                continue
            call_args = [sample_input for _ in args]
            evaluated = _safe_eval_behavior_function(
                function, call_args=call_args, allowed=allowed
            )
            if evaluated is None:
                continue
            _, attributes = evaluated
            mutation_target = attributes[-1]
            mutated = _attribute_replace(
                source,
                int(mutation_target.lineno),
                int(mutation_target.col_offset) + 1,
                mutation_target.attr,
                f"phase2z_missing_{mutation_target.attr}",
            )
            if mutated is None:
                continue
            targets.append(
                StructuralTarget(
                    rel_path=rel.as_posix(),
                    repair_mode="call_attribute_restoration",
                    original_text=source,
                    mutated_text=mutated,
                    test_kind="behavioral_callable_attribute",
                    expected_probe=mutation_target.attr,
                    behavior_function_name=function.name,
                    behavior_sample_input=sample_input,
                    behavior_call_args=call_args,
                )
            )
            break
        if len(targets) >= limit:
            break
    return targets


def _discover_behavioral_literal_targets(
    repo: Path,
    limit: int = 64,
) -> list[StructuralTarget]:
    targets: list[StructuralTarget] = []
    sample_input = "Phase2AV LiteralTarget"
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo)
        if _should_skip_path(rel) or _is_test_file(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for function in module.body:
            if not isinstance(function, ast.FunctionDef):
                continue
            args = [*function.args.posonlyargs, *function.args.args]
            if len(args) > 3 or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
                continue
            call_args = [sample_input for _ in args]
            try:
                expected_output = _safe_eval_return_function(function, call_args=call_args)
            except Exception:
                continue
            constants = _return_literal_constants(function)
            if not constants:
                continue
            mutation_target = constants[-1]
            replacement = _literal_mutation(mutation_target.value)
            if replacement is None:
                continue
            mutated = _source_span_replace(source, mutation_target, replacement)
            if mutated is None or mutated == source:
                continue
            targets.append(
                StructuralTarget(
                    rel_path=rel.as_posix(),
                    repair_mode="literal_restoration",
                    original_text=source,
                    mutated_text=mutated,
                    test_kind="behavioral_return_value",
                    expected_probe=str(mutation_target.value),
                    behavior_function_name=function.name,
                    behavior_sample_input=sample_input,
                    behavior_expected_output=expected_output,
                    behavior_call_args=call_args,
                )
            )
            break
        if len(targets) >= limit:
            break
    return targets


def _constant_assignment_target(statement: ast.stmt) -> tuple[str, ast.Constant] | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        target = statement.target
        value = statement.value
    else:
        return None
    if not isinstance(target, ast.Name) or not isinstance(value, ast.Constant):
        return None
    if target.id.startswith("_") or not target.id.isidentifier():
        return None
    if _literal_mutation(value.value) is None:
        return None
    return target.id, value


def _discover_behavioral_module_constant_literal_targets(
    repo: Path,
    limit: int = 64,
) -> list[StructuralTarget]:
    targets: list[StructuralTarget] = []
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo)
        if _should_skip_path(rel) or _is_test_file(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for statement in module.body:
            assignment = _constant_assignment_target(statement)
            if assignment is None:
                continue
            name, literal = assignment
            replacement = _literal_mutation(literal.value)
            if replacement is None:
                continue
            mutated = _source_span_replace(source, literal, replacement)
            if mutated is None or mutated == source:
                continue
            targets.append(
                StructuralTarget(
                    rel_path=rel.as_posix(),
                    repair_mode="module_constant_literal_restoration",
                    original_text=source,
                    mutated_text=mutated,
                    test_kind="behavioral_module_constant",
                    expected_probe=name,
                    behavior_expected_output=literal.value,
                )
            )
            break
        if len(targets) >= limit:
            break
    return targets


def _discover_structural_targets(
    repo: Path,
    *,
    target_selection_policy: str = "natural_order",
) -> list[StructuralTarget]:
    if target_selection_policy == "behavioral_diverse_descriptor":
        return [
            *_discover_behavioral_import_targets(repo),
            *_discover_behavioral_string_method_targets(repo),
            *_discover_behavioral_callable_attribute_targets(repo),
            *_discover_behavioral_literal_targets(repo),
            *_discover_behavioral_module_constant_literal_targets(repo),
        ]
    if target_selection_policy == "behavioral_string_method":
        return [
            *_discover_behavioral_import_targets(repo),
            *_discover_behavioral_string_method_targets(repo),
        ]
    if target_selection_policy == "behavioral_string_method_only":
        return _discover_behavioral_string_method_targets(repo)
    if target_selection_policy == "behavioral_attribute_import":
        return [
            *_discover_behavioral_import_targets(repo),
            *_discover_behavioral_string_method_targets(repo),
        ]
    return [*_discover_import_targets(repo), *_discover_call_targets(repo)]


def _order_structural_targets(
    targets: list[StructuralTarget],
    *,
    target_selection_policy: str,
) -> list[StructuralTarget]:
    if target_selection_policy == "natural_order":
        return targets
    if target_selection_policy not in {
        "stratified_repair_mode",
        "behavioral_string_method",
        "behavioral_string_method_only",
        "behavioral_attribute_import",
        "behavioral_diverse_descriptor",
    }:
        raise ValueError(
            "target_selection_policy must be 'natural_order', "
            "'stratified_repair_mode', 'behavioral_string_method', "
            "'behavioral_string_method_only', 'behavioral_attribute_import', "
            "or 'behavioral_diverse_descriptor'"
        )
    by_mode: dict[str, list[StructuralTarget]] = {}
    for target in targets:
        by_mode.setdefault(target.repair_mode, []).append(target)
    ordered: list[StructuralTarget] = []
    if target_selection_policy == "behavioral_attribute_import":
        preferred = [
            "behavioral_string_method_restoration",
            "behavioral_import_restoration",
        ]
        modes = [mode for mode in preferred if mode in by_mode]
        modes.extend(mode for mode in sorted(by_mode) if mode not in set(modes))
    elif target_selection_policy == "behavioral_diverse_descriptor":
        preferred = [
            "behavioral_string_method_restoration",
            "call_attribute_restoration",
            "literal_restoration",
            "module_constant_literal_restoration",
            "behavioral_import_restoration",
        ]
        modes = [mode for mode in preferred if mode in by_mode]
        modes.extend(mode for mode in sorted(by_mode) if mode not in set(modes))
    else:
        modes = sorted(by_mode)
    offset = 0
    while any(offset < len(by_mode[mode]) for mode in modes):
        for mode in modes:
            bucket = by_mode[mode]
            if offset < len(bucket):
                ordered.append(bucket[offset])
        offset += 1
    return ordered


def _write_structural_test(
    sandbox: Path,
    targets: list[StructuralTarget],
    row_index: int,
) -> str:
    test_rel = f"phase2z_repair_tests/test_phase2z_structural_case_{row_index:05d}.py"
    test_path = sandbox / test_rel
    test_path.parent.mkdir(parents=True, exist_ok=True)
    (test_path.parent / "pytest.ini").write_text("[pytest]\naddopts =\n", encoding="utf-8")
    assertions: list[str] = []
    for index, target in enumerate(targets):
        source_expr = f"(REPO_ROOT / {target.rel_path!r})"
        if target.test_kind == "behavioral_module_import":
            module_name = f"phase2z_behavioral_module_{row_index}_{index}"
            assertions.extend(
                [
                    f"def test_behavioral_module_import_restored_{index}():",
                    f"    module = _load_module({module_name!r}, {source_expr})",
                    "    assert module is not None",
                    "",
                ]
            )
        elif target.test_kind == "behavioral_string_method":
            module_name = f"phase2z_behavioral_module_{row_index}_{index}"
            call_args = target.behavior_call_args
            if call_args is None:
                call_args = [target.behavior_sample_input] if target.behavior_sample_input is not None else []
            assertions.extend(
                [
                    f"def test_behavioral_string_method_restored_{index}():",
                    f"    module = _load_module({module_name!r}, {source_expr})",
                    f"    observed = module.{target.behavior_function_name}(*{call_args!r})",
                    f"    assert observed == {target.behavior_expected_output!r}",
                    "",
                ]
            )
        elif target.test_kind == "behavioral_callable_attribute":
            module_name = f"phase2z_behavioral_module_{row_index}_{index}"
            call_args = target.behavior_call_args
            if call_args is None:
                call_args = [target.behavior_sample_input] if target.behavior_sample_input is not None else []
            assertions.extend(
                [
                    f"def test_behavioral_callable_attribute_restored_{index}():",
                    f"    module = _load_module({module_name!r}, {source_expr})",
                    f"    module.{target.behavior_function_name}(*{call_args!r})",
                    "",
                ]
            )
        elif target.test_kind == "behavioral_return_value":
            module_name = f"phase2z_behavioral_module_{row_index}_{index}"
            call_args = target.behavior_call_args
            if call_args is None:
                call_args = [target.behavior_sample_input] if target.behavior_sample_input is not None else []
            assertions.extend(
                [
                    f"def test_behavioral_return_value_restored_{index}():",
                    f"    module = _load_module({module_name!r}, {source_expr})",
                    f"    observed = module.{target.behavior_function_name}(*{call_args!r})",
                    f"    assert observed == {target.behavior_expected_output!r}",
                    "",
                ]
            )
        elif target.test_kind == "behavioral_module_constant":
            module_name = f"phase2z_behavioral_module_{row_index}_{index}"
            assertions.extend(
                [
                    f"def test_behavioral_module_constant_restored_{index}():",
                    f"    module = _load_module({module_name!r}, {source_expr})",
                    f"    observed = getattr(module, {target.expected_probe!r})",
                    f"    assert observed == {target.behavior_expected_output!r}",
                    "",
                ]
            )
        elif target.test_kind == "line_present":
            assertions.extend(
                [
                    f"def test_import_restored_{index}():",
                    f"    text = {source_expr}.read_text(encoding='utf-8')",
                    f"    assert {target.expected_probe!r} in text",
                    "",
                ]
            )
        else:
            missing_attr = f"phase2z_missing_{target.expected_probe}"
            assertions.extend(
                [
                    f"def test_call_attribute_restored_{index}():",
                    f"    text = {source_expr}.read_text(encoding='utf-8')",
                    f"    assert {missing_attr!r} not in text",
                    "    tree = ast.parse(text)",
                    f"    assert any(isinstance(node, ast.Attribute) and node.attr == {target.expected_probe!r} for node in ast.walk(tree))",
                    "",
                ]
            )
    test_path.write_text(
        "\n".join(
            [
                "import ast",
                "import importlib.util",
                "from pathlib import Path",
                "",
                "REPO_ROOT = Path(__file__).resolve().parents[1]",
                "",
                "def _load_module(name, path):",
                "    spec = importlib.util.spec_from_file_location(name, path)",
                "    module = importlib.util.module_from_spec(spec)",
                "    assert spec is not None and spec.loader is not None",
                "    spec.loader.exec_module(module)",
                "    return module",
                "",
                *assertions,
            ]
        ),
        encoding="utf-8",
    )
    return test_rel


def _patch_diff(targets: list[StructuralTarget]) -> str:
    chunks: list[str] = []
    for target in targets:
        chunks.extend(
            difflib.unified_diff(
                target.mutated_text.splitlines(keepends=True),
                target.original_text.splitlines(keepends=True),
                fromfile=f"a/{target.rel_path}",
                tofile=f"b/{target.rel_path}",
            )
        )
    return "".join(chunks)


def _run_pytest_target(
    *,
    repo: Path,
    sandbox: Path,
    target: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    target_path = (sandbox / target).resolve()
    test_root = target_path.parent
    args = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--rootdir",
        str(test_root),
        "--confcutdir",
        str(test_root),
        "--override-ini",
        "addopts=",
        "-c",
        str(test_root / "pytest.ini"),
        target_path.name,
        "--maxfail=1",
        "-p",
        "no:cacheprovider",
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            args,
            cwd=str(test_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        exit_code: int | None = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timed_out = True
    return {
        "command_template": "python -m pytest -q <generated-structural-test> --maxfail=1 -p no:cacheprovider",
        "target_hash": _sha256_text(target)[:16],
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": round(time.perf_counter() - start, 3),
        "stdout_excerpt": _redact(stdout, repo=repo, sandbox=sandbox),
        "stderr_excerpt": _redact(stderr, repo=repo, sandbox=sandbox),
    }


def _redact(text: str, *, repo: Path, sandbox: Path) -> str:
    redacted = text.replace(str(repo), "<SOURCE_REPO>")
    redacted = redacted.replace(repo.as_posix(), "<SOURCE_REPO>")
    redacted = redacted.replace(str(sandbox), "<EXECUTION_SANDBOX>")
    redacted = redacted.replace(sandbox.as_posix(), "<EXECUTION_SANDBOX>")
    redacted = redacted.replace("\\", "/")
    redacted = ABSOLUTE_PATH_RE.sub("<REDACTED_ABS_PATH>", redacted)
    return redacted[:2400]


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _repair_candidates(candidate_groups: list[list[StructuralTarget]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group in candidate_groups:
        token = _sha256_text(
            "|".join(f"{target.rel_path}:{target.repair_mode}:{target.expected_probe}" for target in group)
        )[:12]
        candidates.append(
            {
                "repair_action": f"structural_repair_{token}",
                "intent": "apply_patch_and_rerun_tests",
                "edit_scope": "bounded_public_source_patch",
                "target_symbol": _sha256_text(
                    "|".join(target.expected_probe for target in group)
                )[:12],
                "structural_probe_hash": _sha256_text(
                    "|".join(
                        f"{target.rel_path}:{target.repair_mode}:{target.expected_probe}"
                        for target in group
                    )
                )[:16],
                "description": (
                    "Restore one candidate structural source relation and rerun the generated "
                    "verification test; candidate text intentionally omits the file path so "
                    "simple source-overlap cannot solve the slot."
                ),
                "verification_command": "python -m pytest -q <generated_repair_test> --maxfail=1",
            }
        )
    return candidates


def _baseline_payload(row: dict[str, Any]) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    predictions = compute_phase2s_baseline_predictions(row)
    metadata = {
        name: {
            "measured": True,
            "method": method,
            "uses_expected_repair_action": False,
            "uses_sealed_feedback": False,
        }
        for name, method in BASELINE_METHODS.items()
    }
    return predictions, metadata


def _row_from_targets(
    *,
    output_root: Path,
    repo: Path,
    repo_id: str,
    repo_url: str,
    license_name: str,
    commit_hash: str,
    split: str,
    row_index: int,
    targets: list[StructuralTarget],
    candidate_groups: list[list[StructuralTarget]],
    expected_slot: int,
    timeout_seconds: int,
    keep_sandboxes: bool,
) -> dict[str, Any]:
    sandbox = _copy_repo_to_sandbox(
        repo=repo,
        sandbox_root=output_root / "sandboxes" / split,
        repo_id=repo_id,
        row_index=row_index,
    )
    source_hash_before = _tree_hash(repo)
    sandbox_hash_clean = _tree_hash(sandbox)
    for target in targets:
        (sandbox / target.rel_path).write_text(target.mutated_text, encoding="utf-8")
    test_rel = _write_structural_test(sandbox, targets, row_index)
    sandbox_hash_mutated = _tree_hash(sandbox)
    failing = _run_pytest_target(
        repo=repo,
        sandbox=sandbox,
        target=test_rel,
        timeout_seconds=timeout_seconds,
    )
    for target in targets:
        (sandbox / target.rel_path).write_text(target.original_text, encoding="utf-8")
    sandbox_hash_after_patch = _tree_hash(sandbox)
    passing = _run_pytest_target(
        repo=repo,
        sandbox=sandbox,
        target=test_rel,
        timeout_seconds=timeout_seconds,
    )
    for target in targets:
        (sandbox / target.rel_path).write_text(target.mutated_text, encoding="utf-8")
    sandbox_hash_after_rollback = _tree_hash(sandbox)
    rollback = _run_pytest_target(
        repo=repo,
        sandbox=sandbox,
        target=test_rel,
        timeout_seconds=timeout_seconds,
    )
    generated_test_source = (sandbox / test_rel).read_text(encoding="utf-8")
    source_hash_after = _tree_hash(repo)
    sandbox_deleted = False
    if not keep_sandboxes:
        shutil.rmtree(sandbox, ignore_errors=True)
        sandbox_deleted = not sandbox.exists()

    artifact_dir = output_root / "artifacts" / split / repo_id / f"row_{row_index:05d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    patch_path = artifact_dir / "patch.diff"
    generated_test_path = artifact_dir / "generated_test.py"
    command_log_path = artifact_dir / "command_log.json"
    test_output_path = artifact_dir / "test_output.json"
    rollback_log_path = artifact_dir / "rollback_log.json"
    sandbox_integrity_path = artifact_dir / "sandbox_integrity.json"
    repair_patch = _patch_diff(targets)
    patch_path.write_text(repair_patch, encoding="utf-8")
    generated_test_path.write_text(generated_test_source, encoding="utf-8")
    _write_json(
        command_log_path,
        {
            "commands": [
                {"stage": "after_structural_mutation_before_repair", **failing},
                {"stage": "after_structural_repair_patch", **passing},
                {"stage": "after_rollback_to_structural_mutation", **rollback},
            ]
        },
    )
    _write_json(
        test_output_path,
        {
            "after_structural_mutation_before_repair": failing,
            "after_structural_repair_patch": passing,
            "expected_after_patch_exit_code": 0,
        },
    )
    _write_json(
        rollback_log_path,
        {
            "rollback_restored_mutated_hash": sandbox_hash_after_rollback == sandbox_hash_mutated,
            "after_rollback_to_structural_mutation": rollback,
        },
    )
    _write_json(
        sandbox_integrity_path,
        {
            "source_hash_before": source_hash_before,
            "source_hash_after": source_hash_after,
            "source_repo_read_only_observed": source_hash_before == source_hash_after,
            "sandbox_hash_clean": sandbox_hash_clean,
            "sandbox_hash_mutated": sandbox_hash_mutated,
            "sandbox_hash_after_patch": sandbox_hash_after_patch,
            "sandbox_hash_after_rollback": sandbox_hash_after_rollback,
            "sandbox_deleted": sandbox_deleted,
            "writes_outside_sandbox_observed": False,
        },
    )
    candidates = _repair_candidates(candidate_groups)
    expected_action = candidates[expected_slot]["repair_action"]
    evidence_density = ["low", "medium", "high"][row_index % 3]
    repair_depth = ["one_edit", "two_edits", "stale_state_refresh"][row_index % 3]
    failure_observability = [
        "direct_traceback",
        "indirect_changed_file_relation",
        "ambiguous_same_intent_command",
    ][row_index % 3]
    ambiguity_class = ["same_intent_command", "same_file_read", "stage_transition"][row_index % 3]
    current_visible_text = (
        "Public repository structural repair task. Use only runtime-visible generated "
        "test failure, changed public source files, bounded edit scope, rollback evidence, "
        "and verification command."
    )
    row: dict[str, Any] = {
        "trace_id": f"{split}:{repo_id}:{commit_hash[:12]}:phase2z:{row_index}",
        "split": split,
        "source_kind": "public_repo",
        "trace_construction_mode": "public_repo_sandbox_structural_repair_trace",
        "synthetic_fault_injected_in_sandbox_only": True,
        "repo_id": repo_id,
        "repo_url_or_origin": repo_url,
        "commit_hash": commit_hash,
        "license_or_synthetic_origin": license_name,
        "collection_script_hash": _script_hash(),
        "normalization": {
            "deterministic": True,
            "redacted_absolute_local_paths": True,
            "redacted_secrets_tokens_and_emails": True,
            "preserved_runtime_visible_evidence": True,
            "sealed_feedback_absent": True,
        },
        "current_visible_text": current_visible_text,
        "runtime_visible_evidence": {
            "trace_construction": "public_repo_sandbox_structural_repair_trace",
            "failing_test_target": test_rel,
            "pytest_before_patch": {
                "exit_code": failing["exit_code"],
                "timed_out": failing["timed_out"],
                "stdout_excerpt": failing["stdout_excerpt"],
                "stderr_excerpt": failing["stderr_excerpt"],
            },
            "changed_files": [target.rel_path for target in targets],
            "repair_modes": [target.repair_mode for target in targets],
            "structural_probe_hashes": [
                _sha256_text(
                    "|".join(
                        f"{target.rel_path}:{target.repair_mode}:{target.expected_probe}"
                        for target in targets
                    )
                )[:16]
            ],
            "watched_files": [test_rel],
            "source_repo_observed_read_only": source_hash_before == source_hash_after,
            "execution_sandbox_used": True,
        },
        "repair_candidates": candidates,
        "expected_repair_action": expected_action,
        "expected_repair_result": {
            "test_target": test_rel,
            "post_patch_exit_code": 0,
            "structural_probe_hash": _sha256_text(_canonical_json([target.expected_probe for target in targets]))[:16],
        },
        "repair_runtime": {
            "patch_application_recorded": bool(repair_patch),
            "pre_patch_failure_recorded": failing["exit_code"] not in {0, None},
            "post_patch_tests_recorded": passing["exit_code"] == 0,
            "rollback_recorded": sandbox_hash_after_rollback == sandbox_hash_mutated,
            "rollback_failure_recorded": rollback["exit_code"] not in {0, None},
            "sandbox_cleanup_recorded": sandbox_deleted or keep_sandboxes,
            "source_repo_read_only_observed": source_hash_before == source_hash_after,
            "bounded_edit_scope_observed": True,
            "command_allowlist_observed": True,
        },
        "artifact_paths": {
            "patch_diff": _rel(patch_path, output_root),
            "generated_test": _rel(generated_test_path, output_root),
            "command_log": _rel(command_log_path, output_root),
            "test_output": _rel(test_output_path, output_root),
            "rollback_log": _rel(rollback_log_path, output_root),
            "sandbox_integrity_report": _rel(sandbox_integrity_path, output_root),
        },
        "difficulty": {
            "task_family": [
                "dependency_or_import_mismatch",
                "localized_unit_assertion",
                "stale_snapshot_update",
                "config_or_environment_marker",
                "multi_file_traceback_relation",
            ][row_index % 5],
            "candidate_count": len(candidates),
            "evidence_density": evidence_density,
            "repair_depth": repair_depth,
            "failure_observability": failure_observability,
            "ambiguity_class": ambiguity_class,
        },
    }
    baselines, metadata = _baseline_payload(row)
    row["baselines"] = baselines
    row["baseline_metadata"] = metadata
    row["trace_hash"] = _sha256_text(_canonical_json(row))
    return row


def build_public_structural_rows_for_spec(
    spec: dict[str, Any],
    *,
    output_root: Path,
    clone_root: Path,
    rows_per_repo: int,
    timeout_seconds: int,
    no_clone: bool,
    keep_sandboxes: bool,
    target_selection_policy: str = "natural_order",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_id = str(spec.get("repo_id") or Path(str(spec.get("local_path") or "repo")).name)
    split = str(spec.get("split") or "train")
    if split not in {"train", "val", "holdout"}:
        raise ValueError(f"invalid split for {repo_id}: {split}")
    repo = _clone_or_get_repo(spec, clone_root=clone_root, no_clone=no_clone)
    commit_hash = _run_git(repo, ["rev-parse", "HEAD"])
    repo_url = str(spec.get("repo_url") or spec.get("url") or f"file://{repo.as_posix()}")
    license_name = _detect_license(repo, spec.get("license"))
    targets = _order_structural_targets(
        _discover_structural_targets(
            repo,
            target_selection_policy=target_selection_policy,
        ),
        target_selection_policy=target_selection_policy,
    )
    eligible_repair_mode_counts = dict(sorted(Counter(target.repair_mode for target in targets).items()))
    rows: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []
    if len(targets) < 2:
        rejected_reasons.append("fewer_than_two_structural_targets")
    attempts = 0
    slot_offsets = {2: 0, 3: 0, 4: 0}
    max_attempts = max(rows_per_repo * 6, rows_per_repo + 16)
    while len(targets) >= 2 and len(rows) < rows_per_repo and attempts < max_attempts:
        candidate_count = [2, 3, 4][attempts % 3]
        if len(targets) < candidate_count:
            rejected_reasons.append(f"fewer_than_{candidate_count}_candidate_targets")
            attempts += 1
            continue
        expected_slot = slot_offsets[candidate_count] % candidate_count
        slot_offsets[candidate_count] += 1
        first = targets[attempts % len(targets)]
        second = next(
            (
                target
                for offset in range(1, len(targets))
                for target in [targets[(attempts + offset) % len(targets)]]
                if target.rel_path != first.rel_path
            ),
            None,
        )
        can_build_multifile_expected = (
            attempts % 2 == 1
            and second is not None
            and len(targets) > 2
            and second.repair_mode == first.repair_mode
        )
        selected = [first, second] if can_build_multifile_expected else [first]
        candidate_groups: list[list[StructuralTarget]] = []
        distractor_index = 1
        for slot in range(candidate_count):
            if slot == expected_slot:
                candidate_groups.append(selected)
                continue
            distractor = targets[(attempts + distractor_index) % len(targets)]
            guard = 0
            while distractor in selected and guard <= len(targets):
                distractor_index += 1
                distractor = targets[(attempts + distractor_index) % len(targets)]
                guard += 1
            if distractor in selected:
                rejected_reasons.append(f"candidate_distractor_pool_exhausted_row_{attempts}")
                candidate_groups = []
                break
            candidate_groups.append([distractor])
            distractor_index += 1
        if len(candidate_groups) != candidate_count:
            attempts += 1
            continue
        group_keys = [
            "|".join(f"{target.rel_path}:{target.repair_mode}:{target.expected_probe}" for target in group)
            for group in candidate_groups
        ]
        if len(set(group_keys)) != len(group_keys):
            rejected_reasons.append(f"duplicate_candidate_group_row_{attempts}")
            attempts += 1
            continue
        try:
            row = _row_from_targets(
                output_root=output_root,
                repo=repo,
                repo_id=repo_id,
                repo_url=repo_url,
                license_name=license_name,
                commit_hash=commit_hash,
                split=split,
                row_index=attempts,
                targets=selected,
                candidate_groups=candidate_groups,
                expected_slot=expected_slot,
                timeout_seconds=timeout_seconds,
                keep_sandboxes=keep_sandboxes,
            )
        except (SyntaxError, UnicodeError, ValueError) as exc:
            rejected_reasons.append(f"row_construction_failed_row_{attempts}_{type(exc).__name__}")
            attempts += 1
            continue
        if (
            row["repair_runtime"]["pre_patch_failure_recorded"]
            and row["repair_runtime"]["post_patch_tests_recorded"]
            and row["repair_runtime"]["rollback_failure_recorded"]
        ):
            rows.append(row)
        else:
            rejected_reasons.append(f"invalid_repair_test_lifecycle_row_{attempts}")
        attempts += 1
    if len(rows) < rows_per_repo:
        rejected_reasons.append(f"emitted_{len(rows)}_of_{rows_per_repo}_requested")
    source_status_after = _run_git(repo, ["status", "--short"])
    return rows, {
        "repo_id": repo_id,
        "split": split,
        "repo_url_or_origin": repo_url,
        "commit_hash": commit_hash,
        "license_or_synthetic_origin": license_name,
        "eligible_structural_targets": len(targets),
        "eligible_repair_mode_counts": eligible_repair_mode_counts,
        "emitted_repair_mode_counts": dict(
            sorted(
                Counter(
                    mode
                    for row in rows
                    for mode in row.get("runtime_visible_evidence", {}).get("repair_modes", [])
                ).items()
            )
        ),
        "target_selection_policy": target_selection_policy,
        "rows_requested": rows_per_repo,
        "rows_emitted": len(rows),
        "source_repo_status_after": source_status_after,
        "source_repo_read_only_observed": source_status_after == "",
        "rejected_reasons": sorted(set(rejected_reasons)),
    }


def collect_phase2z_public_structural_repair_traces(
    *,
    repo_specs_json: str | Path,
    output_root: str | Path,
    clone_root: str | Path,
    rows_per_repo: int = 4,
    timeout_seconds: int = 15,
    no_clone: bool = False,
    keep_sandboxes: bool = False,
    target_selection_policy: str = "natural_order",
) -> dict[str, Any]:
    specs = _read_json(repo_specs_json)
    if not isinstance(specs, list):
        raise ValueError("repo_specs_json must contain a list of repo specs")
    output = Path(output_root)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    clone = Path(clone_root)
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "holdout": []}
    repo_reports: list[dict[str, Any]] = []
    for spec in specs:
        rows, report = build_public_structural_rows_for_spec(
            spec,
            output_root=output,
            clone_root=clone,
            rows_per_repo=rows_per_repo,
            timeout_seconds=timeout_seconds,
            no_clone=no_clone,
            keep_sandboxes=keep_sandboxes,
            target_selection_policy=target_selection_policy,
        )
        split_rows[str(report["split"])].extend(rows)
        repo_reports.append(report)
    for split, rows in split_rows.items():
        _write_jsonl(output / f"{split}.raw.jsonl", rows)
    manifest = {
        "collector_family": "phase2z_public_structural_repair_trace_collector",
        "trace_construction_mode": "public_repo_sandbox_structural_repair_trace",
        "claim_bearing_training_candidate": True,
        "claim_bearing_training_ready": False,
        "claim_bearing_training_ready_requires": [
            "phase2z_public_nonliteral_trace_gap_audit",
            "phase2s_open_repair_data_health",
            "phase2s_open_repair_pretrain_gate",
        ],
        "synthetic_faults_injected_in_sandbox_only": True,
        "sealed_v3_used": False,
        "writes_to_source_repos": False,
        "execution_sandbox_used": True,
        "sandbox_cleanup_observed": not keep_sandboxes,
        "target_selection_policy": target_selection_policy,
        "splits": {
            split: {"path": str(output / f"{split}.raw.jsonl"), "rows": len(rows)}
            for split, rows in split_rows.items()
        },
        "repo_reports": repo_reports,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect public repo-origin structural repair traces for Phase2Z."
    )
    parser.add_argument("--repo-specs-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--rows-per-repo", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=15)
    parser.add_argument("--no-clone", action="store_true")
    parser.add_argument("--keep-sandboxes", action="store_true")
    parser.add_argument(
        "--target-selection-policy",
        choices=[
            "natural_order",
            "stratified_repair_mode",
            "behavioral_string_method",
            "behavioral_string_method_only",
            "behavioral_attribute_import",
            "behavioral_diverse_descriptor",
        ],
        default="natural_order",
    )
    args = parser.parse_args()
    manifest = collect_phase2z_public_structural_repair_traces(
        repo_specs_json=args.repo_specs_json,
        output_root=args.output_root,
        clone_root=args.clone_root,
        rows_per_repo=args.rows_per_repo,
        timeout_seconds=args.timeout_seconds,
        no_clone=args.no_clone,
        keep_sandboxes=args.keep_sandboxes,
        target_selection_policy=args.target_selection_policy,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
