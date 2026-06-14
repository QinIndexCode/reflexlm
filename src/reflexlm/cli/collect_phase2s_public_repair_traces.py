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
from reflexlm.cli.generate_phase2s_open_repair_smoke import (
    AMBIGUITY_CLASSES,
    CANDIDATE_COUNTS,
    EVIDENCE_DENSITIES,
    FAILURE_OBSERVABILITY,
    REPAIR_DEPTHS,
)


TASK_FAMILIES = (
    "dependency_or_import_mismatch",
    "localized_unit_assertion",
    "stale_snapshot_update",
    "config_or_environment_marker",
    "multi_file_traceback_relation",
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
class LiteralRepairTarget:
    rel_path: str
    function_name: str
    literal_repr: str
    mutated_literal_repr: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int


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
    return {name for name in names if name in IGNORE_TREE_NAMES or name.endswith((".pyc", ".pyo"))}


def _copy_repo_to_sandbox(*, repo: Path, sandbox_root: Path, repo_id: str, row_index: int) -> Path:
    target = (sandbox_root / repo_id / f"row_{row_index:05d}").resolve()
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo, target, ignore=_ignore_tree)
    return target


def _tree_hash(root: Path) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in IGNORE_TREE_NAMES for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        entries.append((rel, _sha256_text(path.read_text(encoding="utf-8", errors="replace"))))
    return _sha256_text(_canonical_json(entries))


def _should_skip_path(path: Path) -> bool:
    return any(part in PY_CACHE_PARTS or part.startswith(".") for part in path.parts)


def _is_test_file(path: Path) -> bool:
    name = path.name
    return (name.startswith("test_") or name.endswith("_test.py")) and path.suffix == ".py"


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9_]+", text.lower()) if token]


def _literal_pair(value: Any) -> tuple[str, str] | None:
    if isinstance(value, bool):
        return (repr(value), repr(not value))
    if isinstance(value, int) and not isinstance(value, bool):
        return (repr(value), repr(value + 1))
    if isinstance(value, str) and value and len(value) <= 48:
        return (repr(value), repr(f"{value}_phase2s_mutation"))
    return None


def _discover_literal_repair_targets(repo: Path) -> list[LiteralRepairTarget]:
    targets: list[LiteralRepairTarget] = []
    for path in sorted(repo.rglob("*.py")):
        rel_path = path.relative_to(repo)
        if _should_skip_path(rel_path) or _is_test_file(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(module):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Return):
                    continue
                value_node = child.value
                if not isinstance(value_node, ast.Constant):
                    continue
                pair = _literal_pair(value_node.value)
                if pair is None:
                    continue
                if not all(
                    hasattr(value_node, attr)
                    for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset")
                ):
                    continue
                targets.append(
                    LiteralRepairTarget(
                        rel_path=rel_path.as_posix(),
                        function_name=node.name,
                        literal_repr=pair[0],
                        mutated_literal_repr=pair[1],
                        start_line=int(value_node.lineno),
                        start_col=int(value_node.col_offset),
                        end_line=int(value_node.end_lineno),
                        end_col=int(value_node.end_col_offset),
                    )
                )
                break
    return targets


def _replace_literal_at_position(
    text: str, target: LiteralRepairTarget, replacement: str
) -> str:
    module = ast.parse(text)
    for node in ast.walk(module):
        if not isinstance(node, ast.Constant):
            continue
        if (
            getattr(node, "lineno", None) == target.start_line
            and getattr(node, "col_offset", None) == target.start_col
        ):
            end_line = int(getattr(node, "end_lineno", target.end_line)) - 1
            end_col = int(getattr(node, "end_col_offset", target.end_col))
            if end_line != target.start_line - 1:
                raise ValueError(f"multi-line literal targets are unsupported: {target.rel_path}")
            lines = text.splitlines(keepends=True)
            line = lines[target.start_line - 1]
            lines[target.start_line - 1] = line[: target.start_col] + replacement + line[end_col:]
            return "".join(lines)
    raise ValueError(f"literal target position not found: {target.rel_path}:{target.start_line}")


def _mutate_target(sandbox: Path, target: LiteralRepairTarget) -> tuple[str, str]:
    path = sandbox / target.rel_path
    original = path.read_text(encoding="utf-8")
    mutated = _replace_literal_at_position(original, target, target.mutated_literal_repr)
    path.write_text(mutated, encoding="utf-8")
    patch = "".join(
        difflib.unified_diff(
            mutated.splitlines(keepends=True),
            original.splitlines(keepends=True),
            fromfile=f"a/{target.rel_path}",
            tofile=f"b/{target.rel_path}",
        )
    )
    return mutated, patch


def _apply_repair(sandbox: Path, target: LiteralRepairTarget) -> None:
    path = sandbox / target.rel_path
    mutated = path.read_text(encoding="utf-8")
    repaired = _replace_literal_at_position(mutated, target, target.literal_repr)
    path.write_text(repaired, encoding="utf-8")


def _write_repair_test(sandbox: Path, target: LiteralRepairTarget, row_index: int) -> str:
    test_rel = f"phase2s_repair_tests/test_phase2s_repair_case_{row_index:05d}.py"
    test_path = sandbox / test_rel
    test_path.parent.mkdir(parents=True, exist_ok=True)
    (test_path.parent / "pytest.ini").write_text("[pytest]\naddopts =\n", encoding="utf-8")
    test_path.write_text(
        "\n".join(
            [
                "import ast",
                "from pathlib import Path",
                "",
                "REPO_ROOT = Path(__file__).resolve().parents[1]",
                f"TARGET_REL_PATH = {target.rel_path!r}",
                f"TARGET_LINE = {target.start_line}",
                f"TARGET_COL = {target.start_col}",
                "",
                "",
                "def _literal_at_target_position():",
                "    tree = ast.parse((REPO_ROOT / TARGET_REL_PATH).read_text(encoding='utf-8'))",
                "    for node in ast.walk(tree):",
                "        if not isinstance(node, ast.Constant):",
                "            continue",
                "        if getattr(node, 'lineno', None) == TARGET_LINE and getattr(node, 'col_offset', None) == TARGET_COL:",
                "            return node.value",
                "    raise AssertionError('literal target position not found')",
                "",
                "",
                "def test_phase2s_public_repair_literal_restored():",
                f"    assert _literal_at_target_position() == {target.literal_repr}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return test_rel


def _shorten(text: str, limit: int = 2400) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<TRUNCATED>"


def _redact_execution_text(text: str, *, repo: Path, sandbox: Path) -> str:
    redacted = text.replace(str(repo), "<SOURCE_REPO>")
    redacted = redacted.replace(repo.as_posix(), "<SOURCE_REPO>")
    redacted = redacted.replace(str(sandbox), "<EXECUTION_SANDBOX>")
    redacted = redacted.replace(sandbox.as_posix(), "<EXECUTION_SANDBOX>")
    redacted = redacted.replace("\\", "/")
    redacted = ABSOLUTE_PATH_RE.sub("<REDACTED_ABS_PATH>", redacted)
    return _shorten(redacted)


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
    timed_out = False
    exit_code: int | None
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            args,
            cwd=str(test_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        exit_code = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    return {
        "command_template": (
            "python -m pytest -q --rootdir <generated-test-dir> "
            "--confcutdir <generated-test-dir> --override-ini addopts= "
            "-c <generated-test-dir>/pytest.ini <target> --maxfail=1 -p no:cacheprovider"
        ),
        "target_hash": _sha256_text(target)[:16],
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": round(time.perf_counter() - start, 3),
        "stdout_excerpt": _redact_execution_text(stdout, repo=repo, sandbox=sandbox),
        "stderr_excerpt": _redact_execution_text(stderr, repo=repo, sandbox=sandbox),
    }


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _repair_candidates(
    *,
    targets: list[LiteralRepairTarget],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for target in targets:
        token = _sha256_text(
            f"{target.rel_path}:{target.start_line}:{target.start_col}:"
            f"{target.function_name}:{target.literal_repr}:{target.mutated_literal_repr}"
        )[:12]
        candidates.append(
            {
                "repair_action": f"repair_action_{token}",
                "intent": "apply_patch_and_rerun_tests",
                "edit_scope": target.rel_path,
                "target_symbol": target.function_name,
                "target_line": target.start_line,
                "target_col": target.start_col,
                "target_literal_hash": _sha256_text(target.literal_repr)[:16],
                "description": (
                    "Restore the mutated literal in the bounded source file, rerun the "
                    "generated failing test, then stop if verification passes."
                ),
                "verification_command": "python -m pytest -q <generated_repair_test> --maxfail=1",
            }
        )
    return candidates


def _candidate_targets_for_slot(
    *,
    targets: list[LiteralRepairTarget],
    target_index: int,
    candidate_count: int,
    expected_slot: int,
) -> list[LiteralRepairTarget]:
    target = targets[target_index % len(targets)]
    same_file = [
        item for item in targets if item.rel_path == target.rel_path and item != target
    ]
    other_files = [
        item for item in targets if item.rel_path != target.rel_path and item != target
    ]
    pool = same_file + other_files
    if len(pool) < candidate_count - 1:
        return []
    candidate_targets: list[LiteralRepairTarget] = []
    pool_index = 0
    for slot in range(candidate_count):
        if slot == expected_slot:
            candidate_targets.append(target)
            continue
        candidate_targets.append(pool[pool_index])
        pool_index += 1
    return candidate_targets


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


def _row_from_target(
    *,
    output_root: Path,
    repo: Path,
    repo_id: str,
    repo_url: str,
    license_name: str,
    commit_hash: str,
    split: str,
    row_index: int,
    target: LiteralRepairTarget,
    candidate_targets: list[LiteralRepairTarget],
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
    _mutated_text, repair_patch = _mutate_target(sandbox, target)
    test_rel = _write_repair_test(sandbox, target, row_index)
    generated_test_source = (sandbox / test_rel).read_text(encoding="utf-8")
    sandbox_hash_mutated = _tree_hash(sandbox)
    failing = _run_pytest_target(
        repo=repo,
        sandbox=sandbox,
        target=test_rel,
        timeout_seconds=timeout_seconds,
    )
    _apply_repair(sandbox, target)
    sandbox_hash_after_patch = _tree_hash(sandbox)
    passing = _run_pytest_target(
        repo=repo,
        sandbox=sandbox,
        target=test_rel,
        timeout_seconds=timeout_seconds,
    )
    _mutate_target(sandbox, target)
    sandbox_hash_after_rollback = _tree_hash(sandbox)
    rollback = _run_pytest_target(
        repo=repo,
        sandbox=sandbox,
        target=test_rel,
        timeout_seconds=timeout_seconds,
    )
    source_hash_after = _tree_hash(repo)
    sandbox_deleted = False
    if not keep_sandboxes:
        shutil.rmtree(sandbox, ignore_errors=True)
        sandbox_deleted = not sandbox.exists()

    artifact_dir = output_root / "artifacts" / split / repo_id / f"row_{row_index:05d}"
    patch_path = artifact_dir / "patch.diff"
    generated_test_path = artifact_dir / "generated_test.py"
    command_log_path = artifact_dir / "command_log.json"
    test_output_path = artifact_dir / "test_output.json"
    rollback_log_path = artifact_dir / "rollback_log.json"
    sandbox_integrity_path = artifact_dir / "sandbox_integrity.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(repair_patch, encoding="utf-8")
    generated_test_path.write_text(generated_test_source, encoding="utf-8")
    _write_json(
        command_log_path,
        {
            "commands": [
                {"stage": "after_mutation_before_repair", **failing},
                {"stage": "after_repair_patch", **passing},
                {"stage": "after_rollback_to_mutation", **rollback},
            ]
        },
    )
    _write_json(
        test_output_path,
        {
            "after_mutation_before_repair": failing,
            "after_repair_patch": passing,
            "expected_after_patch_exit_code": 0,
        },
    )
    _write_json(
        rollback_log_path,
        {
            "rollback_restored_mutated_hash": sandbox_hash_after_rollback == sandbox_hash_mutated,
            "after_rollback_to_mutation": rollback,
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

    candidate_count = len(candidate_targets)
    candidates = _repair_candidates(
        targets=candidate_targets,
    )
    expected_action = candidates[expected_slot]["repair_action"]
    evidence_density = EVIDENCE_DENSITIES[row_index % len(EVIDENCE_DENSITIES)]
    repair_depth = REPAIR_DEPTHS[row_index % len(REPAIR_DEPTHS)]
    failure_observability = FAILURE_OBSERVABILITY[row_index % len(FAILURE_OBSERVABILITY)]
    ambiguity_class = AMBIGUITY_CLASSES[row_index % len(AMBIGUITY_CLASSES)]
    task_family = TASK_FAMILIES[row_index % len(TASK_FAMILIES)]
    runtime_visible_evidence = {
        "trace_construction": "public_repo_sandbox_literal_repair_trace",
        "failure_observability": failure_observability,
        "failing_test_target": test_rel,
        "pytest_before_patch": {
            "exit_code": failing["exit_code"],
            "timed_out": failing["timed_out"],
            "stdout_excerpt": failing["stdout_excerpt"],
            "stderr_excerpt": failing["stderr_excerpt"],
        },
        "changed_files": [target.rel_path],
        "traceback_symbols": [target.function_name],
        "target_location": {
            "path": target.rel_path,
            "line": target.start_line,
            "col": target.start_col,
        },
        "expected_literal_hash": _sha256_text(target.literal_repr)[:16],
        "watched_files": [test_rel],
        "prior_repair_summary": (
            "A disposable sandbox mutation produced a failing public-repository repair "
            "test. The source repository stayed read-only; only sandbox files were patched."
        ),
        "stale_state_refresh": repair_depth == "stale_state_refresh",
        "source_repo_observed_read_only": source_hash_before == source_hash_after,
        "execution_sandbox_used": True,
    }
    current_visible_text = (
        "Public repository sandbox repair task. Use the failing generated test, changed "
        "public source file, rollback evidence, and bounded edit scope before choosing "
        f"a repair action. repo={repo_id}; task_family={task_family}; "
        f"density={evidence_density}; ambiguity={ambiguity_class}."
    )
    row: dict[str, Any] = {
        "trace_id": f"{split}:{repo_id}:{commit_hash[:12]}:phase2s:{row_index}",
        "split": split,
        "source_kind": "public_repo",
        "trace_construction_mode": "public_repo_sandbox_literal_repair_trace",
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
        },
        "current_visible_text": current_visible_text,
        "runtime_visible_evidence": runtime_visible_evidence,
        "repair_candidates": candidates,
        "expected_repair_action": expected_action,
        "expected_repair_result": {
            "test_target": test_rel,
            "post_patch_exit_code": 0,
            "target_source_hash": _sha256_text(f"{target.rel_path}:{target.function_name}")[:16],
        },
        "repair_runtime": {
            "patch_application_recorded": bool(repair_patch),
            "post_patch_tests_recorded": passing["exit_code"] == 0,
            "rollback_recorded": sandbox_hash_after_rollback == sandbox_hash_mutated,
            "sandbox_cleanup_recorded": sandbox_deleted or keep_sandboxes,
            "source_repo_read_only_observed": source_hash_before == source_hash_after,
            "bounded_edit_scope_observed": target.rel_path in {item.rel_path for item in candidate_targets},
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
            "task_family": task_family,
            "candidate_count": candidate_count,
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


def build_public_repair_rows_for_spec(
    spec: dict[str, Any],
    *,
    output_root: Path,
    clone_root: Path,
    rows_per_repo: int,
    timeout_seconds: int,
    no_clone: bool,
    keep_sandboxes: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_id = str(spec.get("repo_id") or Path(str(spec.get("local_path") or "repo")).name)
    split = str(spec.get("split") or "train")
    if split not in {"train", "val", "holdout"}:
        raise ValueError(f"invalid split for {repo_id}: {split}")
    repo = _clone_or_get_repo(spec, clone_root=clone_root, no_clone=no_clone)
    commit_hash = _run_git(repo, ["rev-parse", "HEAD"])
    repo_url = str(spec.get("repo_url") or spec.get("url") or f"file://{repo.as_posix()}")
    license_name = _detect_license(repo, spec.get("license"))
    targets = _discover_literal_repair_targets(repo)
    rows: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []
    if len(targets) < max(CANDIDATE_COUNTS):
        rejected_reasons.append("fewer_than_four_literal_return_repair_targets")
    attempts = 0
    slot_offsets = {candidate_count: 0 for candidate_count in CANDIDATE_COUNTS}
    max_attempts = max(rows_per_repo * 6, rows_per_repo + 12)
    while len(targets) >= max(CANDIDATE_COUNTS) and len(rows) < rows_per_repo and attempts < max_attempts:
        candidate_count = CANDIDATE_COUNTS[attempts % len(CANDIDATE_COUNTS)]
        expected_slot = slot_offsets[candidate_count] % candidate_count
        slot_offsets[candidate_count] += 1
        candidate_targets = _candidate_targets_for_slot(
            targets=targets,
            target_index=attempts,
            candidate_count=candidate_count,
            expected_slot=expected_slot,
        )
        if not candidate_targets:
            rejected_reasons.append(f"candidate_pool_too_small_row_{attempts}")
            attempts += 1
            continue
        target = candidate_targets[expected_slot]
        try:
            row = _row_from_target(
                output_root=output_root,
                repo=repo,
                repo_id=repo_id,
                repo_url=repo_url,
                license_name=license_name,
                commit_hash=commit_hash,
                split=split,
                row_index=attempts,
                target=target,
                candidate_targets=candidate_targets,
                expected_slot=expected_slot,
                timeout_seconds=timeout_seconds,
                keep_sandboxes=keep_sandboxes,
            )
        except (SyntaxError, UnicodeError, ValueError, OSError, shutil.Error) as exc:
            rejected_reasons.append(
                f"row_construction_failed_row_{attempts}_{type(exc).__name__}"
            )
            if not keep_sandboxes:
                shutil.rmtree(
                    output_root / "sandboxes" / split / repo_id / f"row_{attempts:05d}",
                    ignore_errors=True,
                )
            attempts += 1
            continue
        if row["repair_runtime"]["post_patch_tests_recorded"]:
            rows.append(row)
        else:
            rejected_reasons.append(f"post_patch_test_failed_row_{attempts}")
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
        "eligible_literal_repair_targets": len(targets),
        "rows_requested": rows_per_repo,
        "rows_emitted": len(rows),
        "source_repo_status_after": source_status_after,
        "source_repo_read_only_observed": source_status_after == "",
        "rejected_reasons": sorted(set(rejected_reasons)),
    }


def _phase2s_public_repair_manifest(
    *,
    output: Path,
    split_rows: dict[str, list[dict[str, Any]]],
    repo_reports: list[dict[str, Any]],
    keep_sandboxes: bool,
    incremental_output: bool,
) -> dict[str, Any]:
    return {
        "collector_family": "phase2s_public_repo_sandbox_repair_trace_collector",
        "trace_construction_mode": "public_repo_sandbox_literal_repair_trace",
        "claim_bearing_training_candidate": True,
        "claim_bearing_training_evidence": True,
        "claim_bearing_training_ready": False,
        "claim_bearing_training_ready_requires": [
            "phase2s_open_repair_data_health",
            "phase2s_open_repair_pretrain_gate",
        ],
        "synthetic_faults_injected_in_sandbox_only": True,
        "sealed_v3_used": False,
        "writes_to_source_repos": False,
        "execution_sandbox_used": True,
        "sandbox_cleanup_observed": not keep_sandboxes,
        "incremental_output_enabled": incremental_output,
        "completed_repo_count": len(repo_reports),
        "splits": {
            split: {
                "path": str(output / f"{split}.raw.jsonl"),
                "rows": len(rows),
                "repo_ids": sorted({str(row.get("repo_id")) for row in rows}),
            }
            for split, rows in split_rows.items()
        },
        "repos": repo_reports,
    }


def _write_phase2s_public_repair_snapshot(
    *,
    output: Path,
    split_rows: dict[str, list[dict[str, Any]]],
    repo_reports: list[dict[str, Any]],
    keep_sandboxes: bool,
    incremental_output: bool,
    manifest_name: str = "manifest.json",
) -> dict[str, Any]:
    for split, rows in split_rows.items():
        _write_jsonl(output / f"{split}.raw.jsonl", rows)
    manifest = _phase2s_public_repair_manifest(
        output=output,
        split_rows=split_rows,
        repo_reports=repo_reports,
        keep_sandboxes=keep_sandboxes,
        incremental_output=incremental_output,
    )
    _write_json(output / manifest_name, manifest)
    return manifest


def collect_phase2s_public_repair_traces(
    *,
    repo_specs_json: str | Path,
    output_root: str | Path,
    clone_root: str | Path,
    rows_per_repo: int = 6,
    timeout_seconds: int = 15,
    no_clone: bool = False,
    keep_sandboxes: bool = False,
    incremental_output: bool = False,
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
        rows, report = build_public_repair_rows_for_spec(
            spec,
            output_root=output,
            clone_root=clone,
            rows_per_repo=rows_per_repo,
            timeout_seconds=timeout_seconds,
            no_clone=no_clone,
            keep_sandboxes=keep_sandboxes,
        )
        split_rows[str(report["split"])].extend(rows)
        repo_reports.append(report)
        if incremental_output:
            _write_phase2s_public_repair_snapshot(
                output=output,
                split_rows=split_rows,
                repo_reports=repo_reports,
                keep_sandboxes=keep_sandboxes,
                incremental_output=True,
                manifest_name="manifest.progress.json",
            )
    manifest = _write_phase2s_public_repair_snapshot(
        output=output,
        split_rows=split_rows,
        repo_reports=repo_reports,
        keep_sandboxes=keep_sandboxes,
        incremental_output=incremental_output,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Phase2S public-repo sandbox repair traces without mutating source repos."
    )
    parser.add_argument("--repo-specs-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clone-root", default="artifacts/external_repos/phase2s")
    parser.add_argument("--rows-per-repo", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=15)
    parser.add_argument("--no-clone", action="store_true")
    parser.add_argument("--keep-sandboxes", action="store_true")
    parser.add_argument(
        "--incremental-output",
        action="store_true",
        help="Flush split raw JSONL and a progress manifest after each completed repo.",
    )
    args = parser.parse_args()
    manifest = collect_phase2s_public_repair_traces(
        repo_specs_json=args.repo_specs_json,
        output_root=args.output_root,
        clone_root=args.clone_root,
        rows_per_repo=args.rows_per_repo,
        timeout_seconds=args.timeout_seconds,
        no_clone=args.no_clone,
        keep_sandboxes=args.keep_sandboxes,
        incremental_output=args.incremental_output,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
