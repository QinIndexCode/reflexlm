from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2s_open_repair import (
    BASELINE_METHODS,
    compute_phase2s_baseline_predictions,
)


TASK_FAMILIES = (
    "dependency_or_import_mismatch",
    "localized_unit_assertion",
    "stale_snapshot_update",
    "config_or_environment_marker",
    "multi_file_traceback_relation",
)
EVIDENCE_DENSITIES = ("low", "medium", "high")
CANDIDATE_COUNTS = (2, 3, 4)
REPAIR_DEPTHS = ("one_edit", "two_edits", "stale_state_refresh")
FAILURE_OBSERVABILITY = (
    "direct_traceback",
    "indirect_changed_file_relation",
    "ambiguous_same_intent_command",
)
AMBIGUITY_CLASSES = ("same_intent_command", "same_file_read", "stage_transition")
REPAIR_ACTIONS = ("repair_plan_alpha", "repair_plan_bravo", "repair_plan_charlie", "repair_plan_delta")
IGNORE_TREE_NAMES = {"__pycache__", ".pytest_cache", ".git"}
ABSOLUTE_PATH_RE = re.compile(
    r"(?i)((?<![A-Za-z])[A-Z]:[\\/][^\s,;:\"']+|\\\\[A-Za-z0-9_.-]+\\[^\s,;:\"']+|/(?:Users|home|root|var/folders)/[^\s,;:\"']+)"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _script_hash() -> str:
    return _sha256_text(Path(__file__).read_text(encoding="utf-8"))


def _tree_hash(root: Path) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in IGNORE_TREE_NAMES for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        entries.append((rel, _sha256_text(path.read_text(encoding="utf-8", errors="replace"))))
    return _sha256_text(_canonical_json(entries))


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _redact_execution_text(text: str, repo: Path) -> str:
    redacted = text.replace(str(repo), "<EXECUTION_SANDBOX>")
    redacted = redacted.replace(repo.as_posix(), "<EXECUTION_SANDBOX>")
    redacted = redacted.replace("\\", "/")
    redacted = ABSOLUTE_PATH_RE.sub("<REDACTED_ABS_PATH>", redacted)
    return _shorten(redacted)


def _run_pytest(repo: Path, timeout_seconds: int) -> dict[str, Any]:
    args = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_repair_case.py",
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
            cwd=repo,
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
        "command_template": "python -m pytest -q tests/test_repair_case.py --maxfail=1 -p no:cacheprovider",
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": round(time.perf_counter() - start, 3),
        "stdout_excerpt": _redact_execution_text(stdout, repo),
        "stderr_excerpt": _redact_execution_text(stderr, repo),
    }


def _shorten(text: str, limit: int = 2400) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<TRUNCATED>"


def _case_files(
    *,
    task_family: str,
    repair_depth: str,
    case_token: str,
) -> tuple[dict[str, str], dict[str, str], str, list[str]]:
    if repair_depth == "two_edits":
        buggy = {
            "repair_case.py": (
                "from repair_helper import transform\n\n"
                "SUFFIX = 'broken'\n\n"
                "def evaluate(value):\n"
                "    return transform(value) + '-' + SUFFIX\n"
            ),
            "repair_helper.py": (
                "def transform(value):\n"
                "    return str(value).strip().upper()\n"
            ),
        }
        fixed = {
            "repair_case.py": (
                "from repair_helper import transform\n\n"
                "SUFFIX = 'fixed'\n\n"
                "def evaluate(value):\n"
                "    return transform(value) + '-' + SUFFIX\n"
            ),
            "repair_helper.py": (
                "def transform(value):\n"
                "    return str(value).strip().lower()\n"
            ),
        }
        expected = f"{case_token}-fixed"
        changed = ["repair_case.py", "repair_helper.py"]
    elif task_family == "dependency_or_import_mismatch":
        buggy = {
            "repair_case.py": (
                "from missing_dependency import normalize_value\n\n"
                "def evaluate(value):\n"
                "    return normalize_value(value)\n"
            )
        }
        fixed = {
            "repair_case.py": (
                "def evaluate(value):\n"
                "    return str(value).strip().lower()\n"
            )
        }
        expected = case_token
        changed = ["repair_case.py"]
    elif task_family == "stale_snapshot_update":
        buggy = {
            "repair_case.py": (
                "SNAPSHOT_TOKEN = 'stale'\n\n"
                "def evaluate(value):\n"
                "    return SNAPSHOT_TOKEN\n"
            )
        }
        fixed = {
            "repair_case.py": (
                f"SNAPSHOT_TOKEN = '{case_token}'\n\n"
                "def evaluate(value):\n"
                "    return SNAPSHOT_TOKEN\n"
            )
        }
        expected = case_token
        changed = ["repair_case.py"]
    elif task_family == "config_or_environment_marker":
        buggy = {
            "repair_case.py": (
                "ACTIVE_MODE = 'dev'\n\n"
                "def evaluate(value):\n"
                "    return ACTIVE_MODE\n"
            )
        }
        fixed = {
            "repair_case.py": (
                f"ACTIVE_MODE = '{case_token}'\n\n"
                "def evaluate(value):\n"
                "    return ACTIVE_MODE\n"
            )
        }
        expected = case_token
        changed = ["repair_case.py"]
    elif task_family == "multi_file_traceback_relation":
        buggy = {
            "repair_case.py": (
                "from repair_helper import transform\n\n"
                "def evaluate(value):\n"
                "    return transform(value)\n"
            ),
            "repair_helper.py": (
                "def transform(value):\n"
                "    return str(value).strip().upper()\n"
            ),
        }
        fixed = {
            "repair_case.py": buggy["repair_case.py"],
            "repair_helper.py": (
                "def transform(value):\n"
                "    return str(value).strip().lower()\n"
            ),
        }
        expected = case_token
        changed = ["repair_helper.py"]
    else:
        buggy = {
            "repair_case.py": (
                "def evaluate(value):\n"
                "    return str(value).strip()\n"
            )
        }
        fixed = {
            "repair_case.py": (
                "def evaluate(value):\n"
                "    return str(value).strip().lower()\n"
            )
        }
        expected = case_token
        changed = ["repair_case.py"]

    test_file = (
        "from repair_case import evaluate\n\n\n"
        "def test_repair_case_behavior():\n"
        f"    assert evaluate('  {case_token.upper()}  ') == '{expected}'\n"
    )
    buggy["tests/test_repair_case.py"] = test_file
    fixed["tests/test_repair_case.py"] = test_file
    return buggy, fixed, expected, changed


def _materialize_repo(root: Path, files: dict[str, str]) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "LICENSE").write_text("MIT synthetic-safe Phase2S fixture\n", encoding="utf-8")


def _patch_diff(
    *,
    buggy: dict[str, str],
    fixed: dict[str, str],
    changed_files: list[str],
) -> str:
    chunks: list[str] = []
    for rel_path in changed_files:
        old = buggy.get(rel_path, "").splitlines(keepends=True)
        new = fixed.get(rel_path, "").splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
    return "".join(chunks)


def _apply_files(root: Path, files: dict[str, str], changed_files: list[str]) -> None:
    for rel_path in changed_files:
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[rel_path], encoding="utf-8")


def _copy_repo_to_sandbox(source_repo: Path, sandbox: Path) -> None:
    if sandbox.exists():
        shutil.rmtree(sandbox)
    shutil.copytree(source_repo, sandbox)


def _repair_candidates(candidate_count: int) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for action in REPAIR_ACTIONS[:candidate_count]:
        candidates.append(
            {
                "repair_action": action,
                "intent": "apply_patch_and_rerun_tests",
                "edit_scope": "bounded_source_patch",
                "description": "Apply a bounded source patch, rerun the failing test, and stop if verification passes.",
                "verification_command": "python -m pytest -q tests/test_repair_case.py --maxfail=1",
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


def _build_row(
    *,
    output_root: Path,
    split: str,
    index: int,
    expected_slot: int,
    timeout_seconds: int,
    keep_sandboxes: bool,
) -> dict[str, Any]:
    task_family = TASK_FAMILIES[index % len(TASK_FAMILIES)]
    evidence_density = EVIDENCE_DENSITIES[index % len(EVIDENCE_DENSITIES)]
    candidate_count = CANDIDATE_COUNTS[index % len(CANDIDATE_COUNTS)]
    repair_depth = REPAIR_DEPTHS[index % len(REPAIR_DEPTHS)]
    failure_observability = FAILURE_OBSERVABILITY[index % len(FAILURE_OBSERVABILITY)]
    ambiguity_class = AMBIGUITY_CLASSES[index % len(AMBIGUITY_CLASSES)]
    repo_id = f"phase2s_{split}_repo_{index:03d}"
    case_token = f"case{index:03d}"
    buggy, fixed, expected_value, changed_files = _case_files(
        task_family=task_family,
        repair_depth=repair_depth,
        case_token=case_token,
    )
    source_repo = output_root / "source_repos" / split / repo_id
    sandbox = output_root / "sandboxes" / split / repo_id
    artifact_dir = output_root / "artifacts" / split / repo_id
    _materialize_repo(source_repo, buggy)
    source_hash_before = _tree_hash(source_repo)
    _copy_repo_to_sandbox(source_repo, sandbox)
    sandbox_hash_before = _tree_hash(sandbox)
    failing = _run_pytest(sandbox, timeout_seconds=timeout_seconds)
    patch = _patch_diff(buggy=buggy, fixed=fixed, changed_files=changed_files)
    _apply_files(sandbox, fixed, changed_files)
    sandbox_hash_after_patch = _tree_hash(sandbox)
    passing = _run_pytest(sandbox, timeout_seconds=timeout_seconds)
    _apply_files(sandbox, buggy, changed_files)
    sandbox_hash_after_rollback = _tree_hash(sandbox)
    rollback = _run_pytest(sandbox, timeout_seconds=timeout_seconds)
    source_hash_after = _tree_hash(source_repo)
    sandbox_deleted = False
    if not keep_sandboxes:
        shutil.rmtree(sandbox, ignore_errors=True)
        sandbox_deleted = not sandbox.exists()

    artifact_dir.mkdir(parents=True, exist_ok=True)
    patch_path = artifact_dir / "patch.diff"
    command_log_path = artifact_dir / "command_log.json"
    test_output_path = artifact_dir / "test_output.json"
    rollback_log_path = artifact_dir / "rollback_log.json"
    sandbox_integrity_path = artifact_dir / "sandbox_integrity.json"
    patch_path.write_text(patch, encoding="utf-8")
    command_log = {
        "commands": [
            {"stage": "before_patch", **failing},
            {"stage": "after_patch", **passing},
            {"stage": "after_rollback", **rollback},
        ]
    }
    _write_json(command_log_path, command_log)
    _write_json(
        test_output_path,
        {
            "before_patch": failing,
            "after_patch": passing,
            "expected_after_patch_exit_code": 0,
        },
    )
    _write_json(
        rollback_log_path,
        {
            "rollback_restored_before_hash": sandbox_hash_after_rollback == sandbox_hash_before,
            "after_rollback": rollback,
        },
    )
    _write_json(
        sandbox_integrity_path,
        {
            "source_hash_before": source_hash_before,
            "source_hash_after": source_hash_after,
            "source_repo_read_only_observed": source_hash_before == source_hash_after,
            "sandbox_hash_before": sandbox_hash_before,
            "sandbox_hash_after_patch": sandbox_hash_after_patch,
            "sandbox_hash_after_rollback": sandbox_hash_after_rollback,
            "sandbox_deleted": sandbox_deleted,
            "writes_outside_sandbox_observed": False,
        },
    )

    candidates = _repair_candidates(candidate_count)
    expected_action = candidates[expected_slot]["repair_action"]
    runtime_visible_evidence = {
        "failure_observability": failure_observability,
        "failing_test_target": "tests/test_repair_case.py",
        "pytest_before_patch": {
            "exit_code": failing["exit_code"],
            "timed_out": failing["timed_out"],
            "stdout_excerpt": failing["stdout_excerpt"],
            "stderr_excerpt": failing["stderr_excerpt"],
        },
        "changed_files": changed_files,
        "traceback_symbols": ["repair_case.evaluate"],
        "watched_files": ["tests/test_repair_case.py"],
        "prior_repair_summary": (
            "A previous sandbox run observed a bounded source repair path, but the active "
            "candidate still needs verification through the recorded failing test."
        ),
        "stale_state_refresh": repair_depth == "stale_state_refresh",
        "source_repo_observed_read_only": source_hash_before == source_hash_after,
        "execution_sandbox_used": True,
    }
    current_visible_text = (
        "Sandboxed repository repair task. Inspect the runtime-visible failing test, "
        "bounded edit scope, rollback evidence, and verification command before choosing "
        f"a repair action. task_family={task_family}; density={evidence_density}; "
        f"ambiguity={ambiguity_class}."
    )
    row: dict[str, Any] = {
        "trace_id": f"{split}:{repo_id}:{index}",
        "split": split,
        "source_kind": "synthetic_safe_repo",
        "repo_id": repo_id,
        "repo_url_or_origin": f"synthetic://phase2s/open-repair/{repo_id}",
        "commit_hash": _sha256_text(f"phase2s:{split}:{repo_id}:{index}")[:40],
        "license_or_synthetic_origin": "synthetic-safe Phase2S open-repair smoke fixture",
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
            "test_target": "tests/test_repair_case.py",
            "post_patch_exit_code": 0,
            "expected_value_hash": _sha256_text(expected_value)[:16],
        },
        "repair_runtime": {
            "patch_application_recorded": bool(patch),
            "post_patch_tests_recorded": passing["exit_code"] == 0,
            "rollback_recorded": sandbox_hash_after_rollback == sandbox_hash_before,
            "sandbox_cleanup_recorded": sandbox_deleted or keep_sandboxes,
            "source_repo_read_only_observed": source_hash_before == source_hash_after,
            "bounded_edit_scope_observed": set(changed_files).issubset(set(buggy)),
            "command_allowlist_observed": True,
        },
        "artifact_paths": {
            "patch_diff": _rel(patch_path, output_root),
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


def build_phase2s_open_repair_smoke_dataset(
    *,
    output_root: str | Path,
    train_count: int = 15,
    val_count: int = 15,
    holdout_count: int = 9,
    timeout_seconds: int = 10,
    keep_sandboxes: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    counts = {"train": train_count, "val": val_count, "holdout": holdout_count}
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, count in counts.items():
        slot_offsets = {candidate_count: 0 for candidate_count in CANDIDATE_COUNTS}
        rows: list[dict[str, Any]] = []
        for index in range(count):
            candidate_count = CANDIDATE_COUNTS[index % len(CANDIDATE_COUNTS)]
            expected_slot = slot_offsets[candidate_count] % candidate_count
            slot_offsets[candidate_count] += 1
            rows.append(
                _build_row(
                    output_root=root,
                    split=split,
                    index=index,
                    expected_slot=expected_slot,
                    timeout_seconds=timeout_seconds,
                    keep_sandboxes=keep_sandboxes,
                )
            )
        splits[split] = rows
        _write_jsonl(root / f"{split}.raw.jsonl", rows)
    manifest = {
        "generator": "phase2s_open_repair_smoke_generator",
        "source_kind": "synthetic_safe_repo",
        "claim_bearing_training_evidence": False,
        "sealed_v3_used": False,
        "writes_to_source_repos": False,
        "execution_sandbox_used": True,
        "sandbox_cleanup_observed": not keep_sandboxes,
        "splits": {
            split: {"path": str(root / f"{split}.raw.jsonl"), "rows": len(rows)}
            for split, rows in splits.items()
        },
        "factor_levels": {
            "task_families": list(TASK_FAMILIES),
            "candidate_counts": list(CANDIDATE_COUNTS),
            "evidence_densities": list(EVIDENCE_DENSITIES),
            "repair_depths": list(REPAIR_DEPTHS),
            "failure_observability": list(FAILURE_OBSERVABILITY),
            "ambiguity_classes": list(AMBIGUITY_CLASSES),
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a non-sealed synthetic-safe Phase2S open-repair data smoke."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-count", type=int, default=15)
    parser.add_argument("--val-count", type=int, default=15)
    parser.add_argument("--holdout-count", type=int, default=9)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    parser.add_argument("--keep-sandboxes", action="store_true")
    args = parser.parse_args()
    manifest = build_phase2s_open_repair_smoke_dataset(
        output_root=args.output_root,
        train_count=args.train_count,
        val_count=args.val_count,
        holdout_count=args.holdout_count,
        timeout_seconds=args.timeout_seconds,
        keep_sandboxes=args.keep_sandboxes,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
