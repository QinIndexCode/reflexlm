from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ignore_tree(_: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in IGNORE_TREE_NAMES or name.endswith((".pyc", ".pyo"))
    }


def _run_pytest_target(
    sandbox: Path,
    target_rel: str,
    *,
    timeout_seconds: int,
    python_executable: str,
) -> dict[str, Any]:
    target_path = (sandbox / target_rel).resolve()
    root = sandbox.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ini = root / "pytest.ini"
    if not ini.exists():
        ini.write_text("[pytest]\naddopts =\n", encoding="utf-8")
    args = [
        python_executable,
        "-m",
        "pytest",
        "-q",
        "--rootdir",
        str(root),
        "--confcutdir",
        str(root),
        "--override-ini",
        "addopts=",
        "-c",
        str(ini),
        str(target_path.relative_to(root) if target_path.exists() else target_rel),
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
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        return {
            "exit_code": int(completed.returncode),
            "timed_out": False,
            "duration_seconds": round(time.perf_counter() - start, 3),
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
            "target": target_rel,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": None,
            "timed_out": True,
            "duration_seconds": round(time.perf_counter() - start, 3),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "target": target_rel,
        }


def _copy_synthetic_source_repo(row: dict[str, Any], dataset_root: Path, sandbox: Path) -> Path:
    split = str(row.get("split") or "")
    repo_id = str(row.get("repo_id") or "")
    source_repo = dataset_root / "source_repos" / split / repo_id
    if not source_repo.exists():
        raise FileNotFoundError(f"synthetic source repo missing: {source_repo}")
    if sandbox.exists():
        shutil.rmtree(sandbox)
    shutil.copytree(source_repo, sandbox, ignore=_ignore_tree)
    return source_repo


def _state_for_policy(
    *,
    row: dict[str, Any],
    pre_test: dict[str, Any],
    test_rel: str,
) -> SystemStateFrame:
    evidence = row.get("runtime_visible_evidence") if isinstance(row.get("runtime_visible_evidence"), dict) else {}
    watched_files = [test_rel] + [str(path) for path in evidence.get("changed_files", [])]
    candidates = row.get("repair_candidates") if isinstance(row.get("repair_candidates"), list) else []
    return SystemStateFrame(
        time=TimeState(tick=1, runtime_ms=int(float(pre_test["duration_seconds"]) * 1000)),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description=(
                "Phase2Z synthetic-safe nonliteral repair plumbing task. "
                "Authorize only bounded patch, test selection, rollback safety, and verified stop."
            ),
            command_allowlist=[
                str(candidate.get("verification_command"))
                for candidate in candidates
                if isinstance(candidate, dict) and candidate.get("verification_command")
            ],
            watched_paths=[path for path in watched_files if path],
            success_criteria=[
                "selected_tests_pass",
                "bounded_write_scope_respected",
                "recorded_patch_operator_is_non_claim_evidence",
            ],
            safety_notes=[
                "synthetic_safe_non_claim_plumbing_only",
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
            stdout_delta=str(pre_test.get("stdout") or ""),
            stderr_delta=str(pre_test.get("stderr") or ""),
            stdout_lines=len(str(pre_test.get("stdout") or "").splitlines()),
            stderr_lines=len(str(pre_test.get("stderr") or "").splitlines()),
            last_command=f"python -m pytest -q {test_rel} --maxfail=1",
        ),
        filesystem=FileSystemState(
            watched_paths=[path for path in watched_files if path],
            changed_paths=list(evidence.get("changed_files") or []),
            dirty_files=list(evidence.get("changed_files") or []),
        ),
    )


def _manifest_hash(package_manifest: Path) -> str:
    return _sha256_file(package_manifest)


def _manifest_open_repair_outputs(manifest: dict[str, Any]) -> dict[str, int]:
    capabilities = (
        manifest.get("open_repair_capabilities")
        if isinstance(manifest.get("open_repair_capabilities"), dict)
        else {}
    )
    def enabled(*names: str) -> int:
        return int(any(bool(capabilities.get(name)) for name in names))

    return {
        "patch_proposal": enabled("patch_proposal", "patch_proposal_head"),
        "test_selection": enabled("test_selection", "test_selection_head"),
        "rollback_safety": enabled("rollback_safety", "rollback_safety_head"),
        "bounded_edit_scope": enabled("bounded_edit_scope", "bounded_edit_scope_policy"),
        "progress_monitor": enabled("progress_monitor", "progress_monitor_receptors"),
        "verification_state": enabled("verification_state", "verification_state_receptors"),
        "stop_condition": enabled("stop_condition", "stop_condition_head"),
    }


def _git_apply_patch(sandbox: Path, patch_path: Path, timeout_seconds: int) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path.resolve())],
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


def _patch_stats(patch_text: str) -> dict[str, Any]:
    files = [
        line[6:].strip()
        for line in patch_text.splitlines()
        if line.startswith("+++ b/")
    ]
    added = sum(1 for line in patch_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in patch_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    return {
        "changed_file_count": len(files),
        "changed_files": files,
        "added_line_count": added,
        "removed_line_count": removed,
        "multi_file": len(files) >= 2,
        "nonliteral_structure_present": (
            len(files) >= 2
            or "import " in patch_text
            or "from " in patch_text
            or ".lower()" in patch_text
            or ".upper()" in patch_text
            or "normalize_" in patch_text
        ),
    }


def run_phase2z_synthetic_nonliteral_repair_plumbing(
    *,
    source_rows_jsonl: str | Path,
    dataset_root: str | Path,
    package_path: str | Path,
    output_jsonl: str | Path,
    artifact_root: str | Path,
    max_rows: int = 16,
    timeout_seconds: int = 30,
    test_python: str | None = None,
    load_policy: bool = True,
) -> dict[str, Any]:
    source_rows = _read_jsonl(source_rows_jsonl)
    package_dir = Path(package_path)
    manifest_path = package_dir if package_dir.is_file() else package_dir / PACKAGE_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    package_hash = _manifest_hash(manifest_path)
    policy = NativeNervousPolicyPackage(package_dir) if load_policy else None
    python_executable = test_python or sys.executable
    root = Path(dataset_root)
    artifacts = Path(artifact_root)
    rows: list[dict[str, Any]] = []
    for row in source_rows[:max_rows]:
        start = time.perf_counter()
        row_id = str(row.get("trace_id") or f"row-{len(rows)}").replace(":", "_")
        row_artifacts = artifacts / row_id
        sandbox = row_artifacts / "sandbox"
        row_artifacts.mkdir(parents=True, exist_ok=True)
        _copy_synthetic_source_repo(row, root, sandbox)
        expected = row.get("expected_repair_result") if isinstance(row.get("expected_repair_result"), dict) else {}
        test_rel = str(expected.get("test_target") or "tests/test_repair_case.py")
        pre_test = _run_pytest_target(
            sandbox,
            test_rel,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        state = _state_for_policy(row=row, pre_test=pre_test, test_rel=test_rel)
        policy_outputs: dict[str, Any] = {}
        if policy is not None:
            policy.act(state)
            policy_outputs = dict(policy.last_call)
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
        )
        artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
        patch_source = root / str(artifact_paths.get("patch_diff") or "")
        patch_text = patch_source.read_text(encoding="utf-8") if patch_source.exists() else ""
        local_patch = row_artifacts / "recorded_synthetic_patch.diff"
        local_patch.write_text(patch_text, encoding="utf-8")
        progress_trace = [
            {"event": "sandbox_prepared"},
            {"event": "pre_test_finished", "exit_code": pre_test["exit_code"]},
            {"event": "policy_control_heads_observed", "available": bool(policy_outputs)},
        ]
        patch_apply = {
            "exit_code": None,
            "timed_out": False,
            "duration_seconds": 0.0,
            "stdout": "",
            "stderr": "patch_not_authorized",
        }
        if patch_authorized and patch_text:
            patch_apply = _git_apply_patch(sandbox, local_patch, timeout_seconds)
            progress_trace.append({"event": "recorded_synthetic_patch_applied", "exit_code": patch_apply["exit_code"]})
        else:
            progress_trace.append({"event": "recorded_patch_application_blocked_by_open_repair_heads"})
        post_test = _run_pytest_target(
            sandbox,
            test_rel,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        progress_trace.append({"event": "post_test_finished", "exit_code": post_test["exit_code"]})
        stats = _patch_stats(patch_text)
        success = (
            pre_test["exit_code"] not in {0, None}
            and patch_authorized
            and patch_apply["exit_code"] == 0
            and post_test["exit_code"] == 0
        )
        pre_path = row_artifacts / "pre_test_log.json"
        patch_apply_path = row_artifacts / "patch_apply_log.json"
        post_path = row_artifacts / "post_test_log.json"
        transcript_path = row_artifacts / "transcript.json"
        _write_json(pre_path, pre_test)
        _write_json(patch_apply_path, patch_apply)
        _write_json(post_path, post_test)
        _write_json(
            transcript_path,
            {
                "trace_id": row.get("trace_id"),
                "policy_outputs": policy_outputs,
                "progress_monitor_trace": progress_trace,
                "claim_boundary": "synthetic_nonliteral_plumbing_only_not_model_patch_generation",
            },
        )
        rows.append(
            {
                "trace_id": row.get("trace_id"),
                "task_id": f"phase2z:{row.get('trace_id')}",
                "task_family": "open_repair_synthetic_nonliteral_plumbing",
                "repair_mode": "synthetic_nonliteral_patch_plumbing",
                "source_kind": row.get("source_kind"),
                "repo_origin": row.get("repo_url_or_origin"),
                "repo_commit": row.get("commit_hash"),
                "result_source": "phase2z_synthetic_safe_nonliteral_plumbing_execution",
                "native_policy_label": str(manifest.get("policy_label") or ""),
                "policy_package_manifest_sha256": package_hash,
                "policy_open_repair_outputs": open_repair_outputs,
                "patch_source": "recorded_synthetic_patch_diff_operator"
                if patch_authorized
                else "package_runtime_no_patch_authorized",
                "patch_generator": "synthetic_safe_recorded_diff_operator_v1",
                "patch_proposal": patch_text,
                "patch_sha256": _sha256_text(patch_text),
                "patch_stats": stats,
                "selected_tests": [f"python -m pytest -q {test_rel} --maxfail=1"],
                "pre_test_log_sha256": _sha256_file(pre_path),
                "post_test_log_sha256": _sha256_file(post_path),
                "patch_apply_log_sha256": _sha256_file(patch_apply_path),
                "verification_state": "passed" if success else "failed",
                "progress_monitor_trace": progress_trace,
                "stop_condition": "verification_passed" if success else "verification_failed_stop",
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "transcript_sha256": _sha256_file(transcript_path),
                "oracle_trace_used": True,
                "recorded_patch_artifact_used": True,
                "claim_bearing_execution_evidence": False,
                "sealed_feedback_used": False,
                "success": success,
                "full_task_success": success,
                "full_patch_correctness": success,
                "full_test_pass_rate": 1.0 if post_test["exit_code"] == 0 else 0.0,
                "unauthorized_write_count": 0,
                "false_completion": False,
                "claim_boundary": "synthetic_nonliteral_plumbing_only_not_model_patch_generation",
                "artifact_paths": {
                    "patch": str(local_patch),
                    "pre_test_log": str(pre_path),
                    "patch_apply_log": str(patch_apply_path),
                    "post_test_log": str(post_path),
                    "transcript": str(transcript_path),
                },
            }
        )
    _write_jsonl(output_jsonl, rows)
    successes = sum(1 for row in rows if row.get("success") is True)
    return {
        "artifact_family": "phase2z_synthetic_nonliteral_repair_plumbing_runner",
        "rows": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "output_jsonl": str(Path(output_jsonl)),
        "artifact_root": str(Path(artifact_root)),
        "claim_boundary": "synthetic_nonliteral_plumbing_only_not_claim_bearing",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase2Z synthetic-safe nonliteral repair plumbing execution."
    )
    parser.add_argument("--source-rows-jsonl", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--test-python")
    parser.add_argument("--no-load-policy", action="store_true")
    args = parser.parse_args()
    report = run_phase2z_synthetic_nonliteral_repair_plumbing(
        source_rows_jsonl=args.source_rows_jsonl,
        dataset_root=args.dataset_root,
        package_path=args.package_path,
        output_jsonl=args.output_jsonl,
        artifact_root=args.artifact_root,
        max_rows=args.max_rows,
        timeout_seconds=args.timeout_seconds,
        test_python=args.test_python,
        load_policy=not args.no_load_policy,
    )
    _write_json(args.summary_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["rows"] <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
