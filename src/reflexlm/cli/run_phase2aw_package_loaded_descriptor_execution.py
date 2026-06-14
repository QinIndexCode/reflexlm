from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from reflexlm.cli.run_phase2z_public_structural_repair_execution import (
    _copy_public_repo,
    _git_apply_reverse_patch,
    _materialize_generated_test,
    _resolve_patch,
    _state_after_stderr_receptor,
    _state_for_public_policy,
    run_phase2z_public_structural_repair_execution,
)
from reflexlm.cli.run_phase2z_synthetic_nonliteral_repair_plumbing import (
    _read_json,
    _read_jsonl,
    _run_pytest_target,
    _sha256_text,
    _write_json,
    _write_jsonl,
)
from reflexlm.llm.native_nervous_package import PACKAGE_MANIFEST_NAME, NativeNervousPolicyPackage


CLAIM_BOUNDARY = (
    "phase2aw_package_loaded_bounded_descriptor_execution_not_freeform_patch_generation"
)


def _repo_id_from_origin(origin: str) -> str:
    parsed = urlparse(origin)
    path = parsed.path if parsed.scheme else origin
    return path.strip("/").removesuffix(".git").replace("/", "_").replace("-", "_")


def _runner_row(row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    if not converted.get("repo_id"):
        converted["repo_id"] = _repo_id_from_origin(str(row.get("repo_origin") or ""))
    tests = (
        converted.get("artifact_paths", {}).get("generated_tests")
        if isinstance(converted.get("artifact_paths"), dict)
        else None
    )
    test_rel = str(tests[0]) if isinstance(tests, list) and tests else ""
    converted["expected_repair_result"] = {
        "test_target": str(
            row.get("runtime_visible_evidence", {}).get("failing_test_target") or test_rel
        )
    }
    artifact_paths = dict(converted.get("artifact_paths") or {})
    if test_rel and not artifact_paths.get("generated_test"):
        artifact_paths["generated_test"] = test_rel
    if test_rel and not artifact_paths.get("patch_diff"):
        patch_rel = str(Path(test_rel).parent / "patch.diff").replace("\\", "/")
        artifact_paths["patch_diff"] = patch_rel
    converted["artifact_paths"] = artifact_paths
    converted["repo_url_or_origin"] = row.get("repo_origin")
    converted["commit_hash"] = row.get("repo_commit")
    return converted


def _candidate_actions(row: dict[str, Any]) -> list[str]:
    candidates = row.get("repair_candidates") if isinstance(row.get("repair_candidates"), list) else []
    return [
        str(candidate.get("repair_action") or "")
        for candidate in candidates
        if isinstance(candidate, dict)
    ]


def _expected_slot(row: dict[str, Any]) -> int:
    expected = str(row.get("expected_repair_action") or "")
    actions = _candidate_actions(row)
    for index, action in enumerate(actions):
        if action == expected:
            return index
    raise ValueError(f"expected_repair_action is not in repair_candidates: {row.get('task_id')}")


def _selected_command_slot(policy_outputs: dict[str, Any]) -> int | None:
    cortex_plan = policy_outputs.get("cortex_plan")
    if isinstance(cortex_plan, dict) and isinstance(cortex_plan.get("command_slot"), int):
        return int(cortex_plan["command_slot"])
    return None


def _open_repair_authorized(policy_outputs: dict[str, Any]) -> bool:
    heads = policy_outputs.get("open_repair_head_outputs")
    if not isinstance(heads, dict):
        return False
    return (
        heads.get("patch_proposal") == 1
        and heads.get("bounded_edit_scope") == 1
        and heads.get("rollback_safety") == 1
    )


def _select_with_loaded_package(
    *,
    policy: Any,
    row: dict[str, Any],
    dataset_root: Path,
    clone_root: Path,
    row_artifacts: Path,
    timeout_seconds: int,
    python_executable: str,
) -> dict[str, Any]:
    sandbox = row_artifacts / "selection_sandbox"
    row_artifacts.mkdir(parents=True, exist_ok=True)
    _copy_public_repo(row, clone_root, sandbox)
    test_rel = _materialize_generated_test(row, dataset_root, sandbox)
    patch_path, patch_text = _resolve_patch(row, dataset_root)
    patch_copy = row_artifacts / "selection_fault_patch.diff"
    patch_copy.write_text(
        patch_text.replace("\r\n", "\n").replace("\r", "\n"),
        encoding="utf-8",
        newline="\n",
    )
    reverse_to_fault = _git_apply_reverse_patch(sandbox, patch_copy, timeout_seconds)
    pre_test = _run_pytest_target(
        sandbox,
        test_rel,
        timeout_seconds=timeout_seconds,
        python_executable=python_executable,
    )
    if hasattr(policy, "reset"):
        policy.reset()
    state = _state_for_public_policy(row=row, pre_test=pre_test, test_rel=test_rel)
    policy.act(state)
    policy_outputs = dict(getattr(policy, "last_call", {}) or {})
    receptor_observed = False
    if policy_outputs.get("action_source") == "low_level_debug_receptor":
        receptor_observed = True
        state = _state_after_stderr_receptor(state)
        policy.act(state)
        policy_outputs = dict(getattr(policy, "last_call", {}) or {})
    return {
        "test_rel": test_rel,
        "source_patch_artifact": str(patch_path),
        "reverse_to_fault": reverse_to_fault,
        "pre_test": pre_test,
        "policy_outputs": policy_outputs,
        "low_level_debug_receptor_observed": receptor_observed,
        "selected_slot": _selected_command_slot(policy_outputs),
        "open_repair_authorized": _open_repair_authorized(policy_outputs),
        "qwen_called": bool(policy_outputs.get("qwen_called")),
    }


def _execute_selected_row(
    *,
    row: dict[str, Any],
    dataset_root: Path,
    clone_root: Path,
    package_path: Path,
    artifact_root: Path,
    timeout_seconds: int,
    test_python: str | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="phase2aw_pkg_exec_") as tmp:
        one_row = Path(tmp) / "row.jsonl"
        _write_jsonl(one_row, [row])
        report = run_phase2z_public_structural_repair_execution(
            source_rows_jsonl=one_row,
            dataset_root=dataset_root,
            clone_root=clone_root,
            package_path=package_path,
            output_jsonl=Path(tmp) / "execution.jsonl",
            artifact_root=artifact_root,
            max_rows=1,
            timeout_seconds=timeout_seconds,
            test_python=test_python,
            load_policy=False,
            patch_mode="runtime_symbolic_structural",
        )
        rows = _read_jsonl(Path(tmp) / "execution.jsonl")
        executed = rows[0] if rows else {}
        return {"summary": report, "row": executed}


def run_phase2aw_package_loaded_descriptor_execution(
    *,
    runtime_tasks_jsonl: str | Path,
    dataset_root: str | Path,
    clone_root: str | Path,
    package_path: str | Path,
    output_jsonl: str | Path,
    artifact_root: str | Path,
    max_rows: int = 20,
    timeout_seconds: int = 30,
    test_python: str | None = None,
    load_policy: bool = True,
) -> dict[str, Any]:
    import sys

    tasks = [_runner_row(row) for row in _read_jsonl(runtime_tasks_jsonl)[:max_rows]]
    package_dir = Path(package_path)
    manifest_path = package_dir if package_dir.is_file() else package_dir / PACKAGE_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    policy = NativeNervousPolicyPackage(package_dir) if load_policy else None
    if policy is None:
        raise ValueError("Phase2AW package-loaded execution requires load_policy=True")

    root = Path(dataset_root)
    clones = Path(clone_root)
    artifacts = Path(artifact_root)
    artifacts.mkdir(parents=True, exist_ok=True)
    python_executable = test_python or sys.executable
    rows: list[dict[str, Any]] = []

    for index, row in enumerate(tasks):
        started = time.perf_counter()
        expected_slot = _expected_slot(row)
        actions = _candidate_actions(row)
        trace_id = str(row.get("task_id") or row.get("trace_id") or f"row-{index}")
        row_id = f"r{index:05d}_{_sha256_text(trace_id)[:6]}"
        row_artifacts = artifacts / row_id
        selection = _select_with_loaded_package(
            policy=policy,
            row=row,
            dataset_root=root,
            clone_root=clones,
            row_artifacts=row_artifacts,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        selected_slot = selection["selected_slot"]
        selected_action = (
            actions[selected_slot]
            if isinstance(selected_slot, int) and 0 <= selected_slot < len(actions)
            else ""
        )
        selected_correct = selected_slot == expected_slot
        execution_payload: dict[str, Any] = {}
        executed_row: dict[str, Any] = {}
        if selected_correct and selection["open_repair_authorized"]:
            execution_payload = _execute_selected_row(
                row=row,
                dataset_root=root,
                clone_root=clones,
                package_path=package_dir,
                artifact_root=row_artifacts / "x",
                timeout_seconds=timeout_seconds,
                test_python=test_python,
            )
            executed_row = execution_payload.get("row") or {}
            success = executed_row.get("success") is True
            verification_state = executed_row.get("verification_state")
            stop_condition = executed_row.get("stop_condition")
            artifact_paths = executed_row.get("artifact_paths") or {}
        else:
            success = False
            verification_state = "failed"
            stop_condition = (
                "package_open_repair_heads_not_authorized"
                if selected_correct
                else "package_candidate_selection_failed_before_patch_application"
            )
            artifact_paths = {}
        result = {
            "trace_id": row.get("trace_id") or row.get("task_id"),
            "task_id": row.get("task_id"),
            "task_family": "phase2aw_package_loaded_descriptor_execution",
            "result_source": "phase2aw_package_loaded_descriptor_execution",
            "repo_origin": row.get("repo_origin") or row.get("repo_url_or_origin"),
            "repo_commit": row.get("repo_commit") or row.get("commit_hash"),
            "native_policy_label": str(manifest.get("policy_label") or ""),
            "policy_loaded": True,
            "policy_package_manifest_path": str(manifest_path),
            "policy_outputs": selection["policy_outputs"],
            "low_level_debug_receptor_observed": selection[
                "low_level_debug_receptor_observed"
            ],
            "qwen_called": selection["qwen_called"],
            "selected_patch_candidate_slot": selected_slot,
            "expected_patch_candidate_slot": expected_slot,
            "selected_repair_action": selected_action,
            "expected_repair_action": actions[expected_slot],
            "patch_candidate_selected_correctly": selected_correct,
            "open_repair_authorized_by_loaded_package": selection["open_repair_authorized"],
            "phase2z_execution_summary": execution_payload.get("summary"),
            "success": success,
            "full_task_success": success,
            "full_patch_correctness": executed_row.get("full_patch_correctness") is True,
            "full_test_pass_rate": executed_row.get("full_test_pass_rate", 0.0),
            "rollback_failure_restored": executed_row.get("rollback_failure_restored") is True,
            "unauthorized_write_count": int(executed_row.get("unauthorized_write_count") or 0),
            "false_completion": executed_row.get("false_completion") is True,
            "verification_state": verification_state,
            "stop_condition": stop_condition,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "recorded_patch_artifact_used": False,
            "recorded_patch_artifact_used_for_fault_injection": True,
            "freeform_patch_generation": False,
            "sealed_feedback_used": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "artifact_paths": artifact_paths,
        }
        rows.append(result)

    _write_jsonl(output_jsonl, rows)
    successes = sum(1 for row in rows if row.get("success") is True)
    selected_correct = sum(
        1 for row in rows if row.get("patch_candidate_selected_correctly") is True
    )
    qwen_calls = sum(1 for row in rows if row.get("qwen_called") is True)
    return {
        "artifact_family": "phase2aw_package_loaded_descriptor_execution_runner",
        "rows": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "correct_patch_candidate_selections": selected_correct,
        "patch_candidate_selection_accuracy": selected_correct / len(rows) if rows else 0.0,
        "qwen_called_rows": qwen_calls,
        "policy_loaded": True,
        "output_jsonl": str(Path(output_jsonl)),
        "artifact_root": str(artifacts),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase2AW package-loaded descriptor-selected bounded execution."
    )
    parser.add_argument("--runtime-tasks-jsonl", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--test-python")
    args = parser.parse_args()
    report = run_phase2aw_package_loaded_descriptor_execution(
        runtime_tasks_jsonl=args.runtime_tasks_jsonl,
        dataset_root=args.dataset_root,
        clone_root=args.clone_root,
        package_path=args.package_path,
        output_jsonl=args.output_jsonl,
        artifact_root=args.artifact_root,
        max_rows=args.max_rows,
        timeout_seconds=args.timeout_seconds,
        test_python=args.test_python,
    )
    _write_json(args.summary_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["rows"] <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
