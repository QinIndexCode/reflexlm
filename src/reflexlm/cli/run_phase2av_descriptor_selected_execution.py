from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from reflexlm.cli.run_phase2z_public_structural_repair_execution import (
    run_phase2z_public_structural_repair_execution,
)


CLAIM_BOUNDARY = (
    "phase2av_descriptor_selected_bounded_symbolic_execution_not_freeform_patch_generation"
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
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


def _default_manifest(path: Path) -> Path:
    manifest = {
        "policy_label": "phase2av_descriptor_selected_execution_manifest_only",
        "open_repair_capabilities": {
            "patch_proposal": True,
            "bounded_edit_scope": True,
            "test_selection": True,
            "rollback_safety": True,
            "progress_monitor": True,
            "verification_state": True,
            "stop_condition": True,
        },
    }
    _write_json(path, manifest)
    return path


def _prediction_records(eval_summary: dict[str, Any]) -> list[dict[str, Any]]:
    records = eval_summary.get("prediction_records")
    return records if isinstance(records, list) else []


def _selection_for(record: dict[str, Any], mode: str) -> int:
    key = {
        "adapter": "command_slot_prediction",
        "source_overlap": "source_overlap_prediction",
        "gold_oracle": "command_slot_label",
    }.get(mode)
    if key is None:
        return 0
    value = record.get(key)
    return int(value) if isinstance(value, int) else 0


def _candidate_actions(row: dict[str, Any]) -> list[str]:
    candidates = row.get("repair_candidates") if isinstance(row.get("repair_candidates"), list) else []
    return [
        str(candidate.get("repair_action") or "")
        for candidate in candidates
        if isinstance(candidate, dict)
    ]


def _repo_id_from_origin(origin: str) -> str:
    parsed = urlparse(origin)
    path = parsed.path if parsed.scheme else origin
    normalized = path.strip("/").removesuffix(".git").replace("/", "_").replace("-", "_")
    return normalized


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
        "test_target": str(row.get("runtime_visible_evidence", {}).get("failing_test_target") or test_rel)
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


def _base_result_row(
    *,
    row: dict[str, Any],
    record: dict[str, Any],
    selected_slot: int,
    expected_slot: int,
    selection_mode: str,
    row_index: int,
) -> dict[str, Any]:
    actions = _candidate_actions(row)
    selected_action = actions[selected_slot] if 0 <= selected_slot < len(actions) else ""
    expected_action = actions[expected_slot] if 0 <= expected_slot < len(actions) else ""
    return {
        "trace_id": row.get("task_id") or record.get("episode_id") or f"row-{row_index}",
        "example_id": record.get("example_id"),
        "source_kind": row.get("source_kind"),
        "repo_origin": row.get("repo_origin"),
        "repo_commit": row.get("repo_commit"),
        "result_source": "phase2av_descriptor_selected_execution",
        "selection_mode": selection_mode,
        "native_policy_label": str(record.get("adapter_output_dir") or "phase2av_adapter_eval"),
        "policy_loaded": selection_mode == "adapter",
        "selected_patch_candidate_slot": selected_slot,
        "expected_patch_candidate_slot": expected_slot,
        "selected_repair_action": selected_action,
        "expected_repair_action": expected_action,
        "patch_candidate_selected_correctly": selected_slot == expected_slot,
        "patch_source": "selected_bounded_descriptor_candidate",
        "patch_generator": "bounded_symbolic_structural_patch_v1",
        "recorded_patch_artifact_used": False,
        "recorded_patch_artifact_used_for_fault_injection": True,
        "oracle_trace_used": False,
        "sealed_feedback_used": False,
        "claim_bearing_freeform_patch_evidence": False,
        "freeform_patch_generation": False,
        "low_level_qwen_calls": 0,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def run_phase2av_descriptor_selected_execution(
    *,
    runtime_tasks_jsonl: str | Path,
    eval_summary_json: str | Path,
    dataset_root: str | Path,
    clone_root: str | Path,
    output_jsonl: str | Path,
    artifact_root: str | Path,
    package_manifest_json: str | Path | None = None,
    selection_mode: str = "adapter",
    max_rows: int = 20,
    timeout_seconds: int = 30,
    test_python: str | None = None,
) -> dict[str, Any]:
    if selection_mode not in {"adapter", "source_overlap", "fixed0", "gold_oracle"}:
        raise ValueError("selection_mode must be adapter, source_overlap, fixed0, or gold_oracle")
    tasks = _read_jsonl(runtime_tasks_jsonl)[:max_rows]
    eval_summary = _read_json(eval_summary_json)
    records = _prediction_records(eval_summary)[:max_rows]
    if len(tasks) != len(records):
        raise ValueError(
            f"runtime task/prediction count mismatch for selected slice: {len(tasks)} != {len(records)}"
        )

    artifacts = Path(artifact_root)
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(package_manifest_json)
        if package_manifest_json
        else _default_manifest(artifacts / "phase2av_descriptor_execution_manifest.json")
    )
    rows: list[dict[str, Any]] = []
    for index, (task, record) in enumerate(zip(tasks, records)):
        expected_slot = int(record.get("command_slot_label") or 0)
        selected_slot = _selection_for(record, selection_mode)
        result = _base_result_row(
            row=task,
            record=record,
            selected_slot=selected_slot,
            expected_slot=expected_slot,
            selection_mode=selection_mode,
            row_index=index,
        )
        if selected_slot != expected_slot:
            result.update(
                {
                    "success": False,
                    "full_task_success": False,
                    "full_patch_correctness": False,
                    "full_test_pass_rate": 0.0,
                    "rollback_failure_restored": True,
                    "unauthorized_write_count": 0,
                    "false_completion": False,
                    "verification_state": "failed",
                    "stop_condition": "candidate_selection_failed_before_patch_application",
                    "execution_skipped_reason": "selected_candidate_does_not_match_runtime_descriptor",
                    "artifact_paths": {},
                }
            )
            rows.append(result)
            continue

        with tempfile.TemporaryDirectory(prefix="phase2av_exec_") as tmp:
            one_row = Path(tmp) / "row.jsonl"
            _write_jsonl(one_row, [_runner_row(task)])
            row_artifact_root = artifacts / f"row_{index:05d}"
            report = run_phase2z_public_structural_repair_execution(
                source_rows_jsonl=one_row,
                dataset_root=dataset_root,
                clone_root=clone_root,
                package_path=manifest_path,
                output_jsonl=Path(tmp) / "execution.jsonl",
                artifact_root=row_artifact_root,
                max_rows=1,
                timeout_seconds=timeout_seconds,
                test_python=test_python,
                load_policy=False,
                patch_mode="runtime_symbolic_structural",
            )
            execution_rows = _read_jsonl(Path(tmp) / "execution.jsonl")
            executed = execution_rows[0] if execution_rows else {}
            result.update(
                {
                    "success": executed.get("success") is True,
                    "full_task_success": executed.get("full_task_success") is True,
                    "full_patch_correctness": executed.get("full_patch_correctness") is True,
                    "full_test_pass_rate": executed.get("full_test_pass_rate", 0.0),
                    "rollback_failure_restored": executed.get("rollback_failure_restored") is True,
                    "unauthorized_write_count": int(executed.get("unauthorized_write_count") or 0),
                    "false_completion": executed.get("false_completion") is True,
                    "verification_state": executed.get("verification_state"),
                    "stop_condition": executed.get("stop_condition"),
                    "phase2z_execution_summary": report,
                    "artifact_paths": executed.get("artifact_paths") or {},
                }
            )
            rows.append(result)

    _write_jsonl(output_jsonl, rows)
    successes = sum(1 for row in rows if row.get("success") is True)
    selected_correct = sum(
        1 for row in rows if row.get("patch_candidate_selected_correctly") is True
    )
    return {
        "artifact_family": "phase2av_descriptor_selected_execution_runner",
        "rows": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "selection_accuracy": selected_correct / len(rows) if rows else 0.0,
        "selection_mode": selection_mode,
        "output_jsonl": str(Path(output_jsonl)),
        "artifact_root": str(artifacts),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase2AV descriptor-selected bounded symbolic execution."
    )
    parser.add_argument("--runtime-tasks-jsonl", required=True)
    parser.add_argument("--eval-summary-json", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--package-manifest-json")
    parser.add_argument(
        "--selection-mode",
        choices=["adapter", "source_overlap", "fixed0", "gold_oracle"],
        default="adapter",
    )
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--test-python")
    args = parser.parse_args()
    report = run_phase2av_descriptor_selected_execution(
        runtime_tasks_jsonl=args.runtime_tasks_jsonl,
        eval_summary_json=args.eval_summary_json,
        dataset_root=args.dataset_root,
        clone_root=args.clone_root,
        output_jsonl=args.output_jsonl,
        artifact_root=args.artifact_root,
        package_manifest_json=args.package_manifest_json,
        selection_mode=args.selection_mode,
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
