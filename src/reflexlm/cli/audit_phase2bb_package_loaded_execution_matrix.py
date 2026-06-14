from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARTIFACT_FAMILY = "phase2bb_clone_present_package_loaded_execution_matrix_audit"


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def audit_phase2bb_package_loaded_execution_matrix(
    *,
    subset_report_json: str | Path,
    package_gate_json: str | Path,
    package_execution_summary_json: str | Path,
    wrong_cache_summary_json: str | Path,
    package_execution_jsonl: str | Path | None = None,
    output_json: str | Path | None = None,
    min_rows: int = 10,
    min_repos: int = 5,
    min_slot_selection_accuracy: float = 0.85,
    min_execution_success_rate: float = 0.75,
) -> dict[str, Any]:
    subset = _read_json(subset_report_json)
    package_gate = _read_json(package_gate_json)
    package_execution = _read_json(package_execution_summary_json)
    wrong_cache = _read_json(wrong_cache_summary_json)
    execution_rows = _read_jsonl(package_execution_jsonl)

    rows = _int(package_execution.get("rows"))
    subset_head_rows = _int(subset.get("head_rows"))
    subset_task_rows = _int(subset.get("task_rows"))
    repo_count = _int(subset.get("repo_count"))
    slot_accuracy = _float(package_execution.get("slot_selection_accuracy"))
    success_rate = _float(package_execution.get("success_rate"))
    wrong_success_rate = _float(wrong_cache.get("success_rate"))
    subset_inputs = subset.get("inputs") if isinstance(subset.get("inputs"), dict) else {}
    slot_counts = subset.get("slot_counts") if isinstance(subset.get("slot_counts"), dict) else {}
    residual_rows = [row for row in execution_rows if row.get("success") is not True]
    package_model_load_strategy = str(
        package_execution.get("package_model_load_strategy") or ""
    )
    package_offload_state_dict = package_execution.get("package_offload_state_dict")
    test_python_profiles = (
        package_execution.get("test_python_profiles")
        if isinstance(package_execution.get("test_python_profiles"), dict)
        else {}
    )
    residual_failure_counts: dict[str, int] = {}
    for row in residual_rows:
        failure = str(
            row.get("phase2z_symbolic_patch_failure")
            or row.get("stop_condition")
            or "unknown"
        )
        residual_failure_counts[failure] = residual_failure_counts.get(failure, 0) + 1

    checks = {
        "subset_report_passed": subset.get("passed") is True,
        "subset_clone_present_filter_enforced": subset_inputs.get("require_clone_present")
        is True
        and bool(subset_inputs.get("clone_root")),
        "subset_min_repos_met": repo_count >= min_repos,
        "subset_min_rows_met": subset_head_rows >= min_rows and subset_task_rows >= min_rows,
        "subset_tasks_match_head_rows": subset_head_rows == subset_task_rows,
        "execution_rows_match_subset_rows": rows == subset_head_rows,
        "execution_jsonl_rows_match_summary": not execution_rows
        or len(execution_rows) == rows,
        "residual_diagnostics_present": not execution_rows
        or all(
            row.get("phase2z_symbolic_patch_failure") or row.get("stop_condition")
            for row in residual_rows
        ),
        "slot0_and_slot1_present": _int(slot_counts.get("0")) > 0
        and _int(slot_counts.get("1")) > 0,
        "package_gate_passed": package_gate.get("passed") is True
        and package_gate.get("ready_for_phase2az_packaged_adapter_runtime_smoke") is True,
        "package_gate_blocks_epoch_claim": package_gate.get(
            "ready_for_epoch_making_architecture_claim"
        )
        is False,
        "execution_policy_is_package_loaded_native_head": package_execution.get(
            "selection_policy"
        )
        == "package_loaded_native_head",
        "slot_selection_accuracy_gate": isinstance(slot_accuracy, float)
        and slot_accuracy >= min_slot_selection_accuracy,
        "execution_success_rate_gate": isinstance(success_rate, float)
        and success_rate >= min_execution_success_rate,
        "package_policy_loaded_for_all_rows": _int(
            package_execution.get("package_policy_loaded_rows")
        )
        == rows,
        "package_load_config_recorded_for_all_rows": not execution_rows
        or (
            bool(package_model_load_strategy)
            and isinstance(package_offload_state_dict, bool)
            and all(
                isinstance(row.get("package_policy_metadata"), dict)
                and row["package_policy_metadata"].get("model_load_strategy")
                == package_model_load_strategy
                and row["package_policy_metadata"].get("offload_state_dict")
                is package_offload_state_dict
                for row in execution_rows
            )
        ),
        "test_runtime_profile_recorded_for_all_rows": not execution_rows
        or (
            sum(_int(value) for value in test_python_profiles.values()) == rows
            and all(
                bool(row.get("test_python"))
                and bool(row.get("test_python_source"))
                for row in execution_rows
            )
        ),
        "phase2ax_head_record_visible_state_for_all_rows": _int(
            package_execution.get("package_head_record_visible_state_rows")
        )
        == rows,
        "package_qwen_called_for_all_rows": _int(
            package_execution.get("package_qwen_called_rows")
        )
        == rows,
        "package_open_repair_authorized_for_attempts": _int(
            package_execution.get("package_open_repair_authorized_rows")
        )
        >= _int(package_execution.get("execution_attempts")),
        "wrong_cache_control_blocks_execution": wrong_cache.get("selection_policy")
        == "wrong_cache"
        and _int(wrong_cache.get("execution_attempts")) == 0
        and wrong_success_rate == 0.0,
        "no_recorded_patch_as_generated_evidence": _int(
            package_execution.get("recorded_patch_artifact_used_rows")
        )
        == 0,
        "recorded_patch_only_for_fault_injection": _int(
            package_execution.get("recorded_patch_artifact_used_for_fault_injection_rows")
        )
        == _int(package_execution.get("execution_attempts")),
        "claim_bearing_execution_for_attempts": _int(
            package_execution.get("claim_bearing_execution_evidence_rows")
        )
        == _int(package_execution.get("execution_attempts")),
        "no_freeform_patch_generation": _int(
            package_execution.get("freeform_patch_generation_rows")
        )
        == 0,
        "no_sealed_feedback": _int(package_execution.get("sealed_feedback_used_rows")) == 0,
        "model_prediction_json_not_used_as_selector": _int(
            package_execution.get("model_prediction_records_present_rows")
        )
        == 0,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": ARTIFACT_FAMILY,
        "passed": passed,
        "ready_for_phase2bb_clone_present_package_loaded_runtime_matrix": passed,
        "ready_for_phase2ax_package": False,
        "ready_for_package_or_execution_claim": False,
        "ready_for_sealed_eval": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "rows": rows,
            "subset_head_rows": subset_head_rows,
            "subset_task_rows": subset_task_rows,
            "repo_count": repo_count,
            "slot_counts": slot_counts,
            "slot_selection_accuracy": slot_accuracy,
            "execution_attempts": package_execution.get("execution_attempts"),
            "success_rate": success_rate,
            "attempt_success_rate": package_execution.get("attempt_success_rate"),
            "wrong_cache_success_rate": wrong_success_rate,
            "residual_rows": len(residual_rows),
            "residual_failure_counts": residual_failure_counts,
            "package_policy_loaded_rows": package_execution.get(
                "package_policy_loaded_rows"
            ),
            "package_model_load_strategy": package_model_load_strategy or None,
            "package_offload_state_dict": package_offload_state_dict,
            "test_python_profiles": test_python_profiles,
            "package_head_record_visible_state_rows": package_execution.get(
                "package_head_record_visible_state_rows"
            ),
            "package_qwen_called_rows": package_execution.get("package_qwen_called_rows"),
            "clone_filtered_repos": subset.get("clone_filtered_repos", []),
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_repos": min_repos,
            "min_slot_selection_accuracy": min_slot_selection_accuracy,
            "min_execution_success_rate": min_execution_success_rate,
        },
        "claim_boundary": (
            "phase2bb_clone_present_package_loaded_runtime_matrix_not_sealed_or_epoch_claim"
            if passed
            else "phase2bb_clone_present_package_loaded_runtime_matrix_failed_or_incomplete"
        ),
        "supported_claims": [
            "phase2bb_package_loaded_native_head_selects_bounded_repair_slots_on_clone_present_public_repos_and_enables_runtime_execution"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "phase2ax_full_package_claim",
            "sealed_cross_model_transfer",
            "freeform_patch_generation",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": [
            "do_not_claim_sealed_transfer_from_nonsealed_package_loaded_matrix",
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
        "next_required_experiment": (
            "repair_phase2bb_package_loaded_runtime_matrix"
            if not passed
            else (
                "phase2bc_repair_or_explain_residual_runtime_failures_and_expand_clone_present_matrix"
                if residual_rows
                else "phase2bc_expand_clone_present_matrix_then_run_sealed_package_loaded_transfer"
            )
        ),
        "inputs": {
            "subset_report_json": str(Path(subset_report_json)),
            "package_gate_json": str(Path(package_gate_json)),
            "package_execution_summary_json": str(Path(package_execution_summary_json)),
            "package_execution_jsonl": str(Path(package_execution_jsonl))
            if package_execution_jsonl is not None
            else None,
            "wrong_cache_summary_json": str(Path(wrong_cache_summary_json)),
        },
        "notes": [
            "This audit binds subset construction to clone-present public repos before evaluating package-loaded execution.",
            "Execution remains bounded to fixed candidate actions and runtime-symbolic structural repair.",
            "Passing this audit is still nonsealed and does not prove production autonomy or epoch-making architecture.",
        ],
    }
    if output_json is not None:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2BB clone-present package-loaded execution matrix."
    )
    parser.add_argument("--subset-report-json", required=True)
    parser.add_argument("--package-gate-json", required=True)
    parser.add_argument("--package-execution-summary-json", required=True)
    parser.add_argument("--package-execution-jsonl")
    parser.add_argument("--wrong-cache-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--min-repos", type=int, default=5)
    parser.add_argument("--min-slot-selection-accuracy", type=float, default=0.85)
    parser.add_argument("--min-execution-success-rate", type=float, default=0.75)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2bb_package_loaded_execution_matrix(
        subset_report_json=args.subset_report_json,
        package_gate_json=args.package_gate_json,
        package_execution_summary_json=args.package_execution_summary_json,
        package_execution_jsonl=args.package_execution_jsonl,
        wrong_cache_summary_json=args.wrong_cache_summary_json,
        output_json=args.output_json,
        min_rows=args.min_rows,
        min_repos=args.min_repos,
        min_slot_selection_accuracy=args.min_slot_selection_accuracy,
        min_execution_success_rate=args.min_execution_success_rate,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
