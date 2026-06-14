from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARTIFACT_FAMILY = "phase2az_model_predicted_execution_matrix_audit"


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


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


def _source_overlap_accuracy(eval_report: dict[str, Any], split: str) -> float | None:
    baseline = eval_report.get("source_overlap_command_slot_baseline")
    if not isinstance(baseline, dict):
        return None
    split_report = baseline.get(split)
    if not isinstance(split_report, dict):
        return None
    return _float(split_report.get("accuracy"))


def audit_phase2az_model_predicted_execution_matrix(
    *,
    subset_report_json: str | Path,
    model_eval_json: str | Path,
    model_execution_summary_json: str | Path,
    wrong_cache_summary_json: str | Path,
    phase2ay_model_audit_json: str | Path,
    eval_split: str,
    output_json: str | Path | None = None,
    min_repos: int = 3,
    min_rows: int = 6,
    min_model_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.25,
    min_execution_success_rate: float = 0.75,
) -> dict[str, Any]:
    subset = _read_json(subset_report_json)
    model_eval = _read_json(model_eval_json)
    execution = _read_json(model_execution_summary_json)
    wrong_cache = _read_json(wrong_cache_summary_json)
    phase2ay = _read_json(phase2ay_model_audit_json)
    eval_metrics = model_eval.get("eval_metrics") if isinstance(model_eval.get("eval_metrics"), dict) else {}
    model_slot_accuracy = _float(eval_metrics.get("command_slot_accuracy"))
    source_overlap_accuracy = _source_overlap_accuracy(model_eval, eval_split)
    model_minus_source = (
        model_slot_accuracy - source_overlap_accuracy
        if isinstance(model_slot_accuracy, float) and isinstance(source_overlap_accuracy, float)
        else None
    )
    execution_rows = int(execution.get("rows") or 0)
    execution_success_rate = _float(execution.get("success_rate"))
    wrong_success_rate = _float(wrong_cache.get("success_rate"))
    checks = {
        "phase2ay_model_audit_passed": phase2ay.get("passed") is True
        and phase2ay.get("ready_for_phase2ay_model_prediction_execution_eval") is True,
        "subset_passed": subset.get("passed") is True,
        "min_repos_met": int(subset.get("repo_count") or 0) >= min_repos,
        "min_rows_met": int(subset.get("head_rows") or 0) >= min_rows,
        "slot0_and_slot1_present": (subset.get("slot_counts") or {}).get("0", 0) > 0
        and (subset.get("slot_counts") or {}).get("1", 0) > 0,
        "model_eval_records_cover_subset": int(model_eval.get("eval_examples") or 0)
        == int(subset.get("head_rows") or -1),
        "model_slot_accuracy_gate": isinstance(model_slot_accuracy, float)
        and model_slot_accuracy >= min_model_slot_accuracy,
        "source_overlap_recorded": isinstance(source_overlap_accuracy, float),
        "model_beats_source_overlap": isinstance(model_minus_source, float)
        and model_minus_source >= min_model_minus_source_overlap,
        "execution_policy_is_model_prediction": execution.get("selection_policy")
        == "model_prediction_records",
        "execution_rows_match_subset": execution_rows == int(subset.get("task_rows") or -1),
        "execution_success_rate_gate": isinstance(execution_success_rate, float)
        and execution_success_rate >= min_execution_success_rate,
        "model_prediction_records_cover_execution": int(
            execution.get("model_prediction_records_present_rows") or 0
        )
        == execution_rows,
        "wrong_cache_control_blocks_execution": wrong_cache.get("selection_policy") == "wrong_cache"
        and int(wrong_cache.get("execution_attempts") or 0) == 0
        and wrong_success_rate == 0.0,
        "no_recorded_patch_as_generated_evidence": int(
            execution.get("recorded_patch_artifact_used_rows") or 0
        )
        == 0,
        "recorded_patch_only_for_fault_injection": int(
            execution.get("recorded_patch_artifact_used_for_fault_injection_rows") or 0
        )
        == int(execution.get("execution_attempts") or -1),
        "no_freeform_patch_generation": int(execution.get("freeform_patch_generation_rows") or 0)
        == 0,
        "no_sealed_feedback": int(execution.get("sealed_feedback_used_rows") or 0) == 0,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": ARTIFACT_FAMILY,
        "passed": passed,
        "ready_for_phase2az_package_gate": passed,
        "ready_for_phase2ax_package": False,
        "ready_for_package_or_execution_claim": False,
        "ready_for_sealed_eval": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "repo_count": subset.get("repo_count"),
            "repos": subset.get("repos"),
            "head_rows": subset.get("head_rows"),
            "task_rows": subset.get("task_rows"),
            "slot_counts": subset.get("slot_counts"),
            "model_slot_accuracy": model_slot_accuracy,
            "source_overlap_accuracy": source_overlap_accuracy,
            "model_minus_source_overlap": model_minus_source,
            "execution_rows": execution_rows,
            "execution_success_rate": execution_success_rate,
            "wrong_cache_success_rate": wrong_success_rate,
        },
        "claim_boundary": (
            "phase2az_repo_diverse_model_predicted_runtime_execution_matrix_not_package_or_epoch_claim"
            if passed
            else "phase2az_model_predicted_execution_matrix_failed_or_incomplete_not_claim_evidence"
        ),
        "next_required_experiment": (
            "phase2az_package_gate_with_packaged_phase2ax_adapter_runtime"
            if passed
            else "repair_phase2az_matrix_before_package_gate"
        ),
        "blocked_actions": [
            "do_not_claim_phase2ax_package_without_packaged_adapter_runtime_gate",
            "do_not_run_sealed_v3_from_phase2az_matrix_only",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
        "thresholds": {
            "min_repos": min_repos,
            "min_rows": min_rows,
            "min_model_slot_accuracy": min_model_slot_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_execution_success_rate": min_execution_success_rate,
        },
        "inputs": {
            "subset_report_json": str(Path(subset_report_json)),
            "model_eval_json": str(Path(model_eval_json)),
            "model_execution_summary_json": str(Path(model_execution_summary_json)),
            "wrong_cache_summary_json": str(Path(wrong_cache_summary_json)),
            "phase2ay_model_audit_json": str(Path(phase2ay_model_audit_json)),
            "eval_split": eval_split,
        },
        "notes": [
            "This matrix expands model-predicted slot execution across multiple validation repos.",
            "It remains nonsealed and nonpackaged; package readiness requires a packaged Phase2AX adapter gate.",
            "Recorded patch artifacts are allowed only for fault injection, not as generated repair evidence.",
        ],
    }
    if output_json is not None:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AZ model-predicted execution matrix.")
    parser.add_argument("--subset-report-json", required=True)
    parser.add_argument("--model-eval-json", required=True)
    parser.add_argument("--model-execution-summary-json", required=True)
    parser.add_argument("--wrong-cache-summary-json", required=True)
    parser.add_argument("--phase2ay-model-audit-json", required=True)
    parser.add_argument("--eval-split", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-repos", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=6)
    parser.add_argument("--min-model-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.25)
    parser.add_argument("--min-execution-success-rate", type=float, default=0.75)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2az_model_predicted_execution_matrix(
        subset_report_json=args.subset_report_json,
        model_eval_json=args.model_eval_json,
        model_execution_summary_json=args.model_execution_summary_json,
        wrong_cache_summary_json=args.wrong_cache_summary_json,
        phase2ay_model_audit_json=args.phase2ay_model_audit_json,
        eval_split=args.eval_split,
        output_json=args.output_json,
        min_repos=args.min_repos,
        min_rows=args.min_rows,
        min_model_slot_accuracy=args.min_model_slot_accuracy,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_execution_success_rate=args.min_execution_success_rate,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
