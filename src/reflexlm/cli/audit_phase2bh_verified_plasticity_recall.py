from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def audit_phase2bh_verified_plasticity_recall(
    *,
    phase2bg_audit_json: str | Path,
    baseline_summary_json: str | Path,
    learning_summary_json: str | Path,
    recall_summary_json: str | Path,
    wrong_memory_summary_json: str | Path,
    memory_json: str | Path,
    min_rows: int = 12,
    min_recall_delta: float = 0.30,
) -> dict[str, Any]:
    phase2bg = _read(phase2bg_audit_json)
    baseline = _read(baseline_summary_json)
    learning = _read(learning_summary_json)
    recall = _read(recall_summary_json)
    wrong = _read(wrong_memory_summary_json)
    memory = _read(memory_json)
    rows = int(_number(recall, "rows"))
    baseline_success = _number(baseline, "success_rate")
    learning_success = _number(learning, "success_rate")
    recall_success = _number(recall, "success_rate")
    wrong_success = _number(wrong, "success_rate")
    connections = memory.get("connections") if isinstance(memory.get("connections"), dict) else {}
    connection_rows = [
        connection
        for pattern in connections.values()
        if isinstance(pattern, dict)
        for connection in pattern.values()
        if isinstance(connection, dict)
    ]
    checks = {
        "phase2bg_native_structured_receptor_passed": phase2bg.get("passed") is True,
        "row_counts_match_and_meet_minimum": rows >= min_rows
        and all(int(_number(payload, "rows")) == rows for payload in (baseline, learning, wrong)),
        "baseline_and_recall_identity_erased": baseline.get(
            "package_runtime_evidence_control"
        )
        == "identity_erased"
        and recall.get("package_runtime_evidence_control") == "identity_erased"
        and wrong.get("package_runtime_evidence_control") == "identity_erased",
        "runtime_evidence_absent_from_prompts": all(
            _number(payload, "package_runtime_evidence_prompt_present_rows") == 0.0
            for payload in (baseline, learning, recall, wrong)
        ),
        "learning_uses_normal_structured_receptor": learning.get(
            "package_runtime_evidence_control"
        )
        == "normal"
        and _number(learning, "package_structural_probe_receptor_rows") == rows,
        "identity_erased_controls_remove_probe": all(
            _number(payload, "package_structural_probe_receptor_rows") == 0.0
            for payload in (baseline, recall, wrong)
        ),
        "learning_execution_success_gate": learning_success >= 0.85,
        "verified_feedback_accepted_for_all_learning_rows": _number(
            learning, "package_plasticity_feedback_accepted_rows"
        )
        == rows,
        "memory_schema_and_feedback_events": memory.get("schema_version")
        == "reflexlm.synaptic_plasticity.v1"
        and int(_number(memory, "feedback_events")) == rows,
        "memory_connections_are_verified_success_only": len(connection_rows) == rows
        and all(
            int(connection.get("verified_successes", 0)) >= 1
            and int(connection.get("verified_failures", 0)) == 0
            for connection in connection_rows
        ),
        "baseline_is_non_ceiling": baseline_success <= 0.70,
        "recall_success_gate": recall_success >= 0.85,
        "recall_delta_gate": recall_success - baseline_success >= min_recall_delta,
        "recall_memory_hits_all_rows": _number(recall, "package_plasticity_memory_hit_rows")
        == rows,
        "wrong_memory_blocks_execution": wrong_success == 0.0
        and _number(wrong, "execution_attempts") == 0.0
        and wrong.get("package_plasticity_control") == "wrong",
        "real_package_and_qwen_used": all(
            _number(payload, "package_policy_loaded_rows") == rows
            and _number(payload, "package_qwen_called_rows") == rows
            for payload in (baseline, learning, recall, wrong)
        ),
        "no_reference_override_freeform_or_sealed_feedback": all(
            _number(payload, "package_nsi_reference_override_rows") == 0.0
            and _number(payload, "freeform_patch_generation_rows") == 0.0
            and _number(payload, "sealed_feedback_used_rows") == 0.0
            for payload in (baseline, learning, recall, wrong)
        ),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2bh_verified_plasticity_recall",
        "passed": passed,
        "ready_for_bounded_verifier_gated_plasticity_claim": passed,
        "ready_for_repo_disjoint_plasticity_transfer_claim": False,
        "ready_for_differentiable_plasticity_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "rows": rows,
            "baseline_identity_erased_success_rate": baseline_success,
            "learning_success_rate": learning_success,
            "identity_erased_recall_success_rate": recall_success,
            "wrong_memory_success_rate": wrong_success,
            "recall_minus_baseline_success_delta": recall_success - baseline_success,
            "memory_feedback_events": int(_number(memory, "feedback_events")),
            "memory_connection_rows": len(connection_rows),
        },
        "supported_claims": [
            "verified execution feedback creates persistent bounded routine connections that causally restore identity-erased 7B execution"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "repo-disjoint plasticity transfer",
            "differentiable or neural plasticity",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bi_repo_disjoint_temporal_prediction_error_plasticity"
            if passed
            else "repair_phase2bh_verified_plasticity_failure"
        ),
        "inputs": {
            "phase2bg_audit_json": str(Path(phase2bg_audit_json)),
            "baseline_summary_json": str(Path(baseline_summary_json)),
            "learning_summary_json": str(Path(learning_summary_json)),
            "recall_summary_json": str(Path(recall_summary_json)),
            "wrong_memory_summary_json": str(Path(wrong_memory_summary_json)),
            "memory_json": str(Path(memory_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2BH verifier-gated persistent plasticity recall."
    )
    parser.add_argument("--phase2bg-audit-json", required=True)
    parser.add_argument("--baseline-summary-json", required=True)
    parser.add_argument("--learning-summary-json", required=True)
    parser.add_argument("--recall-summary-json", required=True)
    parser.add_argument("--wrong-memory-summary-json", required=True)
    parser.add_argument("--memory-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=12)
    parser.add_argument("--min-recall-delta", type=float, default=0.30)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2bh_verified_plasticity_recall(
        phase2bg_audit_json=args.phase2bg_audit_json,
        baseline_summary_json=args.baseline_summary_json,
        learning_summary_json=args.learning_summary_json,
        recall_summary_json=args.recall_summary_json,
        wrong_memory_summary_json=args.wrong_memory_summary_json,
        memory_json=args.memory_json,
        min_rows=args.min_rows,
        min_recall_delta=args.min_recall_delta,
    )
    _write(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
