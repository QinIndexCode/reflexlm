from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric(report: dict[str, Any], name: str) -> Any:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    return metrics.get(name)


def _breakdown(report: dict[str, Any], name: str) -> dict[str, int]:
    breakdown = (
        report.get("failure_breakdown")
        if isinstance(report.get("failure_breakdown"), dict)
        else {}
    )
    value = breakdown.get(name)
    return value if isinstance(value, dict) else {}


def audit_phase2aw_verified_candidate_pool(
    *,
    execution_gate_json: str | Path,
    failure_audit_json: str | Path,
    min_full_success_rate: float = 0.85,
    min_control_success_rate: float = 0.20,
    max_control_success_rate: float = 0.75,
    min_full_minus_control_success: float = 0.15,
) -> dict[str, Any]:
    gate = _read_json(execution_gate_json)
    failure = _read_json(failure_audit_json)
    full_success = _metric(failure, "full_success_rate")
    control_success = _metric(failure, "control_success_rate")
    delta = _metric(failure, "full_minus_control_success_rate")
    source_splits = _breakdown(failure, "source_artifact_split_counts")
    split_clean = set(source_splits) <= {"holdout"}
    checks = {
        "execution_gate_supplied": bool(gate),
        "failure_audit_supplied": bool(failure),
        "bounded_execution_contract_preserved": gate.get("checks", {}).get(
            "bounded_symbolic_execution_only"
        )
        is True,
        "execution_safety_met": gate.get("checks", {}).get("execution_safety_met")
        is True,
        "full_success_rate_gate": isinstance(full_success, (int, float))
        and float(full_success) >= min_full_success_rate,
        "control_success_nonzero": isinstance(control_success, (int, float))
        and float(control_success) >= min_control_success_rate,
        "control_success_not_ceiling": isinstance(control_success, (int, float))
        and float(control_success) <= max_control_success_rate,
        "full_minus_control_success_delta": isinstance(delta, (int, float))
        and float(delta) >= min_full_minus_control_success,
        "source_artifact_split_clean": split_clean,
        "selection_not_primary_bottleneck": failure.get("checks", {}).get(
            "selection_is_not_primary_bottleneck"
        )
        is True,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2aw_verified_candidate_pool_gate",
        "passed": passed,
        "ready_for_phase2aw_package_or_successor_training": passed,
        "claim_boundary": (
            "Phase2AW gate only authorizes continuing to a split-clean verified "
            "candidate-pool/runtime-hardening stage. It does not prove learned "
            "freeform patch generation, sealed transfer, production autonomy, "
            "open-ended debugging generalization, or an epoch-making architecture."
        ),
        "checks": checks,
        "metrics": {
            "full_success_rate": full_success,
            "control_success_rate": control_success,
            "full_minus_control_success_rate": delta,
            "source_artifact_split_counts": source_splits,
            "thresholds": {
                "min_full_success_rate": min_full_success_rate,
                "min_control_success_rate": min_control_success_rate,
                "max_control_success_rate": max_control_success_rate,
                "min_full_minus_control_success": min_full_minus_control_success,
            },
        },
        "blocking_reasons": [name for name, ok in checks.items() if ok is False],
        "supported_claims": [
            "phase2aw_split_clean_verified_candidate_pool_ready"
        ]
        if passed
        else [],
        "unsupported_claims": (
            ["phase2av_or_phase2aw_package_ready"] if not passed else []
        )
        + [
            "sealed_cross_model_transfer",
            "learned_freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_package_phase2av_or_phase2aw",
            "do_not_run_sealed_eval",
            "rebuild_split_clean_verified_candidate_pool_before_training_or_package",
        ],
        "inputs": {
            "execution_gate_json": str(Path(execution_gate_json)),
            "failure_audit_json": str(Path(failure_audit_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AW verified candidate pool.")
    parser.add_argument("--execution-gate-json", required=True)
    parser.add_argument("--failure-audit-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2aw_verified_candidate_pool(
        execution_gate_json=args.execution_gate_json,
        failure_audit_json=args.failure_audit_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
