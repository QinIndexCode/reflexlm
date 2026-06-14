from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric(report: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = _dict(report.get("metrics")).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def audit_phase2at_runtime_delta_gate(
    *,
    full_runtime_audit_json: str | Path,
    control_runtime_audit_json: str | Path,
    min_rows: int = 256,
    min_full_success_rate: float = 0.85,
    min_full_minus_control: float = 0.15,
) -> dict[str, Any]:
    full = _read_json(full_runtime_audit_json)
    control = _read_json(control_runtime_audit_json)
    full_checks = _dict(full.get("checks"))
    control_checks = _dict(control.get("checks"))
    full_rows = int(_metric(full, "row_count"))
    control_rows = int(_metric(control, "row_count"))
    full_success = _metric(full, "success_rate")
    control_success = _metric(control, "success_rate")
    delta = full_success - control_success
    full_boundary = full.get("claim_boundary")
    control_boundary = control.get("claim_boundary")
    checks = {
        "full_runtime_audit_passed": full.get("passed") is True,
        "full_policy_loaded": full_checks.get("all_rows_policy_loaded") is True,
        "control_policy_not_loaded": control_checks.get("all_rows_policy_loaded") is False,
        "same_execution_boundary": bool(full_boundary)
        and full_boundary == control_boundary,
        "full_row_minimum_met": full_rows >= min_rows,
        "control_row_minimum_met": control_rows >= min_rows,
        "full_success_rate_minimum_met": full_success >= min_full_success_rate,
        "full_minus_control_delta_met": delta >= min_full_minus_control,
        "sealed_feedback_absent": full_checks.get("sealed_feedback_absent") is True
        and control_checks.get("sealed_feedback_absent") is True,
        "freeform_and_openended_blocked": {
            "do_not_claim_freeform_model_generated_patch_repair",
            "do_not_claim_open_ended_debugging_generalization",
        }.issubset(set(full.get("blocked_actions") or [])),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2at_package_runtime_delta_gate",
        "passed": passed,
        "claim_boundary": (
            "This gate tests whether a loaded NativeNervousPolicyPackage adds runtime "
            "repair success beyond the same bounded symbolic structural runner without "
            "loading policy. Passing would support only a bounded package-runtime delta; "
            "it would still not prove freeform patch generation, sealed transfer, "
            "production autonomy, open-ended debugging, or an epoch-making architecture."
        ),
        "checks": checks,
        "metrics": {
            "full_rows": full_rows,
            "control_rows": control_rows,
            "full_success_rate": full_success,
            "control_success_rate": control_success,
            "full_minus_control_success_rate": delta,
            "full_failure_reasons": _dict(_dict(full.get("metrics")).get("failure_reasons")),
            "control_failure_reasons": _dict(
                _dict(control.get("metrics")).get("failure_reasons")
            ),
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_full_success_rate": min_full_success_rate,
            "min_full_minus_control": min_full_minus_control,
        },
        "supported_claims": (
            ["phase2at_package_runtime_delta_supported"] if passed else []
        ),
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_claim_package_runtime_mechanism_delta",
            "do_not_release_phase2at_as_claim_bearing_package_runtime_evidence",
            "design_next_nonsealed_task_where_no_policy_control_cannot_solve",
        ],
        "inputs": {
            "full_runtime_audit_json": str(Path(full_runtime_audit_json)),
            "control_runtime_audit_json": str(Path(control_runtime_audit_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AT package runtime delta gate.")
    parser.add_argument("--full-runtime-audit-json", required=True)
    parser.add_argument("--control-runtime-audit-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=256)
    parser.add_argument("--min-full-success-rate", type=float, default=0.85)
    parser.add_argument("--min-full-minus-control", type=float, default=0.15)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2at_runtime_delta_gate(
        full_runtime_audit_json=args.full_runtime_audit_json,
        control_runtime_audit_json=args.control_runtime_audit_json,
        min_rows=args.min_rows,
        min_full_success_rate=args.min_full_success_rate,
        min_full_minus_control=args.min_full_minus_control,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
