from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PHASE2AU_RUNTIME_BOUNDARY = "phase2au_policy_required_runtime_delta_control"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _metric(report: dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = _dict(report.get("metrics"))
    if key in metrics:
        return _number(metrics.get(key), default)
    return _number(report.get(key), default)


def _rows(report: dict[str, Any]) -> int:
    metrics = _dict(report.get("metrics"))
    for key in ("row_count", "rows", "total"):
        if key in metrics:
            return int(_number(metrics.get(key), 0.0))
        if key in report:
            return int(_number(report.get(key), 0.0))
    return 0


def _success_rate(report: dict[str, Any]) -> float:
    explicit = _metric(report, "success_rate", -1.0)
    if explicit >= 0.0:
        return explicit
    rows = _rows(report)
    successes = _metric(report, "successes", 0.0)
    return successes / rows if rows else 0.0


def _policy_loaded(report: dict[str, Any]) -> bool | None:
    checks = _dict(report.get("checks"))
    if "all_rows_policy_loaded" in checks:
        return checks.get("all_rows_policy_loaded") is True
    if "policy_loaded" in report:
        return report.get("policy_loaded") is True
    return None


def _sealed_absent(report: dict[str, Any]) -> bool:
    checks = _dict(report.get("checks"))
    if "sealed_feedback_absent" in checks:
        return checks.get("sealed_feedback_absent") is True
    if "sealed_feedback_used" in report:
        return report.get("sealed_feedback_used") is False
    return False


def _boundary(report: dict[str, Any]) -> str:
    return str(report.get("claim_boundary") or "")


def _freeform_blocked(report: dict[str, Any]) -> bool:
    blocked = set(str(item) for item in (report.get("blocked_actions") or []))
    return {
        "do_not_claim_freeform_patch_generation",
        "do_not_claim_open_ended_debugging_generalization",
    }.issubset(blocked) or {
        "do_not_claim_freeform_model_generated_patch_repair",
        "do_not_claim_open_ended_debugging_generalization",
    }.issubset(blocked)


def audit_phase2au_runtime_delta_gate(
    *,
    full_runtime_audit_json: str | Path,
    control_runtime_audit_json: str | Path,
    package_gate_json: str | Path,
    min_rows: int = 20,
    min_full_success_rate: float = 0.85,
    min_full_minus_control: float = 0.15,
) -> dict[str, Any]:
    full = _read_json(full_runtime_audit_json)
    control = _read_json(control_runtime_audit_json)
    package_gate = _read_json(package_gate_json)
    full_rows = _rows(full)
    control_rows = _rows(control)
    full_success = _success_rate(full)
    control_success = _success_rate(control)
    delta = full_success - control_success
    full_boundary = _boundary(full)
    control_boundary = _boundary(control)
    checks = {
        "package_gate_passed": package_gate.get("passed") is True
        and package_gate.get("ready_for_phase2au_runtime_delta_eval") is True,
        "full_runtime_audit_passed": full.get("passed") is True,
        "full_policy_loaded": _policy_loaded(full) is True,
        "control_policy_not_loaded": _policy_loaded(control) is False,
        "same_phase2au_execution_boundary": full_boundary == control_boundary
        and full_boundary == PHASE2AU_RUNTIME_BOUNDARY,
        "full_row_minimum_met": full_rows >= min_rows,
        "control_row_minimum_met": control_rows >= min_rows,
        "full_success_rate_minimum_met": full_success >= min_full_success_rate,
        "full_minus_control_delta_met": delta >= min_full_minus_control,
        "sealed_feedback_absent": _sealed_absent(full) and _sealed_absent(control),
        "freeform_and_openended_blocked": _freeform_blocked(full),
        "not_head_eval_or_smoke_postflight": full.get("artifact_family")
        not in {
            "phase2au_policy_required_eval_postflight",
            "phase2au_policy_required_smoke_postflight",
        },
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2au_policy_required_runtime_delta_gate",
        "passed": passed,
        "claim_boundary": (
            "This gate tests only bounded Phase2AU package/no-policy runtime delta "
            "on policy-required tasks. Passing does not prove sealed transfer, "
            "freeform patch generation, production autonomy, open-ended debugging, "
            "or an epoch-making architecture."
        ),
        "checks": checks,
        "metrics": {
            "full_rows": full_rows,
            "control_rows": control_rows,
            "full_success_rate": full_success,
            "control_success_rate": control_success,
            "full_minus_control_success_rate": delta,
            "full_boundary": full_boundary,
            "control_boundary": control_boundary,
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
        "supported_claims": ["phase2au_bounded_package_runtime_delta_supported"]
        if passed
        else [],
        "unsupported_claims": [
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_claim_phase2au_package_runtime_delta",
            "do_not_release_phase2au_as_claim_bearing_runtime_evidence",
            "run_or_fix_real_full_vs_no_policy_runtime_control_before_claim_upgrade",
        ],
        "inputs": {
            "full_runtime_audit_json": str(Path(full_runtime_audit_json)),
            "control_runtime_audit_json": str(Path(control_runtime_audit_json)),
            "package_gate_json": str(Path(package_gate_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AU runtime delta gate.")
    parser.add_argument("--full-runtime-audit-json", required=True)
    parser.add_argument("--control-runtime-audit-json", required=True)
    parser.add_argument("--package-gate-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--min-full-success-rate", type=float, default=0.85)
    parser.add_argument("--min-full-minus-control", type=float, default=0.15)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2au_runtime_delta_gate(
        full_runtime_audit_json=args.full_runtime_audit_json,
        control_runtime_audit_json=args.control_runtime_audit_json,
        package_gate_json=args.package_gate_json,
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
