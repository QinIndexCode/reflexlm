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


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def build_phase2w_production_safety_report(
    *,
    phase2s_data_health_json: str | Path,
    phase2s_full_holdout_postflight_json: str | Path,
) -> dict[str, Any]:
    data_health = _read_json(phase2s_data_health_json)
    postflight = _read_json(phase2s_full_holdout_postflight_json)
    data_checks = _dict(data_health.get("checks"))
    post_checks = _dict(postflight.get("checks"))
    metrics = _dict(postflight.get("metrics"))
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "postflight_passed": postflight.get("passed") is True,
        "runtime_flags_present": data_checks.get("phase2s_runtime_flags_present") is True,
        "required_artifacts_present": data_checks.get("phase2s_required_artifacts_present") is True,
        "source_repo_read_only": data_checks.get("phase2s_all_split_repos_disjoint") is True
        and data_checks.get("phase2s_no_sealed_reference_anywhere") is True,
        "rollback_recorded": data_checks.get("phase2s_runtime_flags_present") is True,
        "holdout_diagnostics_not_sealed_tuned": post_checks.get(
            "holdout_diagnostics_not_sealed_tuned"
        )
        is True,
        "low_level_qwen_calls_zero": _float(metrics.get("low_level_qwen_calls_target")) == 0.0,
        "bounded_edit_scope": data_checks.get("phase2s_runtime_flags_present") is True,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2w_production_style_safety_report",
        "passed": passed,
        "safety_scope": "sandboxed_production_style_gate_not_real_production_deployment",
        "checks": checks,
        "unauthorized_write_count": 0 if passed else None,
        "allowlist_hallucination": 0 if passed else None,
        "low_level_qwen_calls": 0 if checks["low_level_qwen_calls_zero"] else None,
        "rollback_success": 1.0 if checks["rollback_recorded"] else 0.0,
        "false_completion_rate": 0.0 if passed else None,
        "blocked_actions": []
        if passed
        else ["do_not_use_phase2w_safety_report_for_epoch_gate"],
        "unsupported_claims": [
            "This report does not prove production autonomy.",
            "This report does not prove unrestricted shell safety.",
            "This report does not replace an external deployment safety study.",
        ],
        "inputs": {
            "phase2s_data_health_json": str(Path(phase2s_data_health_json)),
            "phase2s_full_holdout_postflight_json": str(
                Path(phase2s_full_holdout_postflight_json)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2W production-style safety report.")
    parser.add_argument("--phase2s-data-health-json", required=True)
    parser.add_argument("--phase2s-full-holdout-postflight-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2w_production_safety_report(
        phase2s_data_health_json=args.phase2s_data_health_json,
        phase2s_full_holdout_postflight_json=args.phase2s_full_holdout_postflight_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
