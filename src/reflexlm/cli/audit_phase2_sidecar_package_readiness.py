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


def _exists(path: str | Path | None) -> bool:
    return bool(path) and Path(path).exists()


def _metric(report: dict[str, Any], key: str) -> float | None:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def audit_phase2_sidecar_package_readiness(
    *,
    architecture_boundary_json: str | Path,
    phase2am_reproduction_json: str | Path,
    phase2ap_control_synthesis_json: str | Path,
    phase2ae_structural_sidecar_comparison_json: str | Path | None = None,
    package_manifest_json: str | Path | None = None,
    package_runtime_execution_jsonl: str | Path | None = None,
    min_full_minus_source: float = 0.15,
    min_full_minus_erased: float = 0.25,
    min_full_minus_wrong: float = 0.25,
) -> dict[str, Any]:
    boundary = _read_json(architecture_boundary_json)
    phase2am = _read_json(phase2am_reproduction_json)
    phase2ap = _read_json(phase2ap_control_synthesis_json)
    phase2ae = (
        _read_json(phase2ae_structural_sidecar_comparison_json)
        if phase2ae_structural_sidecar_comparison_json
        else {}
    )

    source_delta = _metric(phase2am, "observed_min_full_minus_source_overlap")
    erased_delta = _metric(phase2am, "observed_min_full_minus_sidecar_erased")
    wrong_delta = _metric(phase2am, "observed_min_full_minus_wrong_sidecar")
    package_exists = _exists(package_manifest_json)
    runtime_exists = _exists(package_runtime_execution_jsonl)
    phase2ae_package_runtime_supported = (
        phase2ae.get("passed") is True
        and isinstance(phase2ae.get("checks"), dict)
        and phase2ae["checks"].get("provenance_audit_passed") is True
        and phase2ae["checks"].get("structural_sidecar_holdout_solves") is True
        and phase2ae["checks"].get("stripped_identity_holdout_does_not_solve") is True
        and phase2ae["checks"].get("full_beats_policyless_budget") is True
        and phase2ae["checks"].get("erased_structural_counterfactual_fails") is True
        and phase2ae["checks"].get("wrong_structural_counterfactual_fails") is True
    )
    strict_sidecar_ready = (
        phase2ap.get("strict_pure_sidecar_claim_ready") is True
        or phase2ae_package_runtime_supported
    )
    checks = {
        "architecture_boundary_bounded_passed": boundary.get("passed") is True,
        "architecture_boundary_strong_not_ready": boundary.get("strong_architecture_claim_ready")
        is False,
        "phase2am_reproduction_passed": phase2am.get("passed") is True
        and phase2am.get("natural_repo_disjoint_sidecar_dependency_reproduced") is True,
        "phase2am_delta_thresholds_met": isinstance(source_delta, float)
        and source_delta >= min_full_minus_source
        and isinstance(erased_delta, float)
        and erased_delta >= min_full_minus_erased
        and isinstance(wrong_delta, float)
        and wrong_delta >= min_full_minus_wrong,
        "phase2ap_stable_controls_passed": phase2ap.get("passed") is True
        and phase2ap.get("stable_bounded_sidecar_control_supported") is True,
        "phase2ap_strict_pure_sidecar_ready": phase2ap.get("strict_pure_sidecar_claim_ready")
        is True,
        "phase2ae_package_runtime_sidecar_supported": phase2ae_package_runtime_supported,
        "strict_sidecar_ready_from_ap_or_ae": strict_sidecar_ready,
        "package_manifest_present": package_exists,
        "package_runtime_execution_present": runtime_exists,
    }
    required_checks = {
        key: value
        for key, value in checks.items()
        if key
        not in {
            "phase2ap_strict_pure_sidecar_ready",
            "phase2ae_package_runtime_sidecar_supported",
        }
    }
    ready = all(required_checks.values())
    blockers: list[str] = []
    if not checks["strict_sidecar_ready_from_ap_or_ae"]:
        blockers.append("strict_pure_sidecar_causality_not_ready")
    if not package_exists:
        blockers.append("package_manifest_missing")
    if not runtime_exists:
        blockers.append("package_runtime_execution_missing")
    if not checks["phase2am_delta_thresholds_met"]:
        blockers.append("phase2am_delta_thresholds_not_met")

    return {
        "artifact_family": "phase2_sidecar_package_readiness_gate",
        "passed": ready,
        "ready_for_package": ready,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "metrics": {
            "phase2am_min_full_minus_source_overlap": source_delta,
            "phase2am_min_full_minus_sidecar_erased": erased_delta,
            "phase2am_min_full_minus_wrong_sidecar": wrong_delta,
            "phase2ae_package_runtime_sidecar_supported": 1.0
            if phase2ae_package_runtime_supported
            else 0.0,
        },
        "thresholds": {
            "min_full_minus_source": min_full_minus_source,
            "min_full_minus_erased": min_full_minus_erased,
            "min_full_minus_wrong": min_full_minus_wrong,
        },
        "blockers": blockers,
        "blocked_actions": [
            "do_not_package_until_gate_passes",
            "do_not_run_sealed_eval_until_package_runtime_execution_exists",
            "do_not_claim_epoch_making_architecture",
            "do_not_claim_open_ended_debugging_generalization",
        ]
        if not ready
        else ["do_not_run_sealed_eval_until_separate_sealed_gate_is_preregistered"],
        "allowed_next_action": (
            "build_nonsealed_package_runtime_sidecar_task_family"
            if not ready
            else "run_preregistered_package_level_nonsealed_runtime_evaluation"
        ),
        "inputs": {
            "architecture_boundary_json": str(Path(architecture_boundary_json)),
            "phase2am_reproduction_json": str(Path(phase2am_reproduction_json)),
            "phase2ap_control_synthesis_json": str(Path(phase2ap_control_synthesis_json)),
            "phase2ae_structural_sidecar_comparison_json": str(
                Path(phase2ae_structural_sidecar_comparison_json)
            )
            if phase2ae_structural_sidecar_comparison_json
            else None,
            "package_manifest_json": str(Path(package_manifest_json))
            if package_manifest_json
            else None,
            "package_runtime_execution_jsonl": str(Path(package_runtime_execution_jsonl))
            if package_runtime_execution_jsonl
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2 sidecar package readiness.")
    parser.add_argument("--architecture-boundary-json", required=True)
    parser.add_argument("--phase2am-reproduction-json", required=True)
    parser.add_argument("--phase2ap-control-synthesis-json", required=True)
    parser.add_argument("--phase2ae-structural-sidecar-comparison-json")
    parser.add_argument("--package-manifest-json")
    parser.add_argument("--package-runtime-execution-jsonl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2_sidecar_package_readiness(
        architecture_boundary_json=args.architecture_boundary_json,
        phase2am_reproduction_json=args.phase2am_reproduction_json,
        phase2ap_control_synthesis_json=args.phase2ap_control_synthesis_json,
        phase2ae_structural_sidecar_comparison_json=args.phase2ae_structural_sidecar_comparison_json,
        package_manifest_json=args.package_manifest_json,
        package_runtime_execution_jsonl=args.package_runtime_execution_jsonl,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
