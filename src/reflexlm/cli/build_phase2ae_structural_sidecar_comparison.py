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


def build_phase2ae_structural_sidecar_comparison(
    *,
    data_health_json: str | Path,
    full_summary_json: str | Path,
    policyless_summary_json: str | Path,
    output_json: str | Path,
    provenance_audit_json: str | Path | None = None,
    prior_full_summary_json: str | Path | None = None,
    erased_structural_summary_json: str | Path | None = None,
    wrong_structural_summary_json: str | Path | None = None,
    min_full_success_rate: float = 0.85,
    min_full_minus_policyless: float = 0.50,
    max_counterfactual_success_rate: float = 0.05,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    full_summary = _read_json(full_summary_json)
    policyless_summary = _read_json(policyless_summary_json)
    provenance_audit = _read_json(provenance_audit_json) if provenance_audit_json else {}
    prior_full_summary = _read_json(prior_full_summary_json) if prior_full_summary_json else {}
    erased_summary = (
        _read_json(erased_structural_summary_json) if erased_structural_summary_json else {}
    )
    wrong_summary = (
        _read_json(wrong_structural_summary_json) if wrong_structural_summary_json else {}
    )
    metrics = data_health.get("metrics", {}) if isinstance(data_health.get("metrics"), dict) else {}
    structural_holdout = float(
        metrics.get("structural_sidecar_accuracy", {}).get("holdout", 0.0) or 0.0
    )
    stripped_identity_holdout = float(
        metrics.get("stripped_identity_accuracy", {}).get("holdout", 0.0) or 0.0
    )
    full_success = float(full_summary.get("success_rate") or 0.0)
    policyless_success = float(policyless_summary.get("success_rate") or 0.0)
    prior_full_success = (
        float(prior_full_summary.get("success_rate") or 0.0) if prior_full_summary else None
    )
    erased_success = (
        float(erased_summary.get("success_rate") or 0.0) if erased_summary else None
    )
    wrong_success = float(wrong_summary.get("success_rate") or 0.0) if wrong_summary else None
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "provenance_audit_passed": not provenance_audit
        or provenance_audit.get("passed") is True,
        "structural_sidecar_holdout_solves": structural_holdout == 1.0,
        "stripped_identity_holdout_does_not_solve": stripped_identity_holdout == 0.0,
        "full_success_rate_gate": full_success >= min_full_success_rate,
        "full_beats_policyless_budget": (full_success - policyless_success)
        >= min_full_minus_policyless,
        "erased_structural_counterfactual_fails": erased_success is None
        or erased_success <= max_counterfactual_success_rate,
        "wrong_structural_counterfactual_fails": wrong_success is None
        or wrong_success <= max_counterfactual_success_rate,
    }
    report = {
        "artifact_family": "phase2ae_structural_sidecar_budget_pressure_comparison",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "full_success_rate": full_success,
            "policyless_slot0_budget2_success_rate": policyless_success,
            "full_minus_policyless_slot0_budget2": full_success - policyless_success,
            "structural_sidecar_holdout_accuracy": structural_holdout,
            "stripped_identity_holdout_accuracy": stripped_identity_holdout,
            "prior_full_success_rate_before_runtime_signal_fix": prior_full_success,
            "runtime_signal_fix_delta": None
            if prior_full_success is None
            else full_success - prior_full_success,
            "erased_structural_success_rate": erased_success,
            "wrong_structural_success_rate": wrong_success,
            "full_minus_erased_structural": None
            if erased_success is None
            else full_success - erased_success,
            "full_minus_wrong_structural": None
            if wrong_success is None
            else full_success - wrong_success,
        },
        "interpretation": {
            "supported": [
                "runtime structural sidecar can close the residual budget-pressure gap when legacy file/symbol identity is neutralized",
                "selected rows have saved generated-test provenance and structural sidecar fields match those tests",
                "bounded retry with identity_first can execute the correct recorded patch candidate under a two-attempt budget",
                "erasing or corrupting the structural sidecar blocks execution on this benchmark",
            ]
            if all(checks.values())
            else [],
            "claim_boundary": (
                "structural sidecar sufficiency for this non-sealed residual benchmark; not learned-head superiority over deterministic structural sidecar"
            ),
        },
        "thresholds": {
            "min_full_success_rate": min_full_success_rate,
            "min_full_minus_policyless": min_full_minus_policyless,
            "max_counterfactual_success_rate": max_counterfactual_success_rate,
        },
        "unsupported_claims": [
            "learned_head_advantage_over_structural_sidecar",
            "freeform_patch_generation",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "provenance_audit_json": str(Path(provenance_audit_json))
            if provenance_audit_json
            else None,
            "full_summary_json": str(Path(full_summary_json)),
            "policyless_summary_json": str(Path(policyless_summary_json)),
            "prior_full_summary_json": str(Path(prior_full_summary_json))
            if prior_full_summary_json
            else None,
            "erased_structural_summary_json": str(Path(erased_structural_summary_json))
            if erased_structural_summary_json
            else None,
            "wrong_structural_summary_json": str(Path(wrong_structural_summary_json))
            if wrong_structural_summary_json
            else None,
        },
    }
    _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Phase2AE structural-sidecar budget-pressure execution against controls."
    )
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--provenance-audit-json")
    parser.add_argument("--full-summary-json", required=True)
    parser.add_argument("--policyless-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prior-full-summary-json")
    parser.add_argument("--erased-structural-summary-json")
    parser.add_argument("--wrong-structural-summary-json")
    args = parser.parse_args()
    report = build_phase2ae_structural_sidecar_comparison(
        data_health_json=args.data_health_json,
        full_summary_json=args.full_summary_json,
        policyless_summary_json=args.policyless_summary_json,
        output_json=args.output_json,
        provenance_audit_json=args.provenance_audit_json,
        prior_full_summary_json=args.prior_full_summary_json,
        erased_structural_summary_json=args.erased_structural_summary_json,
        wrong_structural_summary_json=args.wrong_structural_summary_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
