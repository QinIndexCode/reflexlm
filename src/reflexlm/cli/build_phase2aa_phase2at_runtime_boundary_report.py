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


def build_phase2aa_phase2at_runtime_boundary_report(
    *,
    phase2aa_data_health_json: str | Path,
    phase2aa_candidate_delta_gate_json: str | Path,
    phase2at_symbolic_runtime_delta_gate_json: str | Path,
    min_phase2aa_rows: int = 24,
) -> dict[str, Any]:
    phase2aa_data = _read_json(phase2aa_data_health_json)
    phase2aa_delta = _read_json(phase2aa_candidate_delta_gate_json)
    phase2at_delta = _read_json(phase2at_symbolic_runtime_delta_gate_json)
    phase2aa_metrics = _dict(phase2aa_delta.get("metrics"))
    phase2at_metrics = _dict(phase2at_delta.get("metrics"))

    phase2aa_rows = int(phase2aa_metrics.get("full_rows") or 0)
    checks = {
        "phase2aa_data_health_passed": phase2aa_data.get("passed") is True,
        "phase2aa_artifacts_resolved": _dict(phase2aa_data.get("checks")).get(
            "required_runtime_artifacts_available"
        )
        is True,
        "phase2aa_candidate_delta_gate_passed": phase2aa_delta.get("passed") is True,
        "phase2aa_row_minimum_met": phase2aa_rows >= min_phase2aa_rows,
        "phase2at_symbolic_delta_gate_present": bool(phase2at_delta),
        "phase2at_symbolic_delta_not_misclaimed": phase2at_delta.get("passed") is not True,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2aa_phase2at_runtime_boundary_report",
        "passed": passed,
        "claim_boundary": (
            "The current runtime evidence supports a bounded Phase2AA patch-candidate "
            "selection package delta on non-sealed public-repo rows. It does not support "
            "a Phase2AT symbolic structural package-runtime delta because the no-policy "
            "symbolic runner tied the loaded package. The supported claim is therefore "
            "candidate selection under bounded alternatives, not freeform repair or "
            "autonomous debugging."
        ),
        "checks": checks,
        "supported_claims": [
            "bounded_patch_candidate_selection_package_delta_on_nonsealed_public_repo_rows"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "phase2at_symbolic_structural_runtime_policy_delta",
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "next_required_evidence": [
            *(
                ["scale_candidate_selection_delta_to_full_256_row_holdout"]
                if phase2aa_rows < 256
                else []
            ),
            "add_non_oracle_no_policy_controls_to_all_runtime_execution_reports",
            "replicate_candidate_selection_delta_across_models_and_seeds",
            "design_nonsealed_runtime_tasks_where_symbolic_no_policy_cannot_solve_by_parser_only",
        ],
        "metrics": {
            "phase2aa_full_rows": phase2aa_rows,
            "phase2aa_control_rows": phase2aa_metrics.get("control_rows"),
            "phase2aa_full_success_rate": phase2aa_metrics.get("full_success_rate"),
            "phase2aa_control_success_rate": phase2aa_metrics.get("control_success_rate"),
            "phase2aa_full_minus_control_success_rate": phase2aa_metrics.get(
                "full_minus_control_success_rate"
            ),
            "phase2aa_full_selection_accuracy": phase2aa_metrics.get(
                "full_selection_accuracy"
            ),
            "phase2aa_control_selection_accuracy": phase2aa_metrics.get(
                "control_selection_accuracy"
            ),
            "phase2aa_full_minus_control_selection_accuracy": phase2aa_metrics.get(
                "full_minus_control_selection_accuracy"
            ),
            "phase2at_full_success_rate": phase2at_metrics.get("full_success_rate"),
            "phase2at_control_success_rate": phase2at_metrics.get("control_success_rate"),
            "phase2at_full_minus_control_success_rate": phase2at_metrics.get(
                "full_minus_control_success_rate"
            ),
        },
        "inputs": {
            "phase2aa_data_health_json": str(Path(phase2aa_data_health_json)),
            "phase2aa_candidate_delta_gate_json": str(
                Path(phase2aa_candidate_delta_gate_json)
            ),
            "phase2at_symbolic_runtime_delta_gate_json": str(
                Path(phase2at_symbolic_runtime_delta_gate_json)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a boundary report separating Phase2AA and Phase2AT runtime evidence."
    )
    parser.add_argument("--phase2aa-data-health-json", required=True)
    parser.add_argument("--phase2aa-candidate-delta-gate-json", required=True)
    parser.add_argument("--phase2at-symbolic-runtime-delta-gate-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-phase2aa-rows", type=int, default=24)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2aa_phase2at_runtime_boundary_report(
        phase2aa_data_health_json=args.phase2aa_data_health_json,
        phase2aa_candidate_delta_gate_json=args.phase2aa_candidate_delta_gate_json,
        phase2at_symbolic_runtime_delta_gate_json=args.phase2at_symbolic_runtime_delta_gate_json,
        min_phase2aa_rows=args.min_phase2aa_rows,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
