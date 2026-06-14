from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2homeostasis_cross_runtime_dynamics import (
    _normalize_runtime_command,
    _read_json,
)
from reflexlm.cli.audit_phase2homeostasis_publication_reproducibility_manifest import (
    _write_json,
)
from reflexlm.runtime.homeostasis import SIDE_EFFECT_ACTIONS


OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_exact_cross_runtime_homeostatic_dynamics_claim",
    "ready_for_unbounded_long_term_memory_claim",
    "ready_for_general_runtime_interpreter_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _side_effect_trace(report: dict[str, Any]) -> list[dict[str, Any]]:
    side_effect_types = {action.value for action in SIDE_EFFECT_ACTIONS}
    runtime_python = str(
        report.get("runtime_interpreter")
        or report.get("runtime_environment", {}).get("executable")
        or ""
    )
    trace: list[dict[str, Any]] = []
    for repository in sorted(
        report.get("repository_reports", []),
        key=lambda item: str(item.get("repository_id", "")),
    ):
        report_json = repository.get("report_json")
        if not report_json:
            continue
        subreport = _read_json(report_json)
        for episode in subreport.get("episode_reports", []):
            trace.append(
                {
                    "repository_id": repository.get("repository_id"),
                    "episode_id": episode.get("episode_id"),
                    "actions": [
                        {
                            "type": action.get("type"),
                            "command": _normalize_runtime_command(
                                action.get("command"),
                                runtime_python=runtime_python,
                            ),
                            "file_target": action.get("file_target"),
                        }
                        for action in episode.get("selected_actions", [])
                        if action.get("type") in side_effect_types
                    ],
                }
            )
    return trace


def _report_passed(path: str | Path) -> bool:
    return _read_json(path).get("passed") is True


def _runtime_summary(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    return {
        "passed": report.get("passed") is True,
        "runtime_interpreter": report.get("runtime_interpreter"),
        "python_version": report.get("runtime_environment", {}).get("version"),
        "episodes": metrics.get("episodes"),
        "executed_actions": metrics.get("executed_actions"),
        "task_completion_success_rate": metrics.get("task_completion_success_rate"),
    }


def validate_phase2homeostasis_fresh_rerun_limitations(
    report: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "artifact_family_matches": (
            report.get("artifact_family")
            == "phase2homeostasis_fresh_rerun_limitations"
        ),
        "top_level_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_homeostasis_fresh_behavioral_claim") is True
            and all(report.get(flag) is not True for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "exact_cross_runtime_limitation_recorded": (
            report.get("limitations", {}).get(
                "exact_cross_runtime_homeostatic_dynamics"
            )
            == "not_supported_by_this_fresh_rerun"
            and report.get("ready_for_exact_cross_runtime_homeostatic_dynamics_claim")
            is False
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": report.get("metrics", {}),
    }


def audit_phase2homeostasis_fresh_rerun_limitations(
    *,
    generation1_report_json: str | Path,
    generation2_py313_report_json: str | Path,
    generation2_py312_report_json: str | Path,
    chain_py313_report_json: str | Path,
    chain_py313_to_py312_report_json: str | Path,
    cross_runtime_dynamics_report_json: str | Path,
    phase2cj_report_json: str | Path,
    output_report_json: str | Path,
    threshold_drift_limit: float = 0.01,
) -> dict[str, Any]:
    generation1 = _read_json(generation1_report_json)
    generation2_py313 = _read_json(generation2_py313_report_json)
    generation2_py312 = _read_json(generation2_py312_report_json)
    chain_py313 = _read_json(chain_py313_report_json)
    chain_py312 = _read_json(chain_py313_to_py312_report_json)
    cross_runtime = _read_json(cross_runtime_dynamics_report_json)
    phase2cj = _read_json(phase2cj_report_json)
    max_delta = cross_runtime.get("metrics", {}).get("maximum_active_threshold_delta")
    side_effect_traces_match = (
        _side_effect_trace(generation2_py313) == _side_effect_trace(generation2_py312)
    )
    checks = {
        "all_fresh_runtime_reports_passed": all(
            report.get("passed") is True
            for report in (generation1, generation2_py313, generation2_py312)
        ),
        "fresh_runtime_metrics_match": (
            generation2_py313.get("metrics") == generation2_py312.get("metrics")
        ),
        "fresh_hmac_chains_passed": (
            chain_py313.get("passed") is True and chain_py312.get("passed") is True
        ),
        "fresh_hmac_chains_preserve_side_effect_trace": (
            chain_py313.get("checks", {}).get("side_effect_action_traces_match") is True
            and chain_py312.get("checks", {}).get("side_effect_action_traces_match")
            is True
        ),
        "fresh_phase2cj_passed": phase2cj.get("passed") is True,
        "fresh_cross_runtime_exact_dynamics_not_claimed": (
            cross_runtime.get("passed") is False
        ),
        "fresh_cross_runtime_side_effect_traces_match": side_effect_traces_match,
        "fresh_threshold_drift_bounded_for_limitation": (
            isinstance(max_delta, (int, float)) and float(max_delta) <= threshold_drift_limit
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_fresh_rerun_limitations",
        "passed": passed,
        "ready_for_bounded_homeostasis_fresh_behavioral_claim": passed,
        "ready_for_exact_cross_runtime_homeostatic_dynamics_claim": False,
        "ready_for_unbounded_long_term_memory_claim": False,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "threshold_drift_limit": threshold_drift_limit,
            "maximum_active_threshold_delta": max_delta,
            "generation1": _runtime_summary(generation1),
            "generation2_py313": _runtime_summary(generation2_py313),
            "generation2_py312": _runtime_summary(generation2_py312),
            "side_effect_trace_rows": len(_side_effect_trace(generation2_py313)),
        },
        "limitations": {
            "exact_cross_runtime_homeostatic_dynamics": (
                "not_supported_by_this_fresh_rerun"
            ),
            "observed_cross_runtime_dynamics_report": str(
                cross_runtime_dynamics_report_json
            ),
        },
        "claim_boundary": (
            "This fresh rerun supports bounded HMAC-authenticated state transfer, "
            "task completion, Phase2CJ runtime-interpreter behavior, and side-effect "
            "trace stability. It does not support exact cross-runtime internal "
            "homeostatic microdynamics, unbounded memory, open-ended native "
            "perception, production autonomy, or epoch-making architecture claims."
        ),
        "supported_claims": [
            (
                "bounded fresh behavioral rerun for HMAC-authenticated homeostatic "
                "persistent-state transfer with explicit internal-dynamics limitation"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "exact cross-runtime homeostatic microdynamics",
            "unbounded or semantic long-term memory",
            "free-form shell autonomy",
            "general runtime interpreter invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "narrow_publication_claim_boundary_or_add_deterministic_calibration"
            if passed
            else "repair_fresh_rerun_limitation_gate"
        ),
        "evidence": {
            "generation1_report_json": str(generation1_report_json),
            "generation2_py313_report_json": str(generation2_py313_report_json),
            "generation2_py312_report_json": str(generation2_py312_report_json),
            "chain_py313_report_json": str(chain_py313_report_json),
            "chain_py313_to_py312_report_json": str(chain_py313_to_py312_report_json),
            "cross_runtime_dynamics_report_json": str(cross_runtime_dynamics_report_json),
            "phase2cj_report_json": str(phase2cj_report_json),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit fresh rerun limitations for Phase2Homeostasis HMAC evidence."
    )
    parser.add_argument("--generation1-report-json", required=True)
    parser.add_argument("--generation2-py313-report-json", required=True)
    parser.add_argument("--generation2-py312-report-json", required=True)
    parser.add_argument("--chain-py313-report-json", required=True)
    parser.add_argument("--chain-py313-to-py312-report-json", required=True)
    parser.add_argument("--cross-runtime-dynamics-report-json", required=True)
    parser.add_argument("--phase2cj-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--threshold-drift-limit", type=float, default=0.01)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_fresh_rerun_limitations(
        generation1_report_json=args.generation1_report_json,
        generation2_py313_report_json=args.generation2_py313_report_json,
        generation2_py312_report_json=args.generation2_py312_report_json,
        chain_py313_report_json=args.chain_py313_report_json,
        chain_py313_to_py312_report_json=args.chain_py313_to_py312_report_json,
        cross_runtime_dynamics_report_json=args.cross_runtime_dynamics_report_json,
        phase2cj_report_json=args.phase2cj_report_json,
        output_report_json=args.output_report_json,
        threshold_drift_limit=args.threshold_drift_limit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
