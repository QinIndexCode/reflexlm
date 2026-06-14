from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.runtime.homeostasis import (
    HomeostaticControlConfig,
    HomeostaticSynapticController,
)
from reflexlm.schema import ActionType, InternalTarget


def _run_shift(
    *,
    prediction_error: float,
    adaptive: bool,
    calibrated_threshold: float,
    max_failure_observations: int,
    stable_recovery_observations: int,
) -> dict[str, Any]:
    controller = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=calibrated_threshold,
            online_failure_sensitivity_enabled=adaptive,
        )
    )
    wake_observation: int | None = None
    failure_trace: list[dict[str, Any]] = []
    for observation in range(1, max_failure_observations + 1):
        decision = controller.observe(
            proposed_action=ActionType.RUN_COMMAND,
            salience=0.20,
            risk=0.10,
            prediction_error=prediction_error,
            temporal_observation_available=True,
            failure_visible=True,
        )
        state = controller.snapshot()
        failure_trace.append(
            {
                "observation": observation,
                "decision_reason": decision.reason,
                "active_threshold": state["active_surprise_wake_threshold"],
                "failure_sensitivity_adaptations": state[
                    "failure_sensitivity_adaptations"
                ],
            }
        )
        if decision.reason == "homeostatic_surprise_wake":
            wake_observation = observation
            break
    threshold_at_wake_or_limit = float(
        controller.snapshot()["active_surprise_wake_threshold"]
    )
    recovery_trace: list[float] = []
    for _ in range(stable_recovery_observations):
        controller.observe(
            proposed_action=ActionType.WAIT,
            salience=0.05,
            risk=0.05,
            prediction_error=0.01,
            temporal_observation_available=True,
            failure_visible=False,
        )
        recovery_trace.append(
            float(controller.snapshot()["active_surprise_wake_threshold"])
        )
    safety_decision = controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=1.0,
        risk=1.0,
        prediction_error=prediction_error,
        temporal_observation_available=True,
        hard_dangerous=True,
        failure_visible=True,
    )
    final = controller.snapshot()
    return {
        "prediction_error": prediction_error,
        "adaptive": adaptive,
        "wake_observation": wake_observation,
        "threshold_at_wake_or_limit": threshold_at_wake_or_limit,
        "final_threshold": final["active_surprise_wake_threshold"],
        "minimum_threshold": final["config"]["minimum_surprise_wake_threshold"],
        "failure_sensitivity_adaptations": final[
            "failure_sensitivity_adaptations"
        ],
        "set_point_recovery_adaptations": final[
            "set_point_recovery_adaptations"
        ],
        "hard_safety_target": safety_decision.target.value,
        "hard_safety_reason": safety_decision.reason,
        "failure_trace": failure_trace,
        "recovery_trace": recovery_trace,
    }


def audit_phase2homeostasis_low_error_failure_shift(
    *,
    calibrated_threshold: float = 0.250585894835774,
    prediction_errors: tuple[float, ...] = (0.08, 0.12, 0.18),
    max_failure_observations: int = 64,
    stable_recovery_observations: int = 8,
    output_report_json: str | Path | None = None,
) -> dict[str, Any]:
    if not 0.0 < calibrated_threshold <= 1.0:
        raise ValueError("calibrated_threshold must be in (0, 1]")
    if any(not 0.0 < error < calibrated_threshold for error in prediction_errors):
        raise ValueError("prediction_errors must be positive and below the threshold")
    if max_failure_observations <= 0 or stable_recovery_observations <= 0:
        raise ValueError("observation counts must be positive")
    adaptive_rows = [
        _run_shift(
            prediction_error=error,
            adaptive=True,
            calibrated_threshold=calibrated_threshold,
            max_failure_observations=max_failure_observations,
            stable_recovery_observations=stable_recovery_observations,
        )
        for error in prediction_errors
    ]
    fixed_rows = [
        _run_shift(
            prediction_error=error,
            adaptive=False,
            calibrated_threshold=calibrated_threshold,
            max_failure_observations=max_failure_observations,
            stable_recovery_observations=stable_recovery_observations,
        )
        for error in prediction_errors
    ]
    checks = {
        "all_shift_signals_are_below_offline_calibrated_threshold": all(
            row["prediction_error"] < calibrated_threshold for row in adaptive_rows
        ),
        "adaptive_control_eventually_wakes_on_all_low_error_failures": all(
            row["wake_observation"] is not None for row in adaptive_rows
        ),
        "fixed_threshold_control_misses_all_low_error_failures": all(
            row["wake_observation"] is None for row in fixed_rows
        ),
        "adaptive_thresholds_remain_above_configured_minimum": all(
            row["threshold_at_wake_or_limit"] >= row["minimum_threshold"]
            for row in adaptive_rows
        ),
        "stable_outcomes_restore_threshold_toward_calibrated_set_point": all(
            row["recovery_trace"]
            and row["recovery_trace"][-1] > row["threshold_at_wake_or_limit"]
            and row["recovery_trace"][-1] <= calibrated_threshold
            for row in adaptive_rows
        ),
        "fixed_thresholds_remain_at_calibrated_set_point": all(
            row["threshold_at_wake_or_limit"] == calibrated_threshold
            and row["final_threshold"] == calibrated_threshold
            for row in fixed_rows
        ),
        "hard_safety_inhibition_remains_dominant_in_all_conditions": all(
            row["hard_safety_target"] == InternalTarget.INHIBIT.value
            and row["hard_safety_reason"] == "hard_safety_inhibition"
            for row in [*adaptive_rows, *fixed_rows]
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_low_error_failure_shift",
        "passed": passed,
        "ready_for_bounded_low_error_failure_shift_utility_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "calibrated_threshold": calibrated_threshold,
            "prediction_errors": list(prediction_errors),
            "max_failure_observations": max_failure_observations,
            "stable_recovery_observations": stable_recovery_observations,
            "adaptive_rows": adaptive_rows,
            "fixed_rows": fixed_rows,
        },
        "supported_claims": [
            (
                "bounded online homeostatic adaptation eliminated repeated low-error "
                "failure misses that remained below the offline calibrated threshold, "
                "while stable outcomes restored the threshold toward its set point and "
                "hard safety inhibition remained dominant"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "general anomaly detection",
            "long-term cross-episode plasticity",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "cross_runtime_online_homeostatic_adaptation_invariance"
            if passed
            else "repair_low_error_failure_shift_adaptation"
        ),
    }
    if output_report_json is not None:
        output = Path(output_report_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit online homeostatic utility under repeated low-error visible failures."
    )
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument(
        "--calibrated-threshold",
        type=float,
        default=0.250585894835774,
    )
    parser.add_argument(
        "--prediction-errors",
        nargs="+",
        type=float,
        default=[0.08, 0.12, 0.18],
    )
    parser.add_argument("--max-failure-observations", type=int, default=64)
    parser.add_argument("--stable-recovery-observations", type=int, default=8)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_low_error_failure_shift(
        calibrated_threshold=args.calibrated_threshold,
        prediction_errors=tuple(args.prediction_errors),
        max_failure_observations=args.max_failure_observations,
        stable_recovery_observations=args.stable_recovery_observations,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
