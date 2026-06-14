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


def _run_condition(
    *,
    preserve_memory: bool,
    calibrated_threshold: float,
    prediction_error: float,
    observations_per_episode: int,
    max_episodes: int,
    stable_recovery_episodes: int,
) -> dict[str, Any]:
    controller = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=calibrated_threshold,
            preserve_adaptive_threshold_across_reset=preserve_memory,
        )
    )
    wake_episode: int | None = None
    wake_observation: int | None = None
    episode_trace: list[dict[str, Any]] = []
    for episode in range(1, max_episodes + 1):
        if episode > 1:
            controller.reset()
        start_threshold = float(
            controller.snapshot()["active_surprise_wake_threshold"]
        )
        reasons: list[str] = []
        for observation in range(1, observations_per_episode + 1):
            decision = controller.observe(
                proposed_action=ActionType.RUN_COMMAND,
                salience=0.20,
                risk=0.10,
                prediction_error=prediction_error,
                temporal_observation_available=True,
                failure_visible=True,
            )
            reasons.append(decision.reason)
            if decision.reason == "homeostatic_surprise_wake":
                wake_episode = episode
                wake_observation = observation
                break
        state = controller.snapshot()
        episode_trace.append(
            {
                "episode": episode,
                "start_threshold": start_threshold,
                "end_threshold": state["active_surprise_wake_threshold"],
                "decision_reasons": reasons,
                "lifetime_failure_sensitivity_adaptations": state[
                    "lifetime_failure_sensitivity_adaptations"
                ],
                "adaptive_threshold_preserved_resets": state[
                    "adaptive_threshold_preserved_resets"
                ],
            }
        )
        if wake_episode is not None:
            break
    threshold_at_wake_or_limit = float(
        controller.snapshot()["active_surprise_wake_threshold"]
    )
    recovery_trace: list[dict[str, Any]] = []
    for episode in range(1, stable_recovery_episodes + 1):
        controller.reset()
        start = float(controller.snapshot()["active_surprise_wake_threshold"])
        controller.observe(
            proposed_action=ActionType.WAIT,
            salience=0.05,
            risk=0.05,
            prediction_error=0.01,
            temporal_observation_available=True,
            failure_visible=False,
        )
        recovery_trace.append(
            {
                "episode": episode,
                "start_threshold": start,
                "end_threshold": controller.snapshot()[
                    "active_surprise_wake_threshold"
                ],
            }
        )
    safety = controller.observe(
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
        "preserve_memory": preserve_memory,
        "wake_episode": wake_episode,
        "wake_observation": wake_observation,
        "threshold_at_wake_or_limit": threshold_at_wake_or_limit,
        "final_threshold": final["active_surprise_wake_threshold"],
        "lifetime_failure_sensitivity_adaptations": final[
            "lifetime_failure_sensitivity_adaptations"
        ],
        "adaptive_threshold_preserved_resets": final[
            "adaptive_threshold_preserved_resets"
        ],
        "hard_safety_target": safety.target.value,
        "hard_safety_reason": safety.reason,
        "episode_trace": episode_trace,
        "recovery_trace": recovery_trace,
    }


def audit_phase2homeostasis_cross_episode_memory(
    *,
    calibrated_threshold: float = 0.250585894835774,
    prediction_error: float = 0.12,
    observations_per_episode: int = 2,
    max_episodes: int = 12,
    stable_recovery_episodes: int = 4,
    output_report_json: str | Path | None = None,
) -> dict[str, Any]:
    if not 0.0 < prediction_error < calibrated_threshold <= 1.0:
        raise ValueError("prediction_error must be below a valid calibrated threshold")
    if min(observations_per_episode, max_episodes, stable_recovery_episodes) <= 0:
        raise ValueError("episode counts must be positive")
    persistent = _run_condition(
        preserve_memory=True,
        calibrated_threshold=calibrated_threshold,
        prediction_error=prediction_error,
        observations_per_episode=observations_per_episode,
        max_episodes=max_episodes,
        stable_recovery_episodes=stable_recovery_episodes,
    )
    erased = _run_condition(
        preserve_memory=False,
        calibrated_threshold=calibrated_threshold,
        prediction_error=prediction_error,
        observations_per_episode=observations_per_episode,
        max_episodes=max_episodes,
        stable_recovery_episodes=stable_recovery_episodes,
    )
    persistent_starts = [
        float(row["start_threshold"]) for row in persistent["episode_trace"][1:]
    ]
    erased_starts = [
        float(row["start_threshold"]) for row in erased["episode_trace"][1:]
    ]
    checks = {
        "per_episode_signal_is_below_offline_threshold": (
            prediction_error < calibrated_threshold
        ),
        "persistent_memory_eventually_wakes": persistent["wake_episode"] is not None,
        "erased_memory_never_wakes": erased["wake_episode"] is None,
        "persistent_memory_carries_lowered_threshold_across_resets": bool(
            persistent_starts
        )
        and any(value < calibrated_threshold for value in persistent_starts),
        "erased_memory_restores_threshold_each_episode": bool(erased_starts)
        and all(value == calibrated_threshold for value in erased_starts),
        "persistent_lifetime_adaptation_accumulates": (
            persistent["lifetime_failure_sensitivity_adaptations"]
            > observations_per_episode
        ),
        "erased_lifetime_adaptation_does_not_accumulate": (
            erased["lifetime_failure_sensitivity_adaptations"]
            <= observations_per_episode
        ),
        "stable_episodes_restore_persistent_threshold_toward_set_point": (
            bool(persistent["recovery_trace"])
            and persistent["recovery_trace"][-1]["end_threshold"]
            > persistent["threshold_at_wake_or_limit"]
            and persistent["recovery_trace"][-1]["end_threshold"]
            <= calibrated_threshold
        ),
        "hard_safety_remains_dominant": all(
            row["hard_safety_target"] == InternalTarget.INHIBIT.value
            and row["hard_safety_reason"] == "hard_safety_inhibition"
            for row in (persistent, erased)
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_cross_episode_memory",
        "passed": passed,
        "ready_for_bounded_cross_episode_homeostatic_memory_claim": passed,
        "ready_for_long_term_general_plasticity_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "calibrated_threshold": calibrated_threshold,
            "prediction_error": prediction_error,
            "observations_per_episode": observations_per_episode,
            "max_episodes": max_episodes,
            "stable_recovery_episodes": stable_recovery_episodes,
            "persistent": persistent,
            "erased": erased,
        },
        "supported_claims": [
            (
                "explicit bounded cross-episode homeostatic memory accumulated "
                "sub-threshold visible-failure evidence across isolated episodes, "
                "eventually woke the bounded cortex, and remained recoverable toward "
                "the calibrated set point"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "unbounded long-term memory",
            "general plasticity",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "package_runtime_cross_episode_homeostatic_memory"
            if passed
            else "repair_cross_episode_homeostatic_memory"
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
        description="Audit bounded cross-episode homeostatic threshold memory."
    )
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument(
        "--calibrated-threshold",
        type=float,
        default=0.250585894835774,
    )
    parser.add_argument("--prediction-error", type=float, default=0.12)
    parser.add_argument("--observations-per-episode", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=12)
    parser.add_argument("--stable-recovery-episodes", type=int, default=4)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_cross_episode_memory(
        calibrated_threshold=args.calibrated_threshold,
        prediction_error=args.prediction_error,
        observations_per_episode=args.observations_per_episode,
        max_episodes=args.max_episodes,
        stable_recovery_episodes=args.stable_recovery_episodes,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
