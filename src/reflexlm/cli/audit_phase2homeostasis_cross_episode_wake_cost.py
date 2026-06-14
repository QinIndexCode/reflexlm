from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _runtime_trace(report: dict[str, Any]) -> dict[str, Any]:
    action_trace: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    subreport_count = 0
    for repository in report.get("repository_reports", []):
        path = repository.get("report_json")
        if not path:
            continue
        subreport = _read_json(path)
        subreport_count += 1
        for episode in subreport.get("episode_reports", []):
            selected_actions = [
                {
                    "type": action.get("type"),
                    "command": action.get("command"),
                    "file_target": action.get("file_target"),
                    "reason": action.get("reason"),
                }
                for action in episode.get("selected_actions", [])
            ]
            action_trace.append(
                {
                    "repository_id": repository.get("repository_id"),
                    "episode_id": episode.get("episode_id"),
                    "selected_actions": selected_actions,
                }
            )
            for step in episode.get("policy_debug_steps", []):
                reason = str(
                    step.get("homeostatic_decision", {}).get("reason", "")
                )
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "subreport_count": subreport_count,
        "action_trace": action_trace,
        "reason_counts": reason_counts,
        "surprise_wakes": reason_counts.get("homeostatic_surprise_wake", 0),
        "persistent_failure_wakes": reason_counts.get(
            "homeostatic_persistent_failure_wake",
            0,
        ),
        "total_wakes": reason_counts.get("homeostatic_surprise_wake", 0)
        + reason_counts.get("homeostatic_persistent_failure_wake", 0),
    }


def _homeostatic_states(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        repository.get("policy_configuration", {})
        .get("policy_metadata", {})
        .get("expert_policy", {})
        .get("homeostatic_control", {})
        for repository in report.get("repository_reports", [])
    ]


def audit_phase2homeostasis_cross_episode_wake_cost(
    *,
    persistent_report_json: str | Path,
    erased_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    persistent = _read_json(persistent_report_json)
    erased = _read_json(erased_report_json)
    persistent_trace = _runtime_trace(persistent)
    erased_trace = _runtime_trace(erased)
    persistent_states = _homeostatic_states(persistent)
    erased_states = _homeostatic_states(erased)
    checks = {
        "persistent_runtime_passed": persistent.get("passed") is True,
        "erased_runtime_passed": erased.get("passed") is True,
        "all_persistent_subreports_loaded": (
            persistent_trace["subreport_count"]
            == len(persistent.get("repository_reports", []))
            > 0
        ),
        "all_erased_subreports_loaded": (
            erased_trace["subreport_count"]
            == len(erased.get("repository_reports", []))
            > 0
        ),
        "persistent_memory_enabled": bool(persistent_states)
        and all(
            state.get("config", {}).get(
                "preserve_adaptive_threshold_across_reset"
            )
            is True
            for state in persistent_states
        ),
        "erased_memory_disabled": bool(erased_states)
        and all(
            state.get("config", {}).get(
                "preserve_adaptive_threshold_across_reset"
            )
            is False
            for state in erased_states
        ),
        "persistent_reset_decay_exercised": bool(persistent_states)
        and all(
            int(state.get("adaptive_threshold_reset_decay_events", 0)) > 0
            for state in persistent_states
        ),
        "persistent_and_erased_action_traces_match": (
            bool(persistent_trace["action_trace"])
            and persistent_trace["action_trace"] == erased_trace["action_trace"]
        ),
        "persistent_memory_adds_no_surprise_wake_cost": (
            persistent_trace["surprise_wakes"] <= erased_trace["surprise_wakes"]
        ),
        "persistent_memory_preserves_failure_wake_behavior": (
            persistent_trace["persistent_failure_wakes"]
            == erased_trace["persistent_failure_wakes"]
        ),
        "persistent_memory_adds_no_total_wake_cost": (
            persistent_trace["total_wakes"] <= erased_trace["total_wakes"]
        ),
        "same_core_completion_metrics": persistent.get("metrics")
        == erased.get("metrics"),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_cross_episode_wake_cost",
        "passed": passed,
        "ready_for_bounded_cross_episode_memory_without_added_wake_cost_claim": passed,
        "ready_for_cross_episode_memory_performance_gain_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "persistent_trace": persistent_trace,
            "erased_trace": erased_trace,
            "persistent_homeostatic_states": persistent_states,
            "erased_homeostatic_states": erased_states,
        },
        "supported_claims": [
            (
                "bounded cross-episode threshold retention accumulated adaptive state "
                "while reset-time set-point decay prevented additional surprise wakes, "
                "total wakes, or selected-action changes on the matched real runtime matrix"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "cross-episode memory improves task completion on this matrix",
            "general false-positive control",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "cross_runtime_package_cross_episode_memory_invariance"
            if passed
            else "repair_cross_episode_wake_cost"
        ),
        "evidence": {
            "persistent_report_json": str(persistent_report_json),
            "erased_report_json": str(erased_report_json),
        },
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit wake and action cost of bounded cross-episode homeostatic memory."
    )
    parser.add_argument("--persistent-report-json", required=True)
    parser.add_argument("--erased-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_cross_episode_wake_cost(
        persistent_report_json=args.persistent_report_json,
        erased_report_json=args.erased_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
