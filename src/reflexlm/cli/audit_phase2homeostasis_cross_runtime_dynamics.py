from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _repo_signature(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "repository_id": row.get("repository_id"),
            "origin": row.get("provenance", {}).get("origin"),
            "head": row.get("provenance", {}).get("head"),
            "recipe_ids": list(row.get("recipe_ids", [])),
            "contract_signatures": list(row.get("contract_signatures", [])),
        }
        for row in sorted(
            report.get("repository_reports", []),
            key=lambda item: str(item.get("repository_id", "")),
        )
    ]


def _runtime_trace(report: dict[str, Any]) -> dict[str, Any]:
    action_trace: list[dict[str, Any]] = []
    wake_reason_counts: dict[str, int] = {}
    subreport_count = 0
    runtime_python = str(
        report.get("runtime_interpreter")
        or report.get("runtime_environment", {}).get("executable")
        or ""
    )
    for repository in sorted(
        report.get("repository_reports", []),
        key=lambda item: str(item.get("repository_id", "")),
    ):
        report_json = repository.get("report_json")
        if not report_json:
            continue
        subreport = _read_json(report_json)
        subreport_count += 1
        for episode in subreport.get("episode_reports", []):
            action_trace.append(
                {
                    "repository_id": repository.get("repository_id"),
                    "episode_id": episode.get("episode_id"),
                    "selected_actions": [
                        {
                            "type": action.get("type"),
                            "command": _normalize_runtime_command(
                                action.get("command"),
                                runtime_python=runtime_python,
                            ),
                            "file_target": action.get("file_target"),
                            "reason": action.get("reason"),
                        }
                        for action in episode.get("selected_actions", [])
                    ],
                }
            )
            for step in episode.get("policy_debug_steps", []):
                reason = str(
                    step.get("homeostatic_decision", {}).get("reason", "")
                )
                if reason:
                    wake_reason_counts[reason] = wake_reason_counts.get(reason, 0) + 1
    return {
        "subreport_count": subreport_count,
        "action_trace": action_trace,
        "wake_reason_counts": wake_reason_counts,
    }


def _normalize_runtime_command(command: Any, *, runtime_python: str) -> Any:
    if not isinstance(command, str) or not runtime_python:
        return command
    if command == runtime_python:
        return "<RUNTIME_PYTHON>"
    prefix = f"{runtime_python} "
    if command.startswith(prefix):
        return f"<RUNTIME_PYTHON>{command[len(runtime_python):]}"
    return command


def _state_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repository in sorted(
        report.get("repository_reports", []),
        key=lambda item: str(item.get("repository_id", "")),
    ):
        state = (
            repository.get("policy_configuration", {})
            .get("policy_metadata", {})
            .get("expert_policy", {})
            .get("homeostatic_control", {})
        )
        config = state.get("config", {})
        rows.append(
            {
                "repository_id": repository.get("repository_id"),
                "preserve_memory": config.get(
                    "preserve_adaptive_threshold_across_reset"
                ),
                "baseline_threshold": config.get("surprise_wake_threshold"),
                "active_threshold": state.get("active_surprise_wake_threshold"),
                "lifetime_failure_adaptations": state.get(
                    "lifetime_failure_sensitivity_adaptations"
                ),
                "lifetime_recovery_adaptations": state.get(
                    "lifetime_set_point_recovery_adaptations"
                ),
                "preserved_resets": state.get("adaptive_threshold_preserved_resets"),
                "reset_decay_events": state.get(
                    "adaptive_threshold_reset_decay_events"
                ),
                "wake_events": state.get("wake_events"),
            }
        )
    return rows


def _discrete_state_signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"baseline_threshold", "active_threshold"}
        }
        for row in rows
    ]


def audit_phase2homeostasis_cross_runtime_dynamics(
    *,
    canonical_report_json: str | Path,
    alternate_report_json: str | Path,
    output_report_json: str | Path,
    threshold_tolerance: float = 0.005,
) -> dict[str, Any]:
    if threshold_tolerance < 0.0:
        raise ValueError("threshold_tolerance must be non-negative")
    canonical = _read_json(canonical_report_json)
    alternate = _read_json(alternate_report_json)
    canonical_trace = _runtime_trace(canonical)
    alternate_trace = _runtime_trace(alternate)
    canonical_rows = _state_rows(canonical)
    alternate_rows = _state_rows(alternate)
    threshold_deltas = [
        {
            "repository_id": canonical_row["repository_id"],
            "absolute_delta": abs(
                float(canonical_row["active_threshold"])
                - float(alternate_row["active_threshold"])
            ),
        }
        for canonical_row, alternate_row in zip(canonical_rows, alternate_rows)
    ]
    checks = {
        "canonical_runtime_passed": canonical.get("passed") is True,
        "alternate_runtime_passed": alternate.get("passed") is True,
        "runtime_interpreters_differ": (
            canonical.get("runtime_environment", {}).get("executable")
            != alternate.get("runtime_environment", {}).get("executable")
        ),
        "same_seed_and_task_matrix": canonical.get("seed") == alternate.get("seed")
        and _repo_signature(canonical) == _repo_signature(alternate),
        "same_core_completion_metrics": canonical.get("metrics")
        == alternate.get("metrics"),
        "all_canonical_subreports_loaded": (
            canonical_trace["subreport_count"]
            == len(canonical.get("repository_reports", []))
            > 0
        ),
        "all_alternate_subreports_loaded": (
            alternate_trace["subreport_count"]
            == len(alternate.get("repository_reports", []))
            > 0
        ),
        "cross_episode_memory_enabled_both_runtimes": bool(canonical_rows)
        and len(canonical_rows) == len(alternate_rows)
        and all(row["preserve_memory"] is True for row in canonical_rows)
        and all(row["preserve_memory"] is True for row in alternate_rows),
        "both_runtimes_exercised_persistent_memory": bool(canonical_rows)
        and all(int(row["preserved_resets"] or 0) > 0 for row in canonical_rows)
        and all(int(row["preserved_resets"] or 0) > 0 for row in alternate_rows),
        "discrete_homeostatic_dynamics_match": (
            bool(canonical_rows)
            and _discrete_state_signature(canonical_rows)
            == _discrete_state_signature(alternate_rows)
        ),
        "active_threshold_deltas_within_tolerance": bool(threshold_deltas)
        and all(
            row["absolute_delta"] <= threshold_tolerance for row in threshold_deltas
        ),
        "wake_reason_counts_match": (
            canonical_trace["wake_reason_counts"]
            == alternate_trace["wake_reason_counts"]
        ),
        "runtime_normalized_executable_action_traces_match": bool(
            canonical_trace["action_trace"]
        )
        and canonical_trace["action_trace"] == alternate_trace["action_trace"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_cross_runtime_dynamics",
        "passed": passed,
        "ready_for_bounded_cross_runtime_homeostatic_dynamics_claim": passed,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "threshold_tolerance": threshold_tolerance,
            "maximum_active_threshold_delta": max(
                (row["absolute_delta"] for row in threshold_deltas),
                default=None,
            ),
            "threshold_deltas": threshold_deltas,
            "canonical_states": canonical_rows,
            "alternate_states": alternate_rows,
            "canonical_wake_reason_counts": canonical_trace["wake_reason_counts"],
            "alternate_wake_reason_counts": alternate_trace["wake_reason_counts"],
            "runtime_command_normalization": (
                "only the exact active runtime interpreter at command start is "
                "replaced with <RUNTIME_PYTHON>"
            ),
        },
        "supported_claims": [
            (
                "bounded cross-episode homeostatic memory produced matching discrete "
                "adaptation, recovery, reset-decay, wake-reason, and executable-action "
                "dynamics across the two recorded Python runtimes, with active threshold "
                "drift bounded by the declared tolerance"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "general interpreter-version invariance",
            "operating-system or shell invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "versioned_persistent_homeostatic_state_artifact"
            if passed
            else "repair_cross_runtime_homeostatic_dynamics"
        ),
        "evidence": {
            "canonical_report_json": str(canonical_report_json),
            "alternate_report_json": str(alternate_report_json),
        },
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit cross-runtime internal homeostatic dynamics."
    )
    parser.add_argument("--canonical-report-json", required=True)
    parser.add_argument("--alternate-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--threshold-tolerance", type=float, default=0.005)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_cross_runtime_dynamics(
        canonical_report_json=args.canonical_report_json,
        alternate_report_json=args.alternate_report_json,
        output_report_json=args.output_report_json,
        threshold_tolerance=args.threshold_tolerance,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
