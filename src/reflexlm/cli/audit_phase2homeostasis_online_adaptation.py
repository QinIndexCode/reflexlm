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


def _homeostatic_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repository in report.get("repository_reports", []):
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
                "enabled": config.get("online_failure_sensitivity_enabled"),
                "baseline_threshold": config.get("surprise_wake_threshold"),
                "active_threshold": state.get("active_surprise_wake_threshold"),
                "minimum_threshold": config.get(
                    "minimum_surprise_wake_threshold"
                ),
                "failure_sensitivity_adaptations": state.get(
                    "failure_sensitivity_adaptations"
                ),
                "set_point_recovery_adaptations": state.get(
                    "set_point_recovery_adaptations"
                ),
                "last_threshold_adaptation": state.get(
                    "last_threshold_adaptation"
                ),
            }
        )
    return rows


def _all_repo_check(report: dict[str, Any], name: str) -> bool:
    rows = report.get("repository_reports", [])
    return bool(rows) and all(row.get("checks", {}).get(name) is True for row in rows)


def audit_phase2homeostasis_online_adaptation(
    *,
    adaptive_report_json: str | Path,
    fixed_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    adaptive = _read_json(adaptive_report_json)
    fixed = _read_json(fixed_report_json)
    adaptive_rows = _homeostatic_rows(adaptive)
    fixed_rows = _homeostatic_rows(fixed)
    checks = {
        "adaptive_runtime_passed": adaptive.get("passed") is True,
        "fixed_threshold_ablation_runtime_passed": fixed.get("passed") is True,
        "same_seed_and_task_matrix": adaptive.get("seed") == fixed.get("seed")
        and _repo_signature(adaptive) == _repo_signature(fixed),
        "same_core_completion_metrics": adaptive.get("metrics")
        == fixed.get("metrics"),
        "adaptive_rows_present": bool(adaptive_rows),
        "fixed_rows_present": bool(fixed_rows),
        "adaptive_control_enabled_all_repositories": bool(adaptive_rows)
        and all(row["enabled"] is True for row in adaptive_rows),
        "fixed_control_disabled_all_repositories": bool(fixed_rows)
        and all(row["enabled"] is False for row in fixed_rows),
        "adaptive_failure_sensitivity_events_observed": bool(adaptive_rows)
        and all(
            int(row["failure_sensitivity_adaptations"] or 0) > 0
            for row in adaptive_rows
        ),
        "adaptive_set_point_recovery_events_observed": bool(adaptive_rows)
        and all(
            int(row["set_point_recovery_adaptations"] or 0) > 0
            for row in adaptive_rows
        ),
        "adaptive_thresholds_changed_but_remained_bounded": bool(adaptive_rows)
        and all(
            float(row["minimum_threshold"])
            <= float(row["active_threshold"])
            < float(row["baseline_threshold"])
            for row in adaptive_rows
        ),
        "fixed_thresholds_and_event_counts_unchanged": bool(fixed_rows)
        and all(
            float(row["active_threshold"]) == float(row["baseline_threshold"])
            and int(row["failure_sensitivity_adaptations"] or 0) == 0
            and int(row["set_point_recovery_adaptations"] or 0) == 0
            and not row["last_threshold_adaptation"]
            for row in fixed_rows
        ),
        "adaptive_actions_allowlisted": _all_repo_check(
            adaptive,
            "all_model_selected_actions_were_allowlisted",
        ),
        "fixed_actions_allowlisted": _all_repo_check(
            fixed,
            "all_model_selected_actions_were_allowlisted",
        ),
        "adaptive_completion_predicates_satisfied": _all_repo_check(
            adaptive,
            "all_task_completion_predicates_satisfied",
        ),
        "fixed_completion_predicates_satisfied": _all_repo_check(
            fixed,
            "all_task_completion_predicates_satisfied",
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_online_adaptation_ablation",
        "passed": passed,
        "ready_for_bounded_online_homeostatic_adaptation_claim": passed,
        "ready_for_online_adaptation_performance_gain_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "repositories": len(adaptive_rows),
            "episodes_per_condition": adaptive.get("metrics", {}).get("episodes"),
            "executed_actions_per_condition": adaptive.get("metrics", {}).get(
                "executed_actions"
            ),
            "task_completion_success_rate_per_condition": adaptive.get(
                "metrics", {}
            ).get("task_completion_success_rate"),
            "adaptive_rows": adaptive_rows,
            "fixed_rows": fixed_rows,
        },
        "supported_claims": [
            (
                "visible runtime failures caused bounded online surprise-threshold "
                "adaptation and stable outcomes partially restored the calibrated "
                "set point; disabling only that mechanism removed the internal "
                "dynamics while preserving the same bounded task matrix"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "online adaptation improves task completion on this matrix",
            "long-term cross-episode plasticity",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "online_adaptation_low_error_failure_shift_utility"
            if passed
            else "repair_online_homeostatic_adaptation_ablation"
        ),
        "evidence": {
            "adaptive_report_json": str(adaptive_report_json),
            "fixed_report_json": str(fixed_report_json),
        },
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit bounded online homeostatic adaptation against a fixed-threshold ablation."
    )
    parser.add_argument("--adaptive-report-json", required=True)
    parser.add_argument("--fixed-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_online_adaptation(
        adaptive_report_json=args.adaptive_report_json,
        fixed_report_json=args.fixed_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
