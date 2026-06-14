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


def _memory_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
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
                "preserve_memory": config.get(
                    "preserve_adaptive_threshold_across_reset"
                ),
                "baseline_threshold": config.get("surprise_wake_threshold"),
                "active_threshold": state.get("active_surprise_wake_threshold"),
                "episode_failure_adaptations": state.get(
                    "failure_sensitivity_adaptations"
                ),
                "episode_recovery_adaptations": state.get(
                    "set_point_recovery_adaptations"
                ),
                "lifetime_failure_adaptations": state.get(
                    "lifetime_failure_sensitivity_adaptations"
                ),
                "lifetime_recovery_adaptations": state.get(
                    "lifetime_set_point_recovery_adaptations"
                ),
                "preserved_resets": state.get("adaptive_threshold_preserved_resets"),
                "wake_events": state.get("wake_events"),
            }
        )
    return rows


def _all_repo_check(report: dict[str, Any], name: str) -> bool:
    repositories = report.get("repository_reports", [])
    return bool(repositories) and all(
        row.get("checks", {}).get(name) is True for row in repositories
    )


def audit_phase2homeostasis_package_cross_episode_memory(
    *,
    persistent_report_json: str | Path,
    erased_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    persistent = _read_json(persistent_report_json)
    erased = _read_json(erased_report_json)
    persistent_rows = _memory_rows(persistent)
    erased_rows = _memory_rows(erased)
    checks = {
        "persistent_package_runtime_passed": persistent.get("passed") is True,
        "erased_package_runtime_passed": erased.get("passed") is True,
        "same_seed_and_task_matrix": persistent.get("seed") == erased.get("seed")
        and _repo_signature(persistent) == _repo_signature(erased),
        "same_core_completion_metrics": persistent.get("metrics")
        == erased.get("metrics"),
        "persistent_memory_enabled_all_repositories": bool(persistent_rows)
        and all(row["preserve_memory"] is True for row in persistent_rows),
        "erased_memory_disabled_all_repositories": bool(erased_rows)
        and all(row["preserve_memory"] is False for row in erased_rows),
        "persistent_runtime_accumulated_cross_episode_failure_adaptation": bool(
            persistent_rows
        )
        and all(
            int(row["lifetime_failure_adaptations"] or 0)
            > int(row["episode_failure_adaptations"] or 0)
            for row in persistent_rows
        ),
        "persistent_runtime_accumulated_cross_episode_recovery": bool(
            persistent_rows
        )
        and all(
            int(row["lifetime_recovery_adaptations"] or 0)
            > int(row["episode_recovery_adaptations"] or 0)
            for row in persistent_rows
        ),
        "persistent_runtime_recorded_preserved_resets": bool(persistent_rows)
        and all(int(row["preserved_resets"] or 0) > 0 for row in persistent_rows),
        "erased_runtime_has_no_preserved_resets": bool(erased_rows)
        and all(int(row["preserved_resets"] or 0) == 0 for row in erased_rows),
        "persistent_thresholds_differ_from_erased_thresholds": (
            len(persistent_rows) == len(erased_rows)
            and all(
                float(persistent_row["active_threshold"])
                < float(erased_row["active_threshold"])
                for persistent_row, erased_row in zip(persistent_rows, erased_rows)
            )
        ),
        "persistent_actions_allowlisted": _all_repo_check(
            persistent,
            "all_model_selected_actions_were_allowlisted",
        ),
        "erased_actions_allowlisted": _all_repo_check(
            erased,
            "all_model_selected_actions_were_allowlisted",
        ),
        "persistent_completion_predicates_satisfied": _all_repo_check(
            persistent,
            "all_task_completion_predicates_satisfied",
        ),
        "erased_completion_predicates_satisfied": _all_repo_check(
            erased,
            "all_task_completion_predicates_satisfied",
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_package_cross_episode_memory",
        "passed": passed,
        "ready_for_bounded_package_cross_episode_homeostatic_memory_claim": passed,
        "ready_for_cross_episode_memory_performance_gain_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "repositories": len(persistent_rows),
            "episodes_per_condition": persistent.get("metrics", {}).get("episodes"),
            "executed_actions_per_condition": persistent.get("metrics", {}).get(
                "executed_actions"
            ),
            "task_completion_success_rate_per_condition": persistent.get(
                "metrics", {}
            ).get("task_completion_success_rate"),
            "persistent_rows": persistent_rows,
            "erased_rows": erased_rows,
        },
        "supported_claims": [
            (
                "the package-internal structured runtime preserved bounded adaptive "
                "homeostatic state across real sealed episodes and repositories; "
                "the erased-memory condition removed that accumulation while both "
                "conditions retained allowlisted execution and full completion"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "cross-episode memory improves task completion on this matrix",
            "unbounded long-term memory",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "cross_episode_memory_wake_cost_and_false_positive_stress"
            if passed
            else "repair_package_cross_episode_homeostatic_memory"
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
        description="Audit package runtime cross-episode homeostatic memory."
    )
    parser.add_argument("--persistent-report-json", required=True)
    parser.add_argument("--erased-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_package_cross_episode_memory(
        persistent_report_json=args.persistent_report_json,
        erased_report_json=args.erased_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
