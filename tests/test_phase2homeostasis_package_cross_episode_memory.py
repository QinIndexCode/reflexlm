import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_package_cross_episode_memory import (
    audit_phase2homeostasis_package_cross_episode_memory,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _report(*, preserve: bool) -> dict:
    state = {
        "config": {
            "preserve_adaptive_threshold_across_reset": preserve,
            "surprise_wake_threshold": 0.25,
        },
        "active_surprise_wake_threshold": 0.12 if preserve else 0.18,
        "failure_sensitivity_adaptations": 2,
        "set_point_recovery_adaptations": 2,
        "lifetime_failure_sensitivity_adaptations": 12 if preserve else 2,
        "lifetime_set_point_recovery_adaptations": 20 if preserve else 2,
        "adaptive_threshold_preserved_resets": 6 if preserve else 0,
        "wake_events": 2,
    }
    return {
        "passed": True,
        "seed": 7,
        "metrics": {
            "repositories": 1,
            "episodes": 6,
            "executed_actions": 38,
            "task_completion_successes": 6,
            "task_completion_success_rate": 1.0,
        },
        "repository_reports": [
            {
                "repository_id": "repo",
                "provenance": {"origin": "https://example/repo", "head": "abc"},
                "recipe_ids": ["failure_recovery"],
                "contract_signatures": ["bounded"],
                "checks": {
                    "all_model_selected_actions_were_allowlisted": True,
                    "all_task_completion_predicates_satisfied": True,
                },
                "policy_configuration": {
                    "policy_metadata": {
                        "expert_policy": {
                            "homeostatic_control": state,
                        }
                    }
                },
            }
        ],
    }


def test_package_cross_episode_memory_audit_accepts_persistent_erased_pair(
    tmp_path: Path,
) -> None:
    report = audit_phase2homeostasis_package_cross_episode_memory(
        persistent_report_json=_write(
            tmp_path / "persistent.json",
            _report(preserve=True),
        ),
        erased_report_json=_write(
            tmp_path / "erased.json",
            _report(preserve=False),
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert (
        report["ready_for_bounded_package_cross_episode_homeostatic_memory_claim"]
        is True
    )
    assert report["ready_for_cross_episode_memory_performance_gain_claim"] is False


def test_package_cross_episode_memory_audit_rejects_inert_persistence(
    tmp_path: Path,
) -> None:
    persistent = _report(preserve=True)
    state = persistent["repository_reports"][0]["policy_configuration"][
        "policy_metadata"
    ]["expert_policy"]["homeostatic_control"]
    state["lifetime_failure_sensitivity_adaptations"] = 2

    report = audit_phase2homeostasis_package_cross_episode_memory(
        persistent_report_json=_write(tmp_path / "persistent.json", persistent),
        erased_report_json=_write(
            tmp_path / "erased.json",
            _report(preserve=False),
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert (
        report["checks"][
            "persistent_runtime_accumulated_cross_episode_failure_adaptation"
        ]
        is False
    )
