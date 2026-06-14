import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_online_adaptation import (
    audit_phase2homeostasis_online_adaptation,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _runtime_report(*, enabled: bool) -> dict:
    baseline = 0.25
    homeostatic = {
        "config": {
            "online_failure_sensitivity_enabled": enabled,
            "surprise_wake_threshold": baseline,
            "minimum_surprise_wake_threshold": 0.05,
        },
        "active_surprise_wake_threshold": 0.18 if enabled else baseline,
        "failure_sensitivity_adaptations": 3 if enabled else 0,
        "set_point_recovery_adaptations": 2 if enabled else 0,
        "last_threshold_adaptation": (
            {"reason": "stable_outcome_restored_set_point"} if enabled else {}
        ),
    }
    return {
        "passed": True,
        "seed": 7,
        "metrics": {
            "repositories": 1,
            "episodes": 2,
            "executed_actions": 10,
            "task_completion_successes": 2,
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
                            "homeostatic_control": homeostatic,
                        }
                    }
                },
            }
        ],
    }


def test_online_homeostatic_adaptation_audit_accepts_causal_ablation(
    tmp_path: Path,
) -> None:
    report = audit_phase2homeostasis_online_adaptation(
        adaptive_report_json=_write(
            tmp_path / "adaptive.json",
            _runtime_report(enabled=True),
        ),
        fixed_report_json=_write(
            tmp_path / "fixed.json",
            _runtime_report(enabled=False),
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_online_homeostatic_adaptation_claim"] is True
    assert report["ready_for_online_adaptation_performance_gain_claim"] is False


def test_online_homeostatic_adaptation_audit_rejects_inert_adaptive_condition(
    tmp_path: Path,
) -> None:
    adaptive = _runtime_report(enabled=True)
    state = adaptive["repository_reports"][0]["policy_configuration"][
        "policy_metadata"
    ]["expert_policy"]["homeostatic_control"]
    state["active_surprise_wake_threshold"] = 0.25
    state["failure_sensitivity_adaptations"] = 0

    report = audit_phase2homeostasis_online_adaptation(
        adaptive_report_json=_write(tmp_path / "adaptive.json", adaptive),
        fixed_report_json=_write(
            tmp_path / "fixed.json",
            _runtime_report(enabled=False),
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["checks"]["adaptive_failure_sensitivity_events_observed"] is False
