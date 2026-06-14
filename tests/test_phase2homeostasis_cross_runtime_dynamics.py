import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_cross_runtime_dynamics import (
    audit_phase2homeostasis_cross_runtime_dynamics,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _report(tmp_path: Path, *, name: str, executable: str, threshold: float) -> dict:
    subreport = _write(
        tmp_path / f"{name}-subreport.json",
        {
            "episode_reports": [
                {
                    "episode_id": "episode-1",
                    "selected_actions": [
                        {
                            "type": "RUN_COMMAND",
                            "command": f"{executable} -m pytest -q",
                            "file_target": None,
                            "reason": "verify",
                            "confidence": threshold,
                        }
                    ],
                    "policy_debug_steps": [
                        {
                            "homeostatic_decision": {
                                "reason": "homeostatic_surprise_wake"
                            }
                        }
                    ],
                }
            ]
        },
    )
    state = {
        "config": {
            "preserve_adaptive_threshold_across_reset": True,
            "surprise_wake_threshold": 0.25,
        },
        "active_surprise_wake_threshold": threshold,
        "lifetime_failure_sensitivity_adaptations": 12,
        "lifetime_set_point_recovery_adaptations": 18,
        "adaptive_threshold_preserved_resets": 6,
        "adaptive_threshold_reset_decay_events": 5,
        "wake_events": 2,
    }
    return {
        "passed": True,
        "seed": 7,
        "runtime_environment": {"executable": executable},
        "metrics": {
            "repositories": 1,
            "episodes": 1,
            "executed_actions": 1,
            "task_completion_successes": 1,
            "task_completion_success_rate": 1.0,
        },
        "repository_reports": [
            {
                "repository_id": "repo",
                "provenance": {"origin": "https://example/repo", "head": "abc"},
                "recipe_ids": ["failure_recovery"],
                "contract_signatures": ["bounded"],
                "report_json": str(subreport),
                "policy_configuration": {
                    "policy_metadata": {
                        "expert_policy": {"homeostatic_control": state}
                    }
                },
            }
        ],
    }


def test_cross_runtime_dynamics_accepts_bounded_numeric_drift(tmp_path: Path) -> None:
    canonical = _report(
        tmp_path,
        name="canonical",
        executable="C:/Python313/python.exe",
        threshold=0.1496,
    )
    alternate = _report(
        tmp_path,
        name="alternate",
        executable="D:/Python312/python.exe",
        threshold=0.1497,
    )

    report = audit_phase2homeostasis_cross_runtime_dynamics(
        canonical_report_json=_write(tmp_path / "canonical.json", canonical),
        alternate_report_json=_write(tmp_path / "alternate.json", alternate),
        output_report_json=tmp_path / "audit.json",
        threshold_tolerance=0.001,
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_cross_runtime_homeostatic_dynamics_claim"] is True
    assert report["metrics"]["maximum_active_threshold_delta"] < 0.001


def test_cross_runtime_dynamics_rejects_discrete_state_mismatch(tmp_path: Path) -> None:
    canonical = _report(
        tmp_path,
        name="canonical",
        executable="C:/Python313/python.exe",
        threshold=0.1496,
    )
    alternate = _report(
        tmp_path,
        name="alternate",
        executable="D:/Python312/python.exe",
        threshold=0.1497,
    )
    state = alternate["repository_reports"][0]["policy_configuration"][
        "policy_metadata"
    ]["expert_policy"]["homeostatic_control"]
    state["lifetime_failure_sensitivity_adaptations"] = 11

    report = audit_phase2homeostasis_cross_runtime_dynamics(
        canonical_report_json=_write(tmp_path / "canonical.json", canonical),
        alternate_report_json=_write(tmp_path / "alternate.json", alternate),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["checks"]["discrete_homeostatic_dynamics_match"] is False
