import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_cross_episode_wake_cost import (
    audit_phase2homeostasis_cross_episode_wake_cost,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _report(tmp_path: Path, *, preserve: bool, surprise_wakes: int) -> dict:
    subreport = {
        "episode_reports": [
            {
                "episode_id": "episode",
                "selected_actions": [{"type": "RUN_COMMAND"}, {"type": "DONE"}],
                "policy_debug_steps": [
                    *[
                        {
                            "homeostatic_decision": {
                                "reason": "homeostatic_surprise_wake"
                            }
                        }
                        for _ in range(surprise_wakes)
                    ],
                    {
                        "homeostatic_decision": {
                            "reason": "homeostatic_persistent_failure_wake"
                        }
                    },
                ],
            }
        ]
    }
    subreport_path = _write(
        tmp_path / f"subreport-{preserve}.json",
        subreport,
    )
    return {
        "passed": True,
        "metrics": {
            "repositories": 1,
            "episodes": 1,
            "executed_actions": 2,
            "task_completion_successes": 1,
            "task_completion_success_rate": 1.0,
        },
        "repository_reports": [
            {
                "repository_id": "repo",
                "report_json": str(subreport_path),
                "policy_configuration": {
                    "policy_metadata": {
                        "expert_policy": {
                            "homeostatic_control": {
                                "config": {
                                    "preserve_adaptive_threshold_across_reset": (
                                        preserve
                                    )
                                },
                                "adaptive_threshold_reset_decay_events": (
                                    3 if preserve else 0
                                ),
                            }
                        }
                    }
                },
            }
        ],
    }


def test_cross_episode_wake_cost_audit_accepts_no_added_cost(tmp_path: Path) -> None:
    report = audit_phase2homeostasis_cross_episode_wake_cost(
        persistent_report_json=_write(
            tmp_path / "persistent.json",
            _report(tmp_path, preserve=True, surprise_wakes=1),
        ),
        erased_report_json=_write(
            tmp_path / "erased.json",
            _report(tmp_path, preserve=False, surprise_wakes=1),
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert (
        report[
            "ready_for_bounded_cross_episode_memory_without_added_wake_cost_claim"
        ]
        is True
    )


def test_cross_episode_wake_cost_audit_rejects_added_surprise_wakes(
    tmp_path: Path,
) -> None:
    report = audit_phase2homeostasis_cross_episode_wake_cost(
        persistent_report_json=_write(
            tmp_path / "persistent.json",
            _report(tmp_path, preserve=True, surprise_wakes=2),
        ),
        erased_report_json=_write(
            tmp_path / "erased.json",
            _report(tmp_path, preserve=False, surprise_wakes=1),
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["checks"]["persistent_memory_adds_no_surprise_wake_cost"] is False
