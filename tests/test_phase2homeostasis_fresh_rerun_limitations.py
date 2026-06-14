import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_fresh_rerun_limitations import (
    audit_phase2homeostasis_fresh_rerun_limitations,
    validate_phase2homeostasis_fresh_rerun_limitations,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _runtime_report(
    tmp_path: Path,
    *,
    name: str,
    executable: str,
    command_tail: str = "-m pytest -q",
) -> Path:
    subreport = _write(
        tmp_path / f"{name}-subreport.json",
        {
            "episode_reports": [
                {
                    "episode_id": "episode-1",
                    "selected_actions": [
                        {
                            "type": "RUN_COMMAND",
                            "command": f"{executable} {command_tail}",
                            "file_target": None,
                            "reason": "verify bounded runtime behavior",
                        }
                    ],
                }
            ]
        },
    )
    return _write(
        tmp_path / f"{name}.json",
        {
            "passed": True,
            "runtime_interpreter": executable,
            "runtime_environment": {
                "executable": executable,
                "version": "3.13.2" if "313" in executable else "3.12.10",
            },
            "metrics": {
                "repositories": 1,
                "episodes": 1,
                "executed_actions": 1,
                "task_completion_success_rate": 1.0,
            },
            "repository_reports": [
                {
                    "repository_id": "repo",
                    "report_json": str(subreport),
                }
            ],
        },
    )


def _chain_report(tmp_path: Path, *, name: str, passed: bool = True) -> Path:
    return _write(
        tmp_path / f"{name}.json",
        {
            "passed": passed,
            "checks": {"side_effect_action_traces_match": passed},
        },
    )


def _cross_runtime_report(
    tmp_path: Path,
    *,
    passed: bool = False,
    maximum_active_threshold_delta: float = 0.008,
) -> Path:
    return _write(
        tmp_path / "cross-runtime.json",
        {
            "passed": passed,
            "metrics": {
                "maximum_active_threshold_delta": maximum_active_threshold_delta
            },
        },
    )


def _phase2cj_report(tmp_path: Path, *, passed: bool = True) -> Path:
    return _write(tmp_path / "phase2cj.json", {"passed": passed})


def _audit(
    tmp_path: Path,
    *,
    py312_command_tail: str = "-m pytest -q",
    cross_runtime_passed: bool = False,
    threshold_delta: float = 0.008,
) -> dict:
    return audit_phase2homeostasis_fresh_rerun_limitations(
        generation1_report_json=_runtime_report(
            tmp_path,
            name="generation1-py313",
            executable="C:/Python313/python.exe",
        ),
        generation2_py313_report_json=_runtime_report(
            tmp_path,
            name="generation2-py313",
            executable="C:/Python313/python.exe",
        ),
        generation2_py312_report_json=_runtime_report(
            tmp_path,
            name="generation2-py312",
            executable="D:/Python312/python.exe",
            command_tail=py312_command_tail,
        ),
        chain_py313_report_json=_chain_report(tmp_path, name="chain-py313"),
        chain_py313_to_py312_report_json=_chain_report(
            tmp_path,
            name="chain-py313-to-py312",
        ),
        cross_runtime_dynamics_report_json=_cross_runtime_report(
            tmp_path,
            passed=cross_runtime_passed,
            maximum_active_threshold_delta=threshold_delta,
        ),
        phase2cj_report_json=_phase2cj_report(tmp_path),
        output_report_json=tmp_path / "limitations.json",
    )


def test_fresh_rerun_limitations_accepts_bounded_negative_evidence(
    tmp_path: Path,
) -> None:
    report = _audit(tmp_path)
    validation = validate_phase2homeostasis_fresh_rerun_limitations(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["ready_for_bounded_homeostasis_fresh_behavioral_claim"] is True
    assert report["ready_for_exact_cross_runtime_homeostatic_dynamics_claim"] is False
    assert (
        report["limitations"]["exact_cross_runtime_homeostatic_dynamics"]
        == "not_supported_by_this_fresh_rerun"
    )


def test_fresh_rerun_limitations_rejects_unbounded_threshold_drift(
    tmp_path: Path,
) -> None:
    report = _audit(tmp_path, threshold_delta=0.011)
    validation = validate_phase2homeostasis_fresh_rerun_limitations(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["fresh_threshold_drift_bounded_for_limitation"] is False


def test_fresh_rerun_limitations_rejects_side_effect_trace_mismatch(
    tmp_path: Path,
) -> None:
    report = _audit(tmp_path, py312_command_tail="-m unittest")
    validation = validate_phase2homeostasis_fresh_rerun_limitations(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["fresh_cross_runtime_side_effect_traces_match"] is False


def test_fresh_rerun_limitations_rejects_claiming_exact_cross_runtime_pass(
    tmp_path: Path,
) -> None:
    report = _audit(tmp_path, cross_runtime_passed=True)
    validation = validate_phase2homeostasis_fresh_rerun_limitations(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["fresh_cross_runtime_exact_dynamics_not_claimed"] is False


def test_fresh_rerun_limitations_validation_rejects_epoch_claim(
    tmp_path: Path,
) -> None:
    report = _audit(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2homeostasis_fresh_rerun_limitations(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
