import json
from pathlib import Path

import reflexlm.cli.audit_phase2cw_runtime_perturbation_recovery_stress_expansion as phase2cw_audit
from reflexlm.cli.run_phase2cw_runtime_perturbation_recovery_stress_expansion import (
    PHASE2CW_EXTRA_STRESS_IDS,
    STRESS_IDS,
    _generate_manifest_for_repository,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2cw_manifest_expands_bounded_recovery_stressors(tmp_path: Path) -> None:
    manifest = _generate_manifest_for_repository(
        suite_seed=20260608,
        repository={
            "repository_id": "repo_0",
            "workspace_root": str(tmp_path / "repo_0"),
        },
    )

    stress_ids = {
        episode["generator"]["stress_id"] for episode in manifest["episodes"]
    }
    assert stress_ids == set(STRESS_IDS)
    assert set(PHASE2CW_EXTRA_STRESS_IDS).issubset(stress_ids)
    for episode in manifest["episodes"]:
        run_steps = [
            step
            for step in episode["completion_requirements"]
            if step["action_type"] == "RUN_COMMAND"
        ]
        assert episode["requires_failure"] is True
        assert len(run_steps) == 2
        assert run_steps[0]["expected_exit_code"] != 0
        assert run_steps[1]["expected_exit_code"] == 0
        assert "shell" not in run_steps[0]
        assert "shell" not in run_steps[1]


def _phase2cw_report(runtime: str, *, drift: bool = False) -> dict:
    version = "3.13.2" if "Python313" in runtime else "3.12.10"
    stress_counts = {stress_id: 3 for stress_id in STRESS_IDS}
    if drift:
        stress_counts[PHASE2CW_EXTRA_STRESS_IDS[0]] = 2
    return {
        "artifact_family": "phase2cw_runtime_perturbation_recovery_stress_expansion",
        "passed": not drift,
        "runtime_interpreter": runtime,
        "runtime_environment": {
            "implementation": "CPython",
            "version": version,
            "executable": runtime,
        },
        "stress_ids": list(STRESS_IDS),
        "extra_stress_ids": list(PHASE2CW_EXTRA_STRESS_IDS),
        "checks": {
            "all_repository_runtime_suites_passed": not drift,
            "all_stress_families_present": not drift,
            "extra_stress_families_present": not drift,
        },
        "metrics": {
            "repositories": 3,
            "stress_counts": stress_counts,
            "episodes": 15,
            "executed_actions": 75,
            "task_completion_successes": 15 if not drift else 14,
            "task_completion_success_rate": 1.0 if not drift else 14 / 15,
            "failure_episodes": 15,
            "observed_failures": 15,
            "observed_recoveries_after_failure": 15 if not drift else 14,
            "failure_recovery_success_rate": 1.0 if not drift else 14 / 15,
        },
        "ready_for_bounded_expanded_recovery_stress_claim": not drift,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
    }


def _phase2cv_fixture(tmp_path: Path) -> Path:
    phase2cp = _write(
        tmp_path / "phase2cp.json",
        {
            "metrics": {
                "runtime_paths": [
                    "C:\\Python313\\python.exe",
                    "D:\\repo\\.venv312\\Scripts\\python.exe",
                    "D:\\alias\\Scripts\\python.exe",
                ]
            },
            "evidence": {"package_build_report_json": str(tmp_path / "build.json")},
        },
    )
    _write(
        tmp_path / "build.json",
        {"passed": True, "package_path": str(tmp_path / "package")},
    )
    phase2cs = _write(
        tmp_path / "phase2cs.json",
        {
            "evidence": {
                "phase2cp_report_json": str(phase2cp),
                "suite_json": str(tmp_path / "suite.json"),
            }
        },
    )
    phase2cu = _write(
        tmp_path / "phase2cu.json",
        {
            "passed": True,
            "evidence": {
                "phase2cs_report_json": str(phase2cs),
                "phase2cp_report_json": str(phase2cp),
                "suite_json": str(tmp_path / "suite.json"),
            },
        },
    )
    return _write(
        tmp_path / "phase2cv.json",
        {"passed": True, "evidence": {"phase2cu_report_json": str(phase2cu)}},
    )


def test_phase2cw_accepts_stable_expanded_stress_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_run(**kwargs):
        _write(
            Path(kwargs["output_report_json"]),
            _phase2cw_report(kwargs["runtime_interpreter"]),
        )
        return {"command": [kwargs["runtime_interpreter"]], "returncode": 0}

    monkeypatch.setattr(phase2cw_audit, "_run_phase2cw_subprocess", _fake_run)
    report = phase2cw_audit.audit_phase2cw_runtime_perturbation_recovery_stress_expansion(
        phase2cv_report_json=_phase2cv_fixture(tmp_path),
        output_dir=tmp_path / "cw",
        output_report_json=tmp_path / "phase2cw.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["fresh_runtime_execution_count"] == 9
    assert report["metrics"]["runtime_signature_mismatch_count"] == 0
    assert set(report["metrics"]["extra_stress_ids"]) == set(PHASE2CW_EXTRA_STRESS_IDS)
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_phase2cw_rejects_expanded_stress_signature_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = {"count": 0}

    def _fake_run(**kwargs):
        _write(
            Path(kwargs["output_report_json"]),
            _phase2cw_report(
                kwargs["runtime_interpreter"],
                drift=calls["count"] == 4,
            ),
        )
        calls["count"] += 1
        return {"command": [kwargs["runtime_interpreter"]], "returncode": 0}

    monkeypatch.setattr(phase2cw_audit, "_run_phase2cw_subprocess", _fake_run)
    report = phase2cw_audit.audit_phase2cw_runtime_perturbation_recovery_stress_expansion(
        phase2cv_report_json=_phase2cv_fixture(tmp_path),
        output_dir=tmp_path / "cw",
        output_report_json=tmp_path / "phase2cw.json",
    )

    assert report["passed"] is False
    assert report["checks"]["all_runtime_signatures_stable_across_perturbations"] is False
    assert report["metrics"]["runtime_signature_mismatch_count"] == 1
