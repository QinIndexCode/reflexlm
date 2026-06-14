import json
from pathlib import Path

import reflexlm.cli.audit_phase2da_cross_seed_runtime_perturbation_composition as phase2da_audit
from reflexlm.cli.audit_phase2cu_fresh_execution_runtime_perturbation_matrix import (
    DEFAULT_PERTURBATIONS,
)
from reflexlm.cli.run_phase2cw_runtime_perturbation_recovery_stress_expansion import (
    PHASE2CW_EXTRA_STRESS_IDS,
    STRESS_IDS,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _runtime_report(
    runtime: str,
    *,
    seed: int,
    drift: bool = False,
) -> dict:
    version = "3.13.2" if "Python313" in runtime else "3.12.10"
    return {
        "artifact_family": "phase2cw_runtime_perturbation_recovery_stress_expansion",
        "passed": True,
        "runtime_interpreter": runtime,
        "runtime_environment": {
            "implementation": "CPython",
            "version": version,
            "version_info": [int(part) for part in version.split(".")],
            "executable": runtime,
        },
        "seed": seed,
        "stress_ids": list(STRESS_IDS),
        "extra_stress_ids": list(PHASE2CW_EXTRA_STRESS_IDS),
        "generated_contract_signatures": [f"seed-dependent-{seed}"],
        "checks": {
            "all_repository_runtime_suites_passed": True,
            "all_stress_families_present": True,
            "extra_stress_families_present": True,
        },
        "metrics": {
            "repositories": 3,
            "stress_counts": {stress_id: 3 for stress_id in STRESS_IDS},
            "episodes": 15,
            "executed_actions": 75 + (1 if drift else 0),
            "task_completion_successes": 15,
            "task_completion_success_rate": 1.0,
            "failure_episodes": 15,
            "observed_failures": 15,
            "observed_recoveries_after_failure": 15,
            "failure_recovery_success_rate": 1.0,
        },
        "ready_for_bounded_expanded_recovery_stress_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
    }


def _phase2cz_fixture(tmp_path: Path) -> Path:
    suite = _write(
        tmp_path / "suite.json",
        {
            "source_repository_root": str(tmp_path),
            "seed": 20260608,
            "minimum_repository_count": 3,
            "repositories": [
                {
                    "repository_id": "repo_a",
                    "workspace_root": str(tmp_path / "repo_a"),
                },
                {
                    "repository_id": "repo_b",
                    "workspace_root": str(tmp_path / "repo_b"),
                },
                {
                    "repository_id": "repo_c",
                    "workspace_root": str(tmp_path / "repo_c"),
                },
            ],
        },
    )
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
                "package_build_report_json": str(tmp_path / "build.json"),
                "suite_json": str(suite),
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
                "suite_json": str(suite),
            },
        },
    )
    phase2cv = _write(
        tmp_path / "phase2cv.json",
        {"passed": True, "evidence": {"phase2cu_report_json": str(phase2cu)}},
    )
    phase2cw = _write(
        tmp_path / "phase2cw.json",
        {
            "passed": True,
            "evidence": {
                "phase2cv_report_json": str(phase2cv),
                "expanded_recovery_output_dir": str(tmp_path / "phase2cw_output"),
            },
        },
    )
    phase2cx = _write(
        tmp_path / "phase2cx.json",
        {"passed": True, "evidence": {"phase2cw_report_json": str(phase2cw)}},
    )
    phase2cy = _write(
        tmp_path / "phase2cy.json",
        {"passed": True, "evidence": {"phase2cx_report_json": str(phase2cx)}},
    )
    return _write(
        tmp_path / "phase2cz.json",
        {"passed": True, "evidence": {"phase2cy_report_json": str(phase2cy)}},
    )


def test_phase2da_accepts_composed_seed_perturbation_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_run(**kwargs):
        suite = json.loads(Path(kwargs["suite_json"]).read_text(encoding="utf-8"))
        _write(
            Path(kwargs["output_report_json"]),
            _runtime_report(kwargs["runtime_interpreter"], seed=suite["seed"]),
        )
        return {"command": [kwargs["runtime_interpreter"]], "returncode": 0}

    monkeypatch.setattr(phase2da_audit, "_run_phase2cw_subprocess", _fake_run)
    report = phase2da_audit.audit_phase2da_cross_seed_runtime_perturbation_composition(
        phase2cz_report_json=_phase2cz_fixture(tmp_path),
        output_dir=tmp_path / "phase2da",
        output_report_json=tmp_path / "phase2da.json",
        seeds=[20260608, 20260609, 20260610],
    )
    validation = (
        phase2da_audit.validate_phase2da_cross_seed_runtime_perturbation_composition(
            report
        )
    )

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["metrics"]["seed_count"] == 3
    assert report["metrics"]["perturbation_count"] == 3
    assert report["metrics"]["fresh_runtime_execution_count"] == 27
    assert report["metrics"]["runtime_signature_mismatch_count"] == 0
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_phase2da_rejects_composed_grid_signature_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_run(**kwargs):
        suite = json.loads(Path(kwargs["suite_json"]).read_text(encoding="utf-8"))
        drift = (
            suite["seed"] == 20260609
            and kwargs["runtime_interpreter"] == "C:\\Python313\\python.exe"
            and float(kwargs["timeout_seconds"]) == 7.5
        )
        _write(
            Path(kwargs["output_report_json"]),
            _runtime_report(
                kwargs["runtime_interpreter"],
                seed=suite["seed"],
                drift=drift,
            ),
        )
        return {"command": [kwargs["runtime_interpreter"]], "returncode": 0}

    monkeypatch.setattr(phase2da_audit, "_run_phase2cw_subprocess", _fake_run)
    report = phase2da_audit.audit_phase2da_cross_seed_runtime_perturbation_composition(
        phase2cz_report_json=_phase2cz_fixture(tmp_path),
        output_dir=tmp_path / "phase2da",
        output_report_json=tmp_path / "phase2da.json",
        seeds=[20260608, 20260609, 20260610],
    )
    validation = (
        phase2da_audit.validate_phase2da_cross_seed_runtime_perturbation_composition(
            report
        )
    )

    assert report["passed"] is False
    assert validation["passed"] is False
    assert (
        report["checks"][
            "all_runtime_signatures_stable_across_seed_perturbation_grid"
        ]
        is False
    )
    assert (
        validation["checks"][
            "all_runtime_signatures_stable_across_seed_perturbation_grid"
        ]
        is False
    )
    assert report["metrics"]["runtime_signature_mismatch_count"] == 1


def test_phase2da_rejects_perturbation_coverage_collapse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_run(**kwargs):
        suite = json.loads(Path(kwargs["suite_json"]).read_text(encoding="utf-8"))
        _write(
            Path(kwargs["output_report_json"]),
            _runtime_report(kwargs["runtime_interpreter"], seed=suite["seed"]),
        )
        return {"command": [kwargs["runtime_interpreter"]], "returncode": 0}

    monkeypatch.setattr(phase2da_audit, "_run_phase2cw_subprocess", _fake_run)
    report = phase2da_audit.audit_phase2da_cross_seed_runtime_perturbation_composition(
        phase2cz_report_json=_phase2cz_fixture(tmp_path),
        output_dir=tmp_path / "phase2da",
        output_report_json=tmp_path / "phase2da.json",
        seeds=[20260608, 20260609, 20260610],
        perturbations=list(DEFAULT_PERTURBATIONS[:2]),
    )
    validation = (
        phase2da_audit.validate_phase2da_cross_seed_runtime_perturbation_composition(
            report
        )
    )

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["minimum_three_perturbations_met"] is False
    assert validation["checks"]["minimum_three_perturbations_recorded"] is False
