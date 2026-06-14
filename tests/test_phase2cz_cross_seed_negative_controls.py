import json
from pathlib import Path

from reflexlm.cli.audit_phase2cz_cross_seed_negative_controls import (
    audit_phase2cz_cross_seed_negative_controls,
)
from reflexlm.cli.run_phase2cw_runtime_perturbation_recovery_stress_expansion import (
    PHASE2CW_EXTRA_STRESS_IDS,
    STRESS_IDS,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _runtime_report(runtime: str, *, seed: int) -> dict:
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
            "executed_actions": 75,
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


def _phase2cy_fixture(tmp_path: Path) -> Path:
    source_output_dir = tmp_path / "source_cy"
    runtimes = [
        "C:\\Python313\\python.exe",
        "D:\\repo\\.venv312\\Scripts\\python.exe",
        "D:\\alias\\Scripts\\python.exe",
    ]
    seeds = [20260608, 20260609, 20260610]
    seed_results = []
    for seed_index, seed in enumerate(seeds):
        suite_json = _write(
            source_output_dir / "suites" / f"phase2cy_seed_{seed}.json",
            {
                "seed": seed,
                "repositories": [
                    {"repository_id": "repo_a"},
                    {"repository_id": "repo_b"},
                    {"repository_id": "repo_c"},
                ],
            },
        )
        runtime_results = []
        for runtime_index, runtime in enumerate(runtimes):
            runtime_dir = (
                source_output_dir
                / f"seed_{seed_index:02d}_{seed}"
                / "00_baseline_budget"
                / f"runtime_{runtime_index:02d}"
            )
            report_json = _write(
                runtime_dir / "phase2cw_report.json",
                _runtime_report(runtime, seed=seed),
            )
            runtime_results.append(
                {
                    "seed": seed,
                    "runtime_index": runtime_index,
                    "runtime_interpreter": runtime,
                    "report_json": str(report_json),
                    "output_dir": str(runtime_dir),
                    "subprocess": {"command": [runtime, "-m", "phase2cw"], "returncode": 0},
                    "report_exists": True,
                    "report_passed": True,
                    "all_stress_ids_present": True,
                    "extra_stress_ids_present": True,
                    "failure_recovery_success_rate": 1.0,
                    "signature_matches_reference": True,
                }
            )
        seed_results.append(
            {
                "seed_index": seed_index,
                "seed": seed,
                "suite_json": str(suite_json),
                "perturbation_results": [
                    {
                        "perturbation_id": "baseline_budget",
                        "timeout_seconds": 5.0,
                        "max_extra_steps": 5,
                        "runtime_results": runtime_results,
                    }
                ],
            }
        )
    report = {
        "artifact_family": "phase2cy_expanded_recovery_cross_seed_stability",
        "passed": True,
        "ready_for_bounded_expanded_recovery_cross_seed_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": {
            "source_phase2cx_passed": True,
            "minimum_three_seeds_met": True,
            "minimum_three_runtime_interpreters_met": True,
            "minimum_one_perturbation_met": True,
            "all_subprocesses_returned_zero": True,
            "all_runtime_reports_exist": True,
            "all_runtime_reports_passed": True,
            "all_expanded_stress_ids_present": True,
            "all_extra_stress_ids_present": True,
            "all_failure_recovery_rates_are_perfect": True,
            "all_runtime_signatures_stable_across_seeds": True,
        },
        "metrics": {
            "seed_count": 3,
            "runtime_count": 3,
            "perturbation_count": 1,
            "fresh_runtime_execution_count": 9,
            "passed_runtime_reports": 9,
            "runtime_signature_mismatch_count": 0,
            "seeds": seeds,
            "stress_ids": list(STRESS_IDS),
            "extra_stress_ids": list(PHASE2CW_EXTRA_STRESS_IDS),
        },
        "seed_results": seed_results,
        "evidence": {
            "phase2cx_report_json": str(tmp_path / "phase2cx.json"),
            "cross_seed_output_dir": str(source_output_dir),
        },
    }
    return _write(tmp_path / "phase2cy.json", report)


def test_phase2cz_rejects_cross_seed_negative_controls(tmp_path: Path) -> None:
    report = audit_phase2cz_cross_seed_negative_controls(
        phase2cy_report_json=_phase2cy_fixture(tmp_path),
        output_dir=tmp_path / "controls",
        output_report_json=tmp_path / "phase2cz.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 12
    assert all(
        row["expected_failed_checks_observed"]
        for row in report["control_results"]
    )
