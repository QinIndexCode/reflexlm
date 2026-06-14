import json
from pathlib import Path

from reflexlm.cli.audit_phase2cu_fresh_execution_runtime_perturbation_matrix import (
    DEFAULT_PERTURBATIONS,
)
from reflexlm.cli.audit_phase2dc_composed_grid_cross_order_stability import (
    audit_phase2dc_composed_grid_cross_order_stability,
    validate_phase2dc_composed_grid_cross_order_stability,
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


def _phase2db_fixture(tmp_path: Path) -> Path:
    source_output_dir = tmp_path / "source_da"
    runtimes = [
        "C:\\Python313\\python.exe",
        "D:\\repo\\.venv312\\Scripts\\python.exe",
        "D:\\alias\\Scripts\\python.exe",
    ]
    seeds = [20260608, 20260609, 20260610]
    seed_results = []
    for seed_index, seed in enumerate(seeds):
        suite_json = _write(
            source_output_dir / "suites" / f"phase2da_seed_{seed}.json",
            {"seed": seed, "repositories": [{"repository_id": "repo_a"}]},
        )
        perturbation_results = []
        for perturbation_index, spec in enumerate(DEFAULT_PERTURBATIONS):
            perturbation_id = str(spec["perturbation_id"])
            runtime_results = []
            for runtime_index, runtime in enumerate(runtimes):
                runtime_dir = (
                    source_output_dir
                    / f"seed_{seed_index:02d}_{seed}"
                    / f"{perturbation_index:02d}_{perturbation_id}"
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
                        "subprocess": {"command": [runtime], "returncode": 0},
                        "report_exists": True,
                        "report_passed": True,
                        "all_stress_ids_present": True,
                        "extra_stress_ids_present": True,
                        "failure_recovery_success_rate": 1.0,
                        "signature_matches_reference": True,
                    }
                )
            perturbation_results.append(
                {
                    "perturbation_id": perturbation_id,
                    "timeout_seconds": float(spec["timeout_seconds"]),
                    "max_extra_steps": int(spec["max_extra_steps"]),
                    "runtime_results": runtime_results,
                }
            )
        seed_results.append(
            {
                "seed_index": seed_index,
                "seed": seed,
                "suite_json": str(suite_json),
                "perturbation_results": perturbation_results,
            }
        )
    phase2da = _write(
        tmp_path / "phase2da.json",
        {
            "artifact_family": "phase2da_cross_seed_runtime_perturbation_composition",
            "passed": True,
            "ready_for_bounded_cross_seed_runtime_perturbation_composition_claim": True,
            "ready_for_general_shell_autonomy_claim": False,
            "ready_for_general_runtime_invariance_claim": False,
            "ready_for_open_ended_native_perception_claim": False,
            "ready_for_production_autonomy_claim": False,
            "ready_for_epoch_making_architecture_claim": False,
            "checks": {
                "source_phase2cz_passed": True,
                "minimum_three_seeds_met": True,
                "minimum_three_runtime_interpreters_met": True,
                "minimum_three_perturbations_met": True,
                "all_subprocesses_returned_zero": True,
                "all_runtime_reports_exist": True,
                "all_runtime_reports_passed": True,
                "all_expanded_stress_ids_present": True,
                "all_extra_stress_ids_present": True,
                "all_failure_recovery_rates_are_perfect": True,
                "all_runtime_signatures_stable_across_seed_perturbation_grid": True,
            },
            "metrics": {
                "seed_count": 3,
                "runtime_count": 3,
                "perturbation_count": 3,
                "fresh_runtime_execution_count": 27,
                "passed_runtime_reports": 27,
                "runtime_signature_mismatch_count": 0,
            },
            "seed_results": seed_results,
            "evidence": {"composed_grid_output_dir": str(source_output_dir)},
        },
    )
    return _write(
        tmp_path / "phase2db.json",
        {"passed": True, "evidence": {"phase2da_report_json": str(phase2da)}},
    )


def test_phase2dc_accepts_cross_order_stable_grid(tmp_path: Path) -> None:
    report = audit_phase2dc_composed_grid_cross_order_stability(
        phase2db_report_json=_phase2db_fixture(tmp_path),
        output_dir=tmp_path / "phase2dc",
        output_report_json=tmp_path / "phase2dc.json",
    )
    validation = validate_phase2dc_composed_grid_cross_order_stability(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["metrics"]["order_count"] >= 5
    assert report["metrics"]["order_validation_signature_mismatch_count"] == 0
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_phase2dc_validation_recomputes_and_rejects_order_drift(
    tmp_path: Path,
) -> None:
    report = audit_phase2dc_composed_grid_cross_order_stability(
        phase2db_report_json=_phase2db_fixture(tmp_path),
        output_dir=tmp_path / "phase2dc",
        output_report_json=tmp_path / "phase2dc.json",
    )
    order_report_path = Path(report["order_results"][1]["order_report_json"])
    order_report = json.loads(order_report_path.read_text(encoding="utf-8"))
    order_report["seed_results"][0]["perturbation_results"] = order_report[
        "seed_results"
    ][0]["perturbation_results"][:2]
    order_report_path.write_text(json.dumps(order_report), encoding="utf-8")
    validation = validate_phase2dc_composed_grid_cross_order_stability(report)

    assert validation["passed"] is False
    assert validation["checks"]["all_recomputed_validations_passed"] is False
    assert validation["checks"]["all_recorded_validations_match_recomputed"] is False
    assert validation["checks"]["all_recomputed_order_signatures_match"] is False


def test_phase2dc_validation_rejects_overstated_epoch_claim(tmp_path: Path) -> None:
    report = audit_phase2dc_composed_grid_cross_order_stability(
        phase2db_report_json=_phase2db_fixture(tmp_path),
        output_dir=tmp_path / "phase2dc",
        output_report_json=tmp_path / "phase2dc.json",
    )
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2dc_composed_grid_cross_order_stability(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
