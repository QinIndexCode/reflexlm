import json
from pathlib import Path

from reflexlm.cli.audit_phase2ct_fresh_runtime_execution_negative_controls import (
    audit_phase2ct_fresh_runtime_execution_negative_controls,
)
from reflexlm.cli.run_phase2co_environment_stress_with_failure_recovery import (
    STRESS_IDS,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _runtime_report(
    *,
    output_dir: Path,
    runtime: str,
    version: str,
) -> dict:
    return {
        "artifact_family": "phase2co_environment_stress_with_failure_recovery",
        "passed": True,
        "seed": 20260608,
        "runtime_interpreter": runtime,
        "runtime_environment": {
            "implementation": "CPython",
            "version": version,
            "executable": runtime,
        },
        "stress_ids": list(STRESS_IDS),
        "generated_manifest_dir": str(output_dir / "generated_manifests"),
        "generated_contract_signatures": ["bounded-contract"],
        "checks": {
            "all_repository_runtime_suites_passed": True,
            "all_repository_actions_were_allowlisted": True,
            "all_repository_task_completion_predicates_satisfied": True,
            "all_repositories_used_package_internal_runtime_cortex": True,
        },
        "metrics": {
            "repositories": 3,
            "stress_counts": {stress_id: 3 for stress_id in STRESS_IDS},
            "episodes": 9,
            "executed_actions": 45,
            "task_completion_successes": 9,
            "task_completion_success_rate": 1.0,
            "failure_episodes": 9,
            "observed_failures": 9,
            "observed_recoveries_after_failure": 9,
            "failure_recovery_success_rate": 1.0,
        },
        "ready_for_bounded_environment_stress_failure_recovery_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
    }


def _matrix_audit(runtime_specs: list[tuple[str, str]]) -> dict:
    return {
        "artifact_family": "phase2cp_cross_runtime_environment_stress_recovery_matrix",
        "passed": True,
        "ready_for_bounded_cross_runtime_environment_stress_recovery_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": {
            "minimum_three_runtime_reports_met": True,
            "all_reports_are_phase2co_family": True,
            "all_runtime_reports_passed": True,
        },
        "metrics": {
            "runtime_paths": [runtime for runtime, _ in runtime_specs],
            "python_versions": [version for _, version in runtime_specs],
            "distinct_runtime_paths": 3,
            "distinct_python_versions": 2,
            "episodes_per_runtime": 9,
            "executed_actions_per_runtime": 45,
            "failure_recovery_success_rate_per_runtime": 1.0,
        },
        "supported_claims": ["bounded matrix"],
        "unsupported_claims": ["epoch-making architecture"],
        "next_required_experiment": "phase2cq_cross_runtime_stress_negative_controls",
    }


def _phase2cs_fixture(tmp_path: Path) -> Path:
    source_output_dir = tmp_path / "source_fresh"
    runtime_specs = [
        ("C:\\Python313\\python.exe", "3.13.2"),
        ("D:\\repo\\.venv312\\Scripts\\python.exe", "3.12.10"),
        ("D:\\alias\\Scripts\\python.exe", "3.12.10"),
    ]
    repetition_results = []
    for repetition_index in range(2):
        repetition_dir = source_output_dir / f"repetition_{repetition_index:02d}"
        runtime_results = []
        for runtime_index, (runtime, version) in enumerate(runtime_specs):
            runtime_dir = repetition_dir / f"runtime_{runtime_index:02d}"
            report_json = _write(
                runtime_dir / "phase2co_report.json",
                _runtime_report(
                    output_dir=runtime_dir,
                    runtime=runtime,
                    version=version,
                ),
            )
            runtime_results.append(
                {
                    "runtime_index": runtime_index,
                    "runtime_interpreter": runtime,
                    "report_json": str(report_json),
                    "output_dir": str(runtime_dir),
                    "subprocess": {"command": [runtime, "-m", "phase2co"], "returncode": 0},
                    "report_exists": True,
                    "report_passed": True,
                    "report_under_repetition_dir": True,
                    "generated_manifest_dir_under_repetition_dir": True,
                    "signature_matches_first_repetition_for_runtime": True,
                }
            )
        matrix_json = _write(
            repetition_dir / "phase2cp_fresh_execution_audit.json",
            _matrix_audit(runtime_specs),
        )
        repetition_results.append(
            {
                "repetition_index": repetition_index,
                "matrix_audit_json": str(matrix_json),
                "matrix_passed": True,
                "runtime_results": runtime_results,
            }
        )
    report = {
        "artifact_family": "phase2cs_fresh_runtime_execution_repetition_stability",
        "passed": True,
        "ready_for_fresh_runtime_execution_repetition_stability_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": {
            "source_phase2cp_passed": True,
            "minimum_three_runtime_interpreters_met": True,
            "minimum_two_repetitions_met": True,
            "all_subprocesses_returned_zero": True,
            "all_fresh_runtime_reports_exist": True,
            "all_fresh_runtime_reports_passed": True,
            "all_repetition_matrix_audits_passed": True,
            "all_repetition_matrix_signatures_match": True,
            "all_runtime_signatures_match_first_repetition": True,
            "bounded_claim_true_only_for_all_repetitions": True,
        },
        "metrics": {
            "runtime_count": 3,
            "repetition_count": 2,
            "fresh_runtime_execution_count": 6,
            "passed_runtime_reports": 6,
            "passed_matrix_audits": 2,
            "matrix_signature_mismatch_count": 0,
            "runtime_signature_mismatch_count": 0,
            "runtime_interpreters": [runtime for runtime, _ in runtime_specs],
        },
        "repetition_results": repetition_results,
        "evidence": {
            "fresh_execution_output_dir": str(source_output_dir),
            "phase2cp_report_json": str(tmp_path / "phase2cp.json"),
        },
    }
    return _write(tmp_path / "phase2cs.json", report)


def test_phase2ct_rejects_fresh_execution_negative_controls(tmp_path: Path) -> None:
    report = audit_phase2ct_fresh_runtime_execution_negative_controls(
        phase2cs_report_json=_phase2cs_fixture(tmp_path),
        output_dir=tmp_path / "controls",
        output_report_json=tmp_path / "phase2ct.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 8
    assert all(
        row["expected_failed_checks_observed"]
        for row in report["control_results"]
    )
