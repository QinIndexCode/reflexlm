import json
from pathlib import Path

from reflexlm.cli.audit_phase2cv_runtime_perturbation_negative_controls import (
    audit_phase2cv_runtime_perturbation_negative_controls,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _matrix_audit(*, drift: bool = False) -> dict:
    return {
        "passed": True,
        "ready_for_bounded_cross_runtime_environment_stress_recovery_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": {"all_runtime_reports_passed": True},
        "metrics": {
            "runtime_paths": [
                "C:\\Python313\\python.exe",
                "D:\\repo\\.venv312\\Scripts\\python.exe",
                "D:\\alias\\Scripts\\python.exe",
            ],
            "python_versions": ["3.13.2", "3.12.10", "3.12.10"],
            "distinct_runtime_paths": 3,
            "distinct_python_versions": 1 if drift else 2,
            "episodes_per_runtime": 9,
        },
        "supported_claims": ["bounded matrix"],
        "unsupported_claims": ["epoch-making architecture"],
        "next_required_experiment": "phase2cq_cross_runtime_stress_negative_controls",
    }


def _phase2cs_report(output_dir: Path, *, drift: bool = False) -> dict:
    repetitions = []
    for repetition_index in range(2):
        matrix_json = _write(
            output_dir
            / f"repetition_{repetition_index:02d}"
            / "phase2cp_fresh_execution_audit.json",
            _matrix_audit(drift=drift),
        )
        repetitions.append(
            {
                "repetition_index": repetition_index,
                "matrix_audit_json": str(matrix_json),
                "matrix_passed": True,
                "runtime_results": [],
            }
        )
    return {
        "artifact_family": "phase2cs_fresh_runtime_execution_repetition_stability",
        "passed": True,
        "ready_for_fresh_runtime_execution_repetition_stability_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": {"all_repetition_matrix_audits_passed": True},
        "metrics": {
            "runtime_count": 3,
            "repetition_count": 2,
            "fresh_runtime_execution_count": 6,
            "passed_runtime_reports": 6,
            "passed_matrix_audits": 2,
            "matrix_signature_mismatch_count": 0,
            "runtime_signature_mismatch_count": 0,
            "runtime_interpreters": [
                "C:\\Python313\\python.exe",
                "D:\\repo\\.venv312\\Scripts\\python.exe",
                "D:\\alias\\Scripts\\python.exe",
            ],
        },
        "repetition_results": repetitions,
    }


def _phase2cu_fixture(tmp_path: Path) -> Path:
    source_output_dir = tmp_path / "source_cu"
    perturbation_results = []
    for index, perturbation_id in enumerate(
        ("baseline_budget", "extended_timeout_budget", "extended_step_budget")
    ):
        perturbation_dir = source_output_dir / f"{index:02d}_{perturbation_id}"
        phase2cs_report_json = _write(
            perturbation_dir / "phase2cs_fresh_execution_report.json",
            _phase2cs_report(perturbation_dir / "fresh_execution"),
        )
        validation_json = _write(
            perturbation_dir / "phase2cs_validation.json",
            {"passed": True, "checks": {"ok": True}, "metrics": {}},
        )
        perturbation_results.append(
            {
                "perturbation_id": perturbation_id,
                "timeout_seconds": 5.0,
                "max_extra_steps": 5,
                "phase2cs_report_json": str(phase2cs_report_json),
                "validation_report_json": str(validation_json),
                "phase2cs_passed": True,
                "validation_passed": True,
                "core_signature": {},
            }
        )
    report = {
        "artifact_family": "phase2cu_fresh_execution_runtime_perturbation_matrix",
        "passed": True,
        "ready_for_bounded_fresh_execution_runtime_perturbation_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": {
            "source_phase2ct_passed": True,
            "minimum_three_perturbations_met": True,
            "all_perturbation_phase2cs_reports_passed": True,
            "all_perturbation_phase2cs_validations_passed": True,
            "all_perturbation_core_signatures_match": True,
            "all_perturbations_keep_bounded_claims_only": True,
        },
        "metrics": {
            "perturbation_count": 3,
            "passed_phase2cs_reports": 3,
            "passed_phase2cs_validations": 3,
            "core_signature_mismatch_count": 0,
            "repetition_count_per_perturbation": 2,
            "fresh_runtime_execution_count": 18,
        },
        "perturbation_results": perturbation_results,
        "evidence": {
            "runtime_perturbation_output_dir": str(source_output_dir),
            "phase2ct_report_json": str(tmp_path / "phase2ct.json"),
        },
    }
    return _write(tmp_path / "phase2cu.json", report)


def test_phase2cv_rejects_runtime_perturbation_negative_controls(
    tmp_path: Path,
) -> None:
    report = audit_phase2cv_runtime_perturbation_negative_controls(
        phase2cu_report_json=_phase2cu_fixture(tmp_path),
        output_dir=tmp_path / "controls",
        output_report_json=tmp_path / "phase2cv.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 8
    assert all(
        row["expected_failed_checks_observed"]
        for row in report["control_results"]
    )
