import json
from pathlib import Path

import reflexlm.cli.audit_phase2cu_fresh_execution_runtime_perturbation_matrix as phase2cu


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _matrix_audit(runtime_specs: list[tuple[str, str]], *, drift: bool = False) -> dict:
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
            "runtime_paths": [runtime for runtime, _ in runtime_specs],
            "python_versions": [version for _, version in runtime_specs],
            "distinct_runtime_paths": 3,
            "distinct_python_versions": 1 if drift else 2,
            "episodes_per_runtime": 9,
        },
        "supported_claims": ["bounded matrix"],
        "unsupported_claims": ["epoch-making architecture"],
        "next_required_experiment": "phase2cq_cross_runtime_stress_negative_controls",
    }


def _phase2cs_report(
    output_dir: Path,
    *,
    drift: bool = False,
) -> dict:
    runtime_specs = [
        ("C:\\Python313\\python.exe", "3.13.2"),
        ("D:\\repo\\.venv312\\Scripts\\python.exe", "3.12.10"),
        ("D:\\alias\\Scripts\\python.exe", "3.12.10"),
    ]
    repetitions = []
    for repetition_index in range(2):
        matrix_json = _write(
            output_dir
            / f"repetition_{repetition_index:02d}"
            / "phase2cp_fresh_execution_audit.json",
            _matrix_audit(runtime_specs, drift=drift),
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
            "runtime_interpreters": [runtime for runtime, _ in runtime_specs],
        },
        "repetition_results": repetitions,
    }


def _phase2ct_fixture(tmp_path: Path) -> Path:
    phase2cs = _write(
        tmp_path / "phase2cs.json",
        {
            "passed": True,
            "evidence": {
                "phase2cp_report_json": str(tmp_path / "phase2cp.json"),
                "suite_json": str(tmp_path / "suite.json"),
            },
        },
    )
    return _write(
        tmp_path / "phase2ct.json",
        {
            "passed": True,
            "evidence": {"phase2cs_report_json": str(phase2cs)},
        },
    )


def test_phase2cu_accepts_stable_runtime_perturbation_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_phase2cs(**kwargs):
        report = _phase2cs_report(Path(kwargs["output_dir"]))
        _write(Path(kwargs["output_report_json"]), report)
        return report

    monkeypatch.setattr(
        phase2cu,
        "audit_phase2cs_fresh_runtime_execution_repetition_stability",
        _fake_phase2cs,
    )
    monkeypatch.setattr(
        phase2cu,
        "validate_phase2cs_fresh_runtime_execution_report",
        lambda report: {"passed": True, "checks": {"ok": True}, "metrics": {}},
    )

    report = phase2cu.audit_phase2cu_fresh_execution_runtime_perturbation_matrix(
        phase2ct_report_json=_phase2ct_fixture(tmp_path),
        output_dir=tmp_path / "cu",
        output_report_json=tmp_path / "phase2cu.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["perturbation_count"] == 3
    assert report["metrics"]["core_signature_mismatch_count"] == 0
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_phase2cu_rejects_perturbation_signature_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = {"count": 0}

    def _fake_phase2cs(**kwargs):
        report = _phase2cs_report(
            Path(kwargs["output_dir"]),
            drift=calls["count"] == 1,
        )
        calls["count"] += 1
        _write(Path(kwargs["output_report_json"]), report)
        return report

    monkeypatch.setattr(
        phase2cu,
        "audit_phase2cs_fresh_runtime_execution_repetition_stability",
        _fake_phase2cs,
    )
    monkeypatch.setattr(
        phase2cu,
        "validate_phase2cs_fresh_runtime_execution_report",
        lambda report: {"passed": True, "checks": {"ok": True}, "metrics": {}},
    )

    report = phase2cu.audit_phase2cu_fresh_execution_runtime_perturbation_matrix(
        phase2ct_report_json=_phase2ct_fixture(tmp_path),
        output_dir=tmp_path / "cu",
        output_report_json=tmp_path / "phase2cu.json",
    )

    assert report["passed"] is False
    assert report["checks"]["all_perturbation_core_signatures_match"] is False
    assert report["metrics"]["core_signature_mismatch_count"] == 1
