import json
from pathlib import Path

from reflexlm.cli.build_phase2aa_phase2at_runtime_boundary_report import (
    build_phase2aa_phase2at_runtime_boundary_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _phase2aa_data(*, passed: bool = True, artifacts_resolved: bool = True) -> dict:
    return {
        "passed": passed,
        "checks": {"required_runtime_artifacts_available": artifacts_resolved},
    }


def _phase2aa_delta(*, passed: bool = True, rows: int = 24) -> dict:
    return {
        "passed": passed,
        "metrics": {
            "full_rows": rows,
            "control_rows": rows,
            "full_success_rate": 1.0,
            "control_success_rate": 0.5,
            "full_minus_control_success_rate": 0.5,
            "full_selection_accuracy": 1.0,
            "control_selection_accuracy": 0.5,
            "full_minus_control_selection_accuracy": 0.5,
        },
    }


def _phase2at_delta(*, passed: bool = False) -> dict:
    return {
        "passed": passed,
        "metrics": {
            "full_success_rate": 1.0,
            "control_success_rate": 1.0,
            "full_minus_control_success_rate": 0.0,
        },
    }


def test_runtime_boundary_report_separates_candidate_delta_from_symbolic_tie(
    tmp_path: Path,
) -> None:
    report = build_phase2aa_phase2at_runtime_boundary_report(
        phase2aa_data_health_json=_write(tmp_path / "aa_data.json", _phase2aa_data()),
        phase2aa_candidate_delta_gate_json=_write(tmp_path / "aa_delta.json", _phase2aa_delta()),
        phase2at_symbolic_runtime_delta_gate_json=_write(
            tmp_path / "at_delta.json", _phase2at_delta()
        ),
    )

    assert report["passed"] is True
    assert "phase2at_symbolic_structural_runtime_policy_delta" in report["unsupported_claims"]
    assert (
        "bounded_patch_candidate_selection_package_delta_on_nonsealed_public_repo_rows"
        in report["supported_claims"]
    )


def test_runtime_boundary_report_rejects_symbolic_delta_overclaim(
    tmp_path: Path,
) -> None:
    report = build_phase2aa_phase2at_runtime_boundary_report(
        phase2aa_data_health_json=_write(tmp_path / "aa_data.json", _phase2aa_data()),
        phase2aa_candidate_delta_gate_json=_write(tmp_path / "aa_delta.json", _phase2aa_delta()),
        phase2at_symbolic_runtime_delta_gate_json=_write(
            tmp_path / "at_delta.json", _phase2at_delta(passed=True)
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["phase2at_symbolic_delta_not_misclaimed"] is False


def test_runtime_boundary_report_rejects_unresolved_runtime_artifacts(
    tmp_path: Path,
) -> None:
    report = build_phase2aa_phase2at_runtime_boundary_report(
        phase2aa_data_health_json=_write(
            tmp_path / "aa_data.json", _phase2aa_data(artifacts_resolved=False)
        ),
        phase2aa_candidate_delta_gate_json=_write(tmp_path / "aa_delta.json", _phase2aa_delta()),
        phase2at_symbolic_runtime_delta_gate_json=_write(
            tmp_path / "at_delta.json", _phase2at_delta()
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["phase2aa_artifacts_resolved"] is False


def test_runtime_boundary_report_does_not_request_24_row_scaling_after_256_rows(
    tmp_path: Path,
) -> None:
    report = build_phase2aa_phase2at_runtime_boundary_report(
        phase2aa_data_health_json=_write(tmp_path / "aa_data.json", _phase2aa_data()),
        phase2aa_candidate_delta_gate_json=_write(
            tmp_path / "aa_delta.json", _phase2aa_delta(rows=256)
        ),
        phase2at_symbolic_runtime_delta_gate_json=_write(
            tmp_path / "at_delta.json", _phase2at_delta()
        ),
        min_phase2aa_rows=256,
    )

    assert report["passed"] is True
    assert "scale_candidate_selection_delta_to_full_256_row_holdout" not in report[
        "next_required_evidence"
    ]
    assert "replicate_candidate_selection_delta_across_models_and_seeds" in report[
        "next_required_evidence"
    ]
