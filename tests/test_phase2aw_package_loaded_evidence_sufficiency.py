import json
from pathlib import Path

from reflexlm.cli.build_phase2aw_package_loaded_evidence_sufficiency_report import (
    build_phase2aw_package_loaded_evidence_sufficiency_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2aw_package_loaded_evidence_sufficiency_accepts_bounded_package_delta(
    tmp_path: Path,
) -> None:
    report = build_phase2aw_package_loaded_evidence_sufficiency_report(
        package_authorization_gate_json=_write(
            tmp_path / "auth.json",
            {"passed": True, "ready_for_sealed_eval": False},
        ),
        postpackage_gate_json=_write(
            tmp_path / "post.json",
            {"passed": True, "ready_for_sealed_eval": False},
        ),
        package_loaded_runtime_gate_json=_write(
            tmp_path / "runtime.json",
            {
                "passed": True,
                "ready_for_sealed_eval": False,
                "metrics": {
                    "full_success_rate": 0.92,
                    "source_overlap_success_rate": 0.41,
                    "full_minus_source_overlap_success_rate": 0.51,
                    "full_selection_accuracy": 1.0,
                },
            },
        ),
        package_loaded_failure_audit_json=_write(
            tmp_path / "failure.json",
            {
                "passed": True,
                "checks": {
                    "selection_is_not_primary_bottleneck": True,
                    "holdout_source_artifact_split_clean": True,
                },
                "metrics": {"full_rows": 156, "full_failures": 12},
            },
        ),
    )

    assert report["passed"] is True
    assert (
        "phase2aw_package_loaded_bounded_nonsealed_runtime_delta_supported"
        in report["supported_claims"]
    )
    assert "sealed_cross_model_transfer" in report["unsupported_claims"]
    assert "epoch_making_architecture" in report["unsupported_claims"]


def test_phase2aw_package_loaded_evidence_sufficiency_rejects_sealed_ready_input(
    tmp_path: Path,
) -> None:
    report = build_phase2aw_package_loaded_evidence_sufficiency_report(
        package_authorization_gate_json=_write(
            tmp_path / "auth.json",
            {"passed": True, "ready_for_sealed_eval": True},
        ),
        postpackage_gate_json=_write(
            tmp_path / "post.json",
            {"passed": True, "ready_for_sealed_eval": False},
        ),
        package_loaded_runtime_gate_json=_write(
            tmp_path / "runtime.json",
            {"passed": True, "ready_for_sealed_eval": False, "metrics": {}},
        ),
        package_loaded_failure_audit_json=_write(
            tmp_path / "failure.json",
            {
                "passed": True,
                "checks": {
                    "selection_is_not_primary_bottleneck": True,
                    "holdout_source_artifact_split_clean": True,
                },
            },
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["package_authorization_does_not_allow_sealed"] is False
    assert report["supported_claims"] == []
