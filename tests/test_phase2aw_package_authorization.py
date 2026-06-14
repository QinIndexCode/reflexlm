import json
from pathlib import Path

from reflexlm.cli.audit_phase2aw_package_authorization import (
    audit_phase2aw_package_authorization,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _split_manifest() -> dict:
    return {
        "passed": True,
        "artifact_paths_rewritten": True,
        "source_artifact_split_clean_by_construction": True,
    }


def _health() -> dict:
    return {"passed": True}


def _runtime_gate() -> dict:
    return {
        "passed": True,
        "metrics": {
            "full_success_rate": 0.92,
            "control_success_rate": 0.41,
            "full_minus_control_success_rate": 0.51,
        },
    }


def _verified_gate() -> dict:
    return {
        "passed": True,
        "ready_for_phase2aw_package_or_successor_training": True,
    }


def _sufficiency() -> dict:
    return {
        "passed": True,
        "claim_scope": "phase2av_bounded_nonsealed_descriptor_runtime_candidate_selection",
        "metrics": {
            "multiseed_unique_seed_count": 2,
            "cross_model_model_count": 2,
        },
        "unsupported_claims": ["phase2av_package_ready"],
    }


def test_phase2aw_package_authorization_accepts_bounded_nonsealed_evidence(
    tmp_path: Path,
) -> None:
    report = audit_phase2aw_package_authorization(
        split_clean_manifest_json=_write(tmp_path / "split.json", _split_manifest()),
        train_data_health_json=_write(tmp_path / "train.json", _health()),
        val_data_health_json=_write(tmp_path / "val.json", _health()),
        holdout_data_health_json=_write(tmp_path / "holdout.json", _health()),
        runtime_execution_gate_json=_write(tmp_path / "runtime.json", _runtime_gate()),
        verified_candidate_pool_gate_json=_write(tmp_path / "verified.json", _verified_gate()),
        evidence_sufficiency_json=_write(tmp_path / "sufficiency.json", _sufficiency()),
    )

    assert report["passed"] is True
    assert report["ready_for_package_build"] is True
    assert report["ready_for_sealed_eval"] is False
    assert (
        "phase2aw_package_build_authorized_for_bounded_nonsealed_descriptor_runtime"
        in report["supported_claims"]
    )


def test_phase2aw_package_authorization_rejects_premature_package_claim(
    tmp_path: Path,
) -> None:
    sufficiency = _sufficiency()
    sufficiency["unsupported_claims"] = []

    report = audit_phase2aw_package_authorization(
        split_clean_manifest_json=_write(tmp_path / "split.json", _split_manifest()),
        train_data_health_json=_write(tmp_path / "train.json", _health()),
        val_data_health_json=_write(tmp_path / "val.json", _health()),
        holdout_data_health_json=_write(tmp_path / "holdout.json", _health()),
        runtime_execution_gate_json=_write(tmp_path / "runtime.json", _runtime_gate()),
        verified_candidate_pool_gate_json=_write(tmp_path / "verified.json", _verified_gate()),
        evidence_sufficiency_json=_write(tmp_path / "sufficiency.json", sufficiency),
    )

    assert report["passed"] is False
    assert report["checks"]["package_not_already_claimed_by_evidence_report"] is False
    assert "do_not_build_phase2aw_package" in report["blocked_actions"]
