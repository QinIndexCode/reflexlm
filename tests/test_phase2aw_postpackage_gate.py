import json
from pathlib import Path

from reflexlm.cli.audit_phase2aw_postpackage_gate import audit_phase2aw_postpackage_gate
from reflexlm.cli.audit_phase2at_learned_patch_generation_package_gate import (
    REQUIRED_OPEN_REPAIR_CAPABILITIES,
)
from reflexlm.llm.native_nervous_package import write_native_nervous_package


SCHEMA = "phase2at.learned_bounded_patch_candidate.v1"


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _authorization() -> dict:
    return {
        "passed": True,
        "ready_for_package_build": True,
        "ready_for_sealed_eval": False,
    }


def _schema_gate() -> dict:
    return {
        "artifact_family": "phase2at_learned_patch_generation_package_gate",
        "passed": True,
    }


def _sufficiency() -> dict:
    return {
        "passed": True,
        "claim_scope": "phase2av_bounded_nonsealed_descriptor_runtime_candidate_selection",
        "unsupported_claims": ["phase2av_package_ready"],
    }


def _package(tmp_path: Path, *, policy_label: str = "phase2aw_rows156_v4") -> Path:
    head_path = tmp_path / "heads"
    low_path = tmp_path / "low" / "model.pt"
    head_path.mkdir(parents=True)
    low_path.parent.mkdir(parents=True)
    low_path.write_text("checkpoint", encoding="utf-8")
    package_dir = tmp_path / "pkg"
    write_native_nervous_package(
        package_dir,
        base_model_name="qwen",
        native_head_path=head_path,
        low_level_checkpoint_path=low_path,
        policy_label=policy_label,
        open_repair_capabilities={name: True for name in REQUIRED_OPEN_REPAIR_CAPABILITIES},
        patch_proposal_strategy="learned_bounded_candidate",
        learned_patch_generation_enabled=True,
        patch_candidate_schema_version=SCHEMA,
    )
    return package_dir


def test_phase2aw_postpackage_gate_authorizes_only_package_loaded_nonsealed_eval(
    tmp_path: Path,
) -> None:
    report = audit_phase2aw_postpackage_gate(
        package_path=_package(tmp_path),
        package_authorization_gate_json=_write(tmp_path / "auth.json", _authorization()),
        package_schema_gate_json=_write(tmp_path / "schema.json", _schema_gate()),
        evidence_sufficiency_json=_write(tmp_path / "sufficiency.json", _sufficiency()),
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_package_loaded_runtime_eval"] is True
    assert report["ready_for_sealed_eval"] is False
    assert (
        "phase2aw_package_built_for_bounded_package_loaded_nonsealed_runtime_eval"
        in report["supported_claims"]
    )
    assert "do_not_run_sealed_eval_until_package_loaded_nonsealed_runtime_gate_passes" in report[
        "postpackage_blocked_actions"
    ]


def test_phase2aw_postpackage_gate_rejects_authorization_that_claims_sealed_ready(
    tmp_path: Path,
) -> None:
    authorization = _authorization()
    authorization["ready_for_sealed_eval"] = True

    report = audit_phase2aw_postpackage_gate(
        package_path=_package(tmp_path),
        package_authorization_gate_json=_write(tmp_path / "auth.json", authorization),
        package_schema_gate_json=_write(tmp_path / "schema.json", _schema_gate()),
        evidence_sufficiency_json=_write(tmp_path / "sufficiency.json", _sufficiency()),
    )

    assert report["passed"] is False
    assert report["checks"]["authorization_does_not_claim_sealed_ready"] is False
    assert "do_not_run_sealed_eval" in report["blocked_actions"]


def test_phase2aw_postpackage_gate_rejects_non_phase2aw_package_label(
    tmp_path: Path,
) -> None:
    report = audit_phase2aw_postpackage_gate(
        package_path=_package(tmp_path, policy_label="phase2at_only"),
        package_authorization_gate_json=_write(tmp_path / "auth.json", _authorization()),
        package_schema_gate_json=_write(tmp_path / "schema.json", _schema_gate()),
        evidence_sufficiency_json=_write(tmp_path / "sufficiency.json", _sufficiency()),
    )

    assert report["passed"] is False
    assert report["checks"]["policy_label_is_phase2aw"] is False


def test_phase2aw_postpackage_gate_rejects_evidence_that_already_claims_package_ready(
    tmp_path: Path,
) -> None:
    sufficiency = _sufficiency()
    sufficiency["unsupported_claims"] = []

    report = audit_phase2aw_postpackage_gate(
        package_path=_package(tmp_path),
        package_authorization_gate_json=_write(tmp_path / "auth.json", _authorization()),
        package_schema_gate_json=_write(tmp_path / "schema.json", _schema_gate()),
        evidence_sufficiency_json=_write(tmp_path / "sufficiency.json", sufficiency),
    )

    assert report["passed"] is False
    assert report["checks"]["evidence_report_did_not_preclaim_package_ready"] is False
