import json
from pathlib import Path

from reflexlm.cli.audit_phase2j_pretrain_gate import build_phase2j_pretrain_gate


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _preregistration() -> dict:
    return {"passed": True}


def _readiness() -> dict:
    return {
        "passed": True,
        "ready_for_data_generation": True,
        "ready_for_training": False,
    }


def _data_health() -> dict:
    return {
        "passed": True,
        "sealed_usage": {
            "sealed_splits_used_for_training": False,
            "sealed_splits_used_for_tuning": False,
        },
        "checks": {
            "phase2j_effective_split_hashes_present": True,
            "phase2j_train_val_target_overlap": True,
            "phase2j_train_val_command_intent_coverage": True,
        },
    }


def _head_manifest() -> dict:
    return {
        "leakage_audit": {"passed": True},
        "coverage_audit": {"passed": True},
    }


def test_phase2j_pretrain_gate_allows_only_smoke_after_all_pretrain_evidence(
    tmp_path: Path,
) -> None:
    preregistration = _write(tmp_path / "preregistration.json", _preregistration())
    readiness = _write(tmp_path / "readiness.json", _readiness())
    data_health = _write(tmp_path / "data_health.json", _data_health())
    head_manifest = _write(tmp_path / "head_manifest.json", _head_manifest())

    report = build_phase2j_pretrain_gate(
        preregistration_json=preregistration,
        readiness_json=readiness,
        data_health_json=data_health,
        head_manifest_json=head_manifest,
    )

    assert report["passed"] is True
    assert report["ready_for_smoke_training"] is True
    assert report["ready_for_full_training"] is False
    assert report["ready_for_package"] is False
    assert report["ready_for_sealed_eval"] is False
    assert report["allowed_next_action"] == "run_nonsealed_phase2j_smoke_training_only"


def test_phase2j_pretrain_gate_rejects_sealed_training_usage(tmp_path: Path) -> None:
    data_health_payload = _data_health()
    data_health_payload["sealed_usage"]["sealed_splits_used_for_training"] = True
    preregistration = _write(tmp_path / "preregistration.json", _preregistration())
    readiness = _write(tmp_path / "readiness.json", _readiness())
    data_health = _write(tmp_path / "data_health.json", data_health_payload)

    report = build_phase2j_pretrain_gate(
        preregistration_json=preregistration,
        readiness_json=readiness,
        data_health_json=data_health,
    )

    assert report["passed"] is False
    assert "do_not_train_with_sealed_inputs_or_tuning_feedback" in report["blocked_actions"]


def test_phase2j_pretrain_gate_rejects_missing_split_hashes(tmp_path: Path) -> None:
    data_health_payload = _data_health()
    data_health_payload["checks"]["phase2j_effective_split_hashes_present"] = False
    preregistration = _write(tmp_path / "preregistration.json", _preregistration())
    readiness = _write(tmp_path / "readiness.json", _readiness())
    data_health = _write(tmp_path / "data_health.json", data_health_payload)

    report = build_phase2j_pretrain_gate(
        preregistration_json=preregistration,
        readiness_json=readiness,
        data_health_json=data_health,
    )

    assert report["passed"] is False
    assert "do_not_train_without_effective_split_hashes" in report["blocked_actions"]


def test_phase2j_pretrain_gate_rejects_source_overlap_hard_baseline_failure(
    tmp_path: Path,
) -> None:
    data_health_payload = _data_health()
    data_health_payload["checks"][
        "phase2j_source_overlap_hard_val_baseline_below_threshold"
    ] = False
    preregistration = _write(tmp_path / "preregistration.json", _preregistration())
    readiness = _write(tmp_path / "readiness.json", _readiness())
    data_health = _write(tmp_path / "data_health.json", data_health_payload)

    report = build_phase2j_pretrain_gate(
        preregistration_json=preregistration,
        readiness_json=readiness,
        data_health_json=data_health,
    )

    assert report["passed"] is False
    assert (
        "do_not_train_when_source_overlap_baseline_solves_phase2j_hard_val"
        in report["blocked_actions"]
    )


def test_phase2j_pretrain_gate_rejects_missing_synapse_reference(
    tmp_path: Path,
) -> None:
    data_health_payload = _data_health()
    data_health_payload["checks"]["phase2j_head_train_synapse_reference_present"] = True
    data_health_payload["checks"]["phase2j_head_val_synapse_reference_present"] = False
    preregistration = _write(tmp_path / "preregistration.json", _preregistration())
    readiness = _write(tmp_path / "readiness.json", _readiness())
    data_health = _write(tmp_path / "data_health.json", data_health_payload)

    report = build_phase2j_pretrain_gate(
        preregistration_json=preregistration,
        readiness_json=readiness,
        data_health_json=data_health,
    )

    assert report["passed"] is False
    assert report["checks"]["phase2j_synapse_reference_present"] is False
    assert "do_not_train_without_phase2j_synapse_reference" in report["blocked_actions"]


def test_phase2j_pretrain_gate_rejects_missing_debug_action_stage(
    tmp_path: Path,
) -> None:
    data_health_payload = _data_health()
    data_health_payload["checks"]["phase2j_head_train_debug_action_stage_present"] = True
    data_health_payload["checks"]["phase2j_head_train_debug_action_stage_coverage"] = True
    data_health_payload["checks"]["phase2j_head_val_debug_action_stage_present"] = True
    data_health_payload["checks"]["phase2j_head_val_debug_action_stage_coverage"] = False
    preregistration = _write(tmp_path / "preregistration.json", _preregistration())
    readiness = _write(tmp_path / "readiness.json", _readiness())
    data_health = _write(tmp_path / "data_health.json", data_health_payload)

    report = build_phase2j_pretrain_gate(
        preregistration_json=preregistration,
        readiness_json=readiness,
        data_health_json=data_health,
    )

    assert report["passed"] is False
    assert report["checks"]["phase2j_debug_action_stage_present"] is False
    assert "do_not_train_without_phase2j_debug_action_stage" in report["blocked_actions"]
