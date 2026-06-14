import hashlib
import hmac
import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_persistent_state_chain import (
    audit_phase2homeostasis_persistent_state_chain,
)
from reflexlm.runtime.homeostasis import (
    HomeostaticControlConfig,
    HomeostaticSynapticController,
)
from reflexlm.schema import ActionType


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _unsigned_artifact(artifact: dict) -> dict:
    return {
        key: value
        for key, value in artifact.items()
        if key not in {"integrity_sha256", "authenticator"}
    }


def _hmac_digest(payload: dict, key: bytes) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _rehash(artifact: dict, *, authenticity_key: str | None = None) -> None:
    unsigned = _unsigned_artifact(artifact)
    artifact["integrity_sha256"] = _stable_hash(unsigned)
    if authenticity_key is None:
        artifact["authenticator"] = {
            "algorithm": "sha256",
            "digest": _stable_hash(unsigned),
            "key_fingerprint_sha256": None,
        }
    else:
        key = authenticity_key.encode("utf-8")
        artifact["authenticator"] = {
            "algorithm": "hmac-sha256",
            "digest": _hmac_digest(unsigned, key),
            "key_fingerprint_sha256": hashlib.sha256(key).hexdigest(),
        }


def _advance(controller: HomeostaticSynapticController) -> None:
    controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.20,
        risk=0.10,
        prediction_error=0.20,
        temporal_observation_available=True,
        failure_visible=True,
    )
    controller.observe(
        proposed_action=ActionType.WAIT,
        salience=0.10,
        risk=0.05,
        prediction_error=0.05,
        temporal_observation_available=True,
        failure_visible=False,
    )
    controller.reset()


def _report(
    tmp_path: Path,
    *,
    name: str,
    artifact: dict,
    io: dict,
    loaded_artifact: dict | None = None,
) -> dict:
    subreport = _write(
        tmp_path / f"{name}-subreport.json",
        {
            "episode_reports": [
                {
                    "episode_id": "episode-1",
                    "selected_actions": [
                        {
                            "type": "RUN_COMMAND",
                            "command": "python -m pytest -q",
                            "file_target": None,
                            "reason": "verify",
                        }
                    ],
                }
            ]
        },
    )
    state = {
        "config": artifact["config"],
        **artifact["state"],
    }
    state_io = dict(io)
    saved_authenticator = artifact.get("authenticator", {})
    loaded_authenticator = (
        loaded_artifact.get("authenticator", {})
        if isinstance(loaded_artifact, dict)
        else {}
    )
    state_io.setdefault(
        "saved_authenticator_algorithm",
        saved_authenticator.get("algorithm") if state_io.get("saved") else None,
    )
    state_io.setdefault(
        "saved_key_fingerprint_sha256",
        saved_authenticator.get("key_fingerprint_sha256")
        if state_io.get("saved")
        else None,
    )
    state_io.setdefault(
        "loaded_authenticator_algorithm",
        loaded_authenticator.get("algorithm") if state_io.get("loaded") else None,
    )
    state_io.setdefault(
        "loaded_key_fingerprint_sha256",
        loaded_authenticator.get("key_fingerprint_sha256")
        if state_io.get("loaded")
        else None,
    )
    return {
        "passed": True,
        "seed": 7,
        "metrics": {
            "repositories": 1,
            "episodes": 1,
            "executed_actions": 1,
            "task_completion_successes": 1,
            "task_completion_success_rate": 1.0,
        },
        "homeostatic_state_io": state_io,
        "repository_reports": [
            {
                "repository_id": "repo",
                "report_json": str(subreport),
                "checks": {
                    "all_model_selected_actions_were_allowlisted": True,
                    "all_task_completion_predicates_satisfied": True,
                },
                "policy_configuration": {
                    "policy_metadata": {
                        "expert_policy": {"homeostatic_control": state}
                    }
                },
            }
        ],
    }


def _replace_selected_actions(report: dict, actions: list[dict]) -> None:
    subreport_path = Path(report["repository_reports"][0]["report_json"])
    subreport = json.loads(subreport_path.read_text(encoding="utf-8"))
    subreport["episode_reports"][0]["selected_actions"] = actions
    subreport_path.write_text(json.dumps(subreport), encoding="utf-8")


def test_persistent_state_chain_audit_accepts_verified_cross_process_chain(
    tmp_path: Path,
) -> None:
    config = HomeostaticControlConfig(
        surprise_wake_threshold=0.80,
        preserve_adaptive_threshold_across_reset=True,
    )
    generation1 = HomeostaticSynapticController(config)
    _advance(generation1)
    artifact1 = generation1.export_persistent_state()
    generation2 = HomeostaticSynapticController(config)
    generation2.load_persistent_state(artifact1)
    _advance(generation2)
    artifact2 = generation2.export_persistent_state()
    report1 = _report(
        tmp_path,
        name="generation1",
        artifact=artifact1,
        io={
            "loaded": False,
            "saved": True,
            "loaded_integrity_sha256": None,
            "saved_integrity_sha256": artifact1["integrity_sha256"],
        },
    )
    report2 = _report(
        tmp_path,
        name="generation2",
        artifact=artifact2,
        loaded_artifact=artifact1,
        io={
            "loaded": True,
            "saved": True,
            "loaded_integrity_sha256": artifact1["integrity_sha256"],
            "saved_integrity_sha256": artifact2["integrity_sha256"],
        },
    )

    audit = audit_phase2homeostasis_persistent_state_chain(
        generation1_report_json=_write(tmp_path / "report1.json", report1),
        generation2_report_json=_write(tmp_path / "report2.json", report2),
        generation1_state_json=_write(tmp_path / "state1.json", artifact1),
        generation2_state_json=_write(tmp_path / "state2.json", artifact2),
        output_report_json=tmp_path / "audit.json",
    )

    assert audit["passed"] is True
    assert audit["ready_for_bounded_cross_process_homeostatic_memory_claim"] is True


def test_persistent_state_chain_audit_accepts_hmac_authenticated_chain(
    tmp_path: Path,
) -> None:
    config = HomeostaticControlConfig(
        surprise_wake_threshold=0.80,
        preserve_adaptive_threshold_across_reset=True,
    )
    authenticity_key = "bounded-chain-key"
    generation1 = HomeostaticSynapticController(config)
    _advance(generation1)
    artifact1 = generation1.export_persistent_state(
        authenticity_key=authenticity_key
    )
    generation2 = HomeostaticSynapticController(config)
    generation2.load_persistent_state(
        artifact1,
        authenticity_key=authenticity_key,
    )
    _advance(generation2)
    artifact2 = generation2.export_persistent_state(
        authenticity_key=authenticity_key
    )
    report1 = _report(
        tmp_path,
        name="generation1-hmac",
        artifact=artifact1,
        io={
            "loaded": False,
            "saved": True,
            "loaded_integrity_sha256": None,
            "saved_integrity_sha256": artifact1["integrity_sha256"],
        },
    )
    report2 = _report(
        tmp_path,
        name="generation2-hmac",
        artifact=artifact2,
        loaded_artifact=artifact1,
        io={
            "loaded": True,
            "saved": True,
            "loaded_integrity_sha256": artifact1["integrity_sha256"],
            "saved_integrity_sha256": artifact2["integrity_sha256"],
        },
    )

    audit = audit_phase2homeostasis_persistent_state_chain(
        generation1_report_json=_write(tmp_path / "report1-hmac.json", report1),
        generation2_report_json=_write(tmp_path / "report2-hmac.json", report2),
        generation1_state_json=_write(tmp_path / "state1-hmac.json", artifact1),
        generation2_state_json=_write(tmp_path / "state2-hmac.json", artifact2),
        output_report_json=tmp_path / "audit-hmac.json",
        authenticity_key=authenticity_key,
    )

    assert audit["passed"] is True
    assert audit["metrics"]["generation1_authenticator_algorithm"] == "hmac-sha256"
    assert audit["metrics"]["generation2_authenticator_algorithm"] == "hmac-sha256"


def test_persistent_state_chain_audit_rejects_hmac_chain_without_key(
    tmp_path: Path,
) -> None:
    config = HomeostaticControlConfig(
        surprise_wake_threshold=0.80,
        preserve_adaptive_threshold_across_reset=True,
    )
    authenticity_key = "bounded-chain-key"
    generation1 = HomeostaticSynapticController(config)
    _advance(generation1)
    artifact1 = generation1.export_persistent_state(
        authenticity_key=authenticity_key
    )
    generation2 = HomeostaticSynapticController(config)
    generation2.load_persistent_state(
        artifact1,
        authenticity_key=authenticity_key,
    )
    _advance(generation2)
    artifact2 = generation2.export_persistent_state(
        authenticity_key=authenticity_key
    )
    report1 = _report(
        tmp_path,
        name="generation1-hmac-missing-key",
        artifact=artifact1,
        io={
            "loaded": False,
            "saved": True,
            "loaded_integrity_sha256": None,
            "saved_integrity_sha256": artifact1["integrity_sha256"],
        },
    )
    report2 = _report(
        tmp_path,
        name="generation2-hmac-missing-key",
        artifact=artifact2,
        loaded_artifact=artifact1,
        io={
            "loaded": True,
            "saved": True,
            "loaded_integrity_sha256": artifact1["integrity_sha256"],
            "saved_integrity_sha256": artifact2["integrity_sha256"],
        },
    )

    audit = audit_phase2homeostasis_persistent_state_chain(
        generation1_report_json=_write(
            tmp_path / "report1-hmac-missing-key.json",
            report1,
        ),
        generation2_report_json=_write(
            tmp_path / "report2-hmac-missing-key.json",
            report2,
        ),
        generation1_state_json=_write(
            tmp_path / "state1-hmac-missing-key.json",
            artifact1,
        ),
        generation2_state_json=_write(
            tmp_path / "state2-hmac-missing-key.json",
            artifact2,
        ),
        output_report_json=tmp_path / "audit-hmac-missing-key.json",
    )

    assert audit["passed"] is False
    assert audit["checks"]["both_artifact_integrities_valid"] is False


def test_persistent_state_chain_accepts_observation_only_trace_modulation(
    tmp_path: Path,
) -> None:
    config = HomeostaticControlConfig(preserve_adaptive_threshold_across_reset=True)
    generation1 = HomeostaticSynapticController(config)
    _advance(generation1)
    artifact1 = generation1.export_persistent_state()
    generation2 = HomeostaticSynapticController(config)
    generation2.load_persistent_state(artifact1)
    _advance(generation2)
    artifact2 = generation2.export_persistent_state()
    report1 = _report(
        tmp_path,
        name="generation1-observation",
        artifact=artifact1,
        io={"saved": True, "saved_integrity_sha256": artifact1["integrity_sha256"]},
    )
    report2 = _report(
        tmp_path,
        name="generation2-observation",
        artifact=artifact2,
        loaded_artifact=artifact1,
        io={
            "loaded": True,
            "saved": True,
            "loaded_integrity_sha256": artifact1["integrity_sha256"],
            "saved_integrity_sha256": artifact2["integrity_sha256"],
        },
    )
    base_action = {
        "type": "RUN_COMMAND",
        "command": "python -m pytest -q",
        "file_target": None,
        "reason": "verify",
    }
    _replace_selected_actions(
        report1,
        [base_action, {"type": "WAIT", "command": None, "file_target": None}],
    )
    _replace_selected_actions(
        report2,
        [base_action, {"type": "REFRESH_STATE", "command": None, "file_target": None}],
    )

    audit = audit_phase2homeostasis_persistent_state_chain(
        generation1_report_json=_write(tmp_path / "report1-observation.json", report1),
        generation2_report_json=_write(tmp_path / "report2-observation.json", report2),
        generation1_state_json=_write(tmp_path / "state1-observation.json", artifact1),
        generation2_state_json=_write(tmp_path / "state2-observation.json", artifact2),
        output_report_json=tmp_path / "audit-observation.json",
    )

    assert audit["passed"] is True
    assert audit["metrics"]["selected_action_traces_match"] is False
    assert audit["checks"]["side_effect_action_traces_match"] is True


def test_persistent_state_chain_rejects_side_effect_trace_modulation(
    tmp_path: Path,
) -> None:
    config = HomeostaticControlConfig(preserve_adaptive_threshold_across_reset=True)
    generation1 = HomeostaticSynapticController(config)
    _advance(generation1)
    artifact1 = generation1.export_persistent_state()
    generation2 = HomeostaticSynapticController(config)
    generation2.load_persistent_state(artifact1)
    _advance(generation2)
    artifact2 = generation2.export_persistent_state()
    report1 = _report(
        tmp_path,
        name="generation1-side-effect",
        artifact=artifact1,
        io={"saved": True, "saved_integrity_sha256": artifact1["integrity_sha256"]},
    )
    report2 = _report(
        tmp_path,
        name="generation2-side-effect",
        artifact=artifact2,
        loaded_artifact=artifact1,
        io={
            "loaded": True,
            "saved": True,
            "loaded_integrity_sha256": artifact1["integrity_sha256"],
            "saved_integrity_sha256": artifact2["integrity_sha256"],
        },
    )
    _replace_selected_actions(
        report2,
        [
            {
                "type": "STOP_PROCESS",
                "command": None,
                "file_target": None,
                "reason": "changed-side-effect",
            }
        ],
    )

    audit = audit_phase2homeostasis_persistent_state_chain(
        generation1_report_json=_write(tmp_path / "report1-side-effect.json", report1),
        generation2_report_json=_write(tmp_path / "report2-side-effect.json", report2),
        generation1_state_json=_write(tmp_path / "state1-side-effect.json", artifact1),
        generation2_state_json=_write(tmp_path / "state2-side-effect.json", artifact2),
        output_report_json=tmp_path / "audit-side-effect.json",
    )

    assert audit["passed"] is False
    assert audit["checks"]["side_effect_action_traces_match"] is False


def test_persistent_state_chain_audit_rejects_broken_integrity_link(
    tmp_path: Path,
) -> None:
    config = HomeostaticControlConfig(
        preserve_adaptive_threshold_across_reset=True,
    )
    generation1 = HomeostaticSynapticController(config)
    _advance(generation1)
    artifact1 = generation1.export_persistent_state()
    generation2 = HomeostaticSynapticController(config)
    generation2.load_persistent_state(artifact1)
    _advance(generation2)
    artifact2 = generation2.export_persistent_state()
    report1 = _report(
        tmp_path,
        name="generation1",
        artifact=artifact1,
        io={"saved": True, "saved_integrity_sha256": artifact1["integrity_sha256"]},
    )
    report2 = _report(
        tmp_path,
        name="generation2",
        artifact=artifact2,
        loaded_artifact=artifact1,
        io={
            "loaded": True,
            "saved": True,
            "loaded_integrity_sha256": "not-generation1",
            "saved_integrity_sha256": artifact2["integrity_sha256"],
        },
    )

    audit = audit_phase2homeostasis_persistent_state_chain(
        generation1_report_json=_write(tmp_path / "report1.json", report1),
        generation2_report_json=_write(tmp_path / "report2.json", report2),
        generation1_state_json=_write(tmp_path / "state1.json", artifact1),
        generation2_state_json=_write(tmp_path / "state2.json", artifact2),
        output_report_json=tmp_path / "audit.json",
    )

    assert audit["passed"] is False
    assert audit["checks"]["state_chain_integrity_linked"] is False


def test_persistent_state_chain_audit_rejects_controller_schema_drift(
    tmp_path: Path,
) -> None:
    config = HomeostaticControlConfig(preserve_adaptive_threshold_across_reset=True)
    generation1 = HomeostaticSynapticController(config)
    _advance(generation1)
    artifact1 = generation1.export_persistent_state()
    generation2 = HomeostaticSynapticController(config)
    generation2.load_persistent_state(artifact1)
    _advance(generation2)
    artifact2 = generation2.export_persistent_state()
    artifact2["controller_schema_version"] = "reflexlm.homeostatic_synaptic_control.v1"
    _rehash(artifact2)
    report1 = _report(
        tmp_path,
        name="generation1-schema",
        artifact=artifact1,
        io={"saved": True, "saved_integrity_sha256": artifact1["integrity_sha256"]},
    )
    report2 = _report(
        tmp_path,
        name="generation2-schema",
        artifact=artifact2,
        loaded_artifact=artifact1,
        io={
            "loaded": True,
            "saved": True,
            "loaded_integrity_sha256": artifact1["integrity_sha256"],
            "saved_integrity_sha256": artifact2["integrity_sha256"],
        },
    )

    audit = audit_phase2homeostasis_persistent_state_chain(
        generation1_report_json=_write(tmp_path / "report1-schema.json", report1),
        generation2_report_json=_write(tmp_path / "report2-schema.json", report2),
        generation1_state_json=_write(tmp_path / "state1-schema.json", artifact1),
        generation2_state_json=_write(tmp_path / "state2-schema.json", artifact2),
        output_report_json=tmp_path / "audit-schema.json",
    )

    assert audit["passed"] is False
    assert audit["checks"]["artifact_schema_config_and_scope_match"] is False
