import json
from pathlib import Path

from reflexlm.cli.audit_phase2au_package_gate import (
    REQUIRED_SCHEMA_VERSION,
    audit_phase2au_package_gate,
)
from reflexlm.llm.native_cortex import OPEN_REPAIR_CAPABILITY_NAMES
from reflexlm.llm.native_nervous_package import write_native_nervous_package


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(adapter_dir: Path) -> dict:
    return {
        "base_model_name": "artifacts/models/Qwen2.5-0.5B-Instruct",
        "adapter_output_dir": str(adapter_dir),
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "open_repair_training_contract": {
            "sealed_feedback_used": False,
            "learned_patch_candidate_targets": True,
            "recorded_patch_artifact_as_generation_target": False,
            "symbolic_generator_as_generation_target": False,
            "freeform_patch_text_target": False,
            "json_text_target": False,
            "low_level_qwen_calls_target": 0,
        },
    }


def _postflight(*, holdout: bool = False) -> dict:
    return {
        "passed": True,
        "claim_boundary": "phase2au_holdout_delta_supported_for_capacity_smoke_not_claim_upgrade"
        if holdout
        else "phase2au_smoke_ready_for_full_training_not_runtime_delta_evidence",
        "metrics": {"model_minus_source_overlap": 0.45},
    }


def test_phase2au_package_gate_accepts_capacity_package_for_runtime_eval_only(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    adapter_dir = tmp_path / "adapter"
    write_native_nervous_package(
        package_dir,
        base_model_name="artifacts/models/Qwen2.5-0.5B-Instruct",
        native_head_path=adapter_dir,
        low_level_checkpoint_path="low.pt",
        policy_label="phase2au_qwen0_5b_policy_required_identityrefv2",
        open_repair_capabilities={name: True for name in OPEN_REPAIR_CAPABILITY_NAMES},
        patch_proposal_strategy="learned_bounded_candidate",
        learned_patch_generation_enabled=True,
        patch_candidate_schema_version=REQUIRED_SCHEMA_VERSION,
    )

    report = audit_phase2au_package_gate(
        package_path=package_dir,
        training_summary_json=_write(tmp_path / "summary.json", _summary(adapter_dir)),
        smoke_postflight_json=_write(tmp_path / "smoke.json", _postflight()),
        holdout_postflight_json=_write(
            tmp_path / "holdout.json", _postflight(holdout=True)
        ),
    )

    assert report["passed"] is True
    assert report["ready_for_phase2au_runtime_delta_eval"] is True
    assert (
        "phase2au_capacity_package_ready_for_bounded_runtime_control_eval"
        in report["supported_claims"]
    )
    assert "phase2au_runtime_delta_before_full_no_policy_execution" in report[
        "unsupported_claims"
    ]


def test_phase2au_package_gate_rejects_old_or_symbolic_package(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    adapter_dir = tmp_path / "adapter"
    write_native_nervous_package(
        package_dir,
        base_model_name="artifacts/models/Qwen2.5-0.5B-Instruct",
        native_head_path=adapter_dir,
        low_level_checkpoint_path="low.pt",
        policy_label="phase2au_qwen0_5b_policy_required_identityrefv2",
        open_repair_capabilities={name: True for name in OPEN_REPAIR_CAPABILITY_NAMES},
        patch_proposal_strategy="symbolic_runtime_generator",
    )

    report = audit_phase2au_package_gate(
        package_path=package_dir,
        training_summary_json=_write(tmp_path / "summary.json", _summary(adapter_dir)),
        smoke_postflight_json=_write(tmp_path / "smoke.json", _postflight()),
        holdout_postflight_json=_write(
            tmp_path / "holdout.json", _postflight(holdout=True)
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["patch_strategy_is_learned_bounded_candidate"] is False
    assert "do_not_run_phase2au_runtime_delta_eval_with_this_package" in report[
        "blocked_actions"
    ]


def test_phase2au_package_gate_rejects_holdout_claim_boundary_upgrade(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    adapter_dir = tmp_path / "adapter"
    write_native_nervous_package(
        package_dir,
        base_model_name="artifacts/models/Qwen2.5-0.5B-Instruct",
        native_head_path=adapter_dir,
        low_level_checkpoint_path="low.pt",
        policy_label="phase2au_qwen0_5b_policy_required_identityrefv2",
        open_repair_capabilities={name: True for name in OPEN_REPAIR_CAPABILITY_NAMES},
        patch_proposal_strategy="learned_bounded_candidate",
        learned_patch_generation_enabled=True,
        patch_candidate_schema_version=REQUIRED_SCHEMA_VERSION,
    )
    holdout = _postflight(holdout=True)
    holdout["claim_boundary"] = "phase2au_runtime_delta_supported"

    report = audit_phase2au_package_gate(
        package_path=package_dir,
        training_summary_json=_write(tmp_path / "summary.json", _summary(adapter_dir)),
        smoke_postflight_json=_write(tmp_path / "smoke.json", _postflight()),
        holdout_postflight_json=_write(tmp_path / "holdout.json", holdout),
    )

    assert report["passed"] is False
    assert report["checks"]["holdout_still_not_runtime_delta_claim"] is False
