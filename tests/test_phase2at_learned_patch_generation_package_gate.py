import json
from pathlib import Path

import pytest

from reflexlm.cli.audit_phase2at_learned_patch_generation_package_gate import (
    audit_phase2at_learned_patch_generation_package_gate,
)
from reflexlm.llm.native_cortex import OPEN_REPAIR_CAPABILITY_NAMES
from reflexlm.llm.native_nervous_package import write_native_nervous_package


SCHEMA = "phase2at.learned_bounded_patch_candidate.v1"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2at_package_gate_rejects_old_open_repair_control_schema(tmp_path: Path) -> None:
    package_dir = tmp_path / "old_pkg"
    write_native_nervous_package(
        package_dir,
        base_model_name="qwen",
        native_head_path="heads",
        low_level_checkpoint_path="low",
        open_repair_capabilities={name: True for name in OPEN_REPAIR_CAPABILITY_NAMES},
    )

    report = audit_phase2at_learned_patch_generation_package_gate(package_path=package_dir)

    assert report["passed"] is False
    assert report["checks"]["patch_strategy_is_learned_bounded_candidate"] is False
    assert report["checks"]["learned_patch_generation_enabled"] is False
    assert "learned_patch_generation" in report["unsupported_claims"]


def test_phase2at_package_gate_accepts_explicit_learned_candidate_schema(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "learned_pkg"
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "open_repair_training_contract": {
                "sealed_feedback_used": False,
                "learned_patch_candidate_targets": True,
                "recorded_patch_artifact_as_generation_target": False,
                "symbolic_generator_as_generation_target": False,
            }
        },
    )
    write_native_nervous_package(
        package_dir,
        base_model_name="qwen",
        native_head_path="heads",
        low_level_checkpoint_path="low",
        open_repair_capabilities={name: True for name in OPEN_REPAIR_CAPABILITY_NAMES},
        patch_proposal_strategy="learned_bounded_candidate",
        learned_patch_generation_enabled=True,
        patch_candidate_schema_version=SCHEMA,
    )

    report = audit_phase2at_learned_patch_generation_package_gate(
        package_path=package_dir,
        training_summary_json=summary,
    )

    assert report["passed"] is True
    assert (
        "phase2at_package_ready_for_learned_bounded_patch_generation_eval"
        in report["supported_claims"]
    )
    assert report["metrics"]["patch_candidate_schema_version"] == SCHEMA


def test_phase2at_package_gate_rejects_training_summary_relabel(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "learned_pkg"
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "open_repair_training_contract": {
                "sealed_feedback_used": False,
                "learned_patch_candidate_targets": True,
                "recorded_patch_artifact_as_generation_target": True,
                "symbolic_generator_as_generation_target": False,
            }
        },
    )
    write_native_nervous_package(
        package_dir,
        base_model_name="qwen",
        native_head_path="heads",
        low_level_checkpoint_path="low",
        open_repair_capabilities={name: True for name in OPEN_REPAIR_CAPABILITY_NAMES},
        patch_proposal_strategy="learned_bounded_candidate",
        learned_patch_generation_enabled=True,
        patch_candidate_schema_version=SCHEMA,
    )

    report = audit_phase2at_learned_patch_generation_package_gate(
        package_path=package_dir,
        training_summary_json=summary,
    )

    assert report["passed"] is False
    assert report["checks"]["training_summary_blocks_recorded_patch_relabel"] is False


def test_write_package_rejects_inconsistent_learned_generation_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="learned_patch_generation_enabled requires"):
        write_native_nervous_package(
            tmp_path / "bad_pkg",
            base_model_name="qwen",
            native_head_path="heads",
            low_level_checkpoint_path="low",
            patch_proposal_strategy="symbolic_runtime_generator",
            learned_patch_generation_enabled=True,
        )
