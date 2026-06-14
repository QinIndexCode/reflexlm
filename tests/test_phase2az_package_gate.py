import json
from pathlib import Path

from reflexlm.cli.audit_phase2az_package_gate import audit_phase2az_package_gate
from reflexlm.llm.native_cortex import OPEN_REPAIR_CAPABILITY_NAMES
from reflexlm.llm.native_nervous_package import write_native_nervous_package


SCHEMA = "phase2at.learned_bounded_patch_candidate.v1"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_adapter(path: Path) -> Path:
    (path / "backbone_adapter").mkdir(parents=True)
    (path / "tokenizer").mkdir(parents=True)
    (path / "head_config.json").write_text("{}", encoding="utf-8")
    (path / "native_heads.pt").write_text("tiny test checkpoint", encoding="utf-8")
    return path


def _matrix(*, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "ready_for_phase2az_package_gate": passed,
        "ready_for_epoch_making_architecture_claim": False,
        "metrics": {
            "repo_count": 3,
            "model_slot_accuracy": 1.0,
            "execution_success_rate": 0.8333333333333334,
        },
    }


def _postflight() -> dict:
    return {
        "passed": True,
        "ready_for_phase2ay_runtime_execution_eval": True,
        "ready_for_phase2ax_package": False,
        "ready_for_epoch_making_architecture_claim": False,
    }


def _training_summary(adapter_dir: Path) -> dict:
    return {
        "base_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_output_dir": str(adapter_dir),
        "open_repair_heads_enabled": True,
        "open_repair_capabilities": {
            name: True for name in OPEN_REPAIR_CAPABILITY_NAMES
        },
        "open_repair_training_contract": {
            "sealed_feedback_used": False,
            "recorded_patch_artifact_as_generation_target": False,
            "symbolic_generator_as_generation_target": False,
            "json_text_target": False,
            "freeform_patch_text_target": False,
            "patch_proposal_strategy": "learned_bounded_candidate",
        },
        "low_level_qwen_calls_target": 0,
    }


def _write_package(
    package_dir: Path,
    adapter_dir: Path,
    low_level_checkpoint: Path,
    *,
    caps: dict[str, bool] | None = None,
) -> None:
    write_native_nervous_package(
        package_dir,
        base_model_name="Qwen/Qwen2.5-7B-Instruct",
        native_head_path=adapter_dir,
        low_level_checkpoint_path=low_level_checkpoint,
        open_repair_capabilities=caps
        if caps is not None
        else {name: True for name in OPEN_REPAIR_CAPABILITY_NAMES},
        policy_label="phase2az_phase2ax_counterfactual_repair_full_7b_repo3_matrix",
        patch_proposal_strategy="learned_bounded_candidate",
        learned_patch_generation_enabled=True,
        patch_candidate_schema_version=SCHEMA,
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    adapter_dir = _make_adapter(tmp_path / "adapter")
    low_level_checkpoint = tmp_path / "model.pt"
    low_level_checkpoint.write_text("tiny low level checkpoint", encoding="utf-8")
    package_dir = tmp_path / "package"
    _write_package(package_dir, adapter_dir, low_level_checkpoint)
    return {
        "package_dir": package_dir,
        "adapter_dir": adapter_dir,
        "matrix_json": _write_json(tmp_path / "matrix.json", _matrix()),
        "postflight_json": _write_json(tmp_path / "postflight.json", _postflight()),
        "training_json": _write_json(
            tmp_path / "training.json",
            _training_summary(adapter_dir),
        ),
    }


def test_phase2az_package_gate_accepts_manifest_but_blocks_epoch(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)

    report = audit_phase2az_package_gate(
        package_path=inputs["package_dir"],
        phase2az_matrix_json=inputs["matrix_json"],
        full_postflight_json=inputs["postflight_json"],
        training_summary_json=inputs["training_json"],
    )

    assert report["passed"] is True
    assert report["ready_for_phase2az_packaged_adapter_runtime_smoke"] is True
    assert report["ready_for_phase2ax_package"] is False
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        "actual_package_loaded_runtime_execution_matrix"
        in report["unsupported_claims"]
    )


def test_phase2az_package_gate_rejects_wrong_adapter_path(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    other_adapter = _make_adapter(tmp_path / "other_adapter")

    report = audit_phase2az_package_gate(
        package_path=inputs["package_dir"],
        phase2az_matrix_json=inputs["matrix_json"],
        full_postflight_json=inputs["postflight_json"],
        training_summary_json=inputs["training_json"],
        expected_native_head_path=other_adapter,
    )

    assert report["passed"] is False
    assert report["checks"]["native_head_path_matches_training_adapter"] is False


def test_phase2az_package_gate_rejects_missing_open_repair_capability(
    tmp_path: Path,
) -> None:
    adapter_dir = _make_adapter(tmp_path / "adapter")
    low_level_checkpoint = tmp_path / "model.pt"
    low_level_checkpoint.write_text("tiny low level checkpoint", encoding="utf-8")
    package_dir = tmp_path / "package"
    caps = {name: True for name in OPEN_REPAIR_CAPABILITY_NAMES}
    caps["patch_proposal_head"] = False
    _write_package(package_dir, adapter_dir, low_level_checkpoint, caps=caps)

    report = audit_phase2az_package_gate(
        package_path=package_dir,
        phase2az_matrix_json=_write_json(tmp_path / "matrix.json", _matrix()),
        full_postflight_json=_write_json(tmp_path / "postflight.json", _postflight()),
        training_summary_json=_write_json(
            tmp_path / "training.json",
            _training_summary(adapter_dir),
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["all_open_repair_capabilities_declared"] is False
    assert "patch_proposal_head" in report["metrics"]["missing_open_repair_capabilities"]


def test_phase2az_package_gate_rejects_failed_matrix(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    report = audit_phase2az_package_gate(
        package_path=inputs["package_dir"],
        phase2az_matrix_json=_write_json(tmp_path / "failed_matrix.json", _matrix(passed=False)),
        full_postflight_json=inputs["postflight_json"],
        training_summary_json=inputs["training_json"],
    )

    assert report["passed"] is False
    assert report["checks"]["phase2az_matrix_passed"] is False
