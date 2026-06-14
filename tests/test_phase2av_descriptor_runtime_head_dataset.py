import json
from pathlib import Path

from reflexlm.cli.build_phase2av_descriptor_runtime_head_dataset import (
    build_phase2av_descriptor_runtime_head_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _task(split: str, index: int = 0) -> dict:
    return {
        "task_id": f"phase2av:{split}:{index:05d}",
        "benchmark_family": "phase2av_graded_descriptor_runtime",
        "claim_boundary": "phase2av_pretrain_gate_before_learned_descriptor_runtime_claim",
        "split": split,
        "source_kind": "public_repo",
        "repo_origin": "https://github.com/example/repo.git",
        "repo_commit": "a" * 40,
        "problem_statement": "Generated behavioral test fails on public source.",
        "current_visible_text": "Use runtime-visible generated test failure only.",
        "difficulty_axes": ["stateful_verification"],
        "runtime_visible_contract": {
            "no_sealed_feedback": True,
            "no_freeform_patch_generation": True,
        },
        "runtime_visible_evidence": {
            "changed_files": ["src/mod.py"],
            "structural_probe_hashes": ["probe_good"],
            "pytest_before_patch": {
                "stdout_excerpt": "test_behavioral_module_import_restored_0 failed: NameError: os is not defined"
            },
        },
        "repair_candidates": [
            {
                "repair_action": "repair_bad",
                "structural_probe_hash": "probe_bad",
                "target_symbol": "wrong",
            },
            {
                "repair_action": "repair_good",
                "structural_probe_hash": "probe_good",
                "target_symbol": "right",
            },
        ],
        "candidate_policy_commands": [
            "phase2av_apply_descriptor_candidate --repair-action repair_bad structural_probe_hash=probe_bad --verify pytest",
            "phase2av_apply_descriptor_candidate --repair-action repair_good structural_probe_hash=probe_good --verify pytest",
        ],
        "expected_repair_action": "repair_good",
        "expected_policy": {
            "patch_proposal": 1,
            "patch_operation": "insert_import",
            "patch_template": "import_restoration",
            "rollback_safety": 1,
            "bounded_edit_scope": 1,
            "verification_state": 1,
        },
        "learned_patch_descriptor_target": {
            "schema_version": "phase2at.learned_bounded_patch_candidate.v1",
            "target_source": "runtime_visible_structural_descriptor_not_recorded_patch",
            "target_path": "src/mod.py",
            "operation": "insert_import",
            "anchor": {"kind": "runtime_structural_probe", "probe_hash": "probe_good"},
            "before_fragment_hash": "deadbeef",
            "after_fragment_template_id": "import_restoration",
            "literal_or_symbol_payload": {"candidate_structural_probe_hash": "probe_good"},
            "safety_constraints": {
                "allowed_paths": ["src/mod.py"],
                "forbid_unbounded_diff_text": True,
            },
            "verification_command_slot": 0,
        },
        "artifact_paths": {"generated_tests": ["tests/generated/test_behavior.py"]},
        "sealed_feedback_used": False,
        "task_spec_sha256": "b" * 64,
    }


def test_phase2av_head_builder_emits_native_head_rows_without_gold_prompt_leak(
    tmp_path: Path,
) -> None:
    train = _write_jsonl(tmp_path / "train.tasks.jsonl", [_task("train")])
    val = _write_jsonl(tmp_path / "val.tasks.jsonl", [_task("val")])

    manifest = build_phase2av_descriptor_runtime_head_dataset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        output_dir=tmp_path / "head",
        manifest_json=tmp_path / "manifest.json",
    )
    row = json.loads((tmp_path / "head" / "train.jsonl").read_text().splitlines()[0])

    assert manifest["passed"] is True
    assert manifest["smoke_training_allowed"] is True
    assert manifest["full_training_allowed"] is False
    assert manifest["package_allowed"] is False
    assert row["prompt_style"] == "phase2av_graded_descriptor_runtime_head_v1"
    assert row["command_slot"] == 1
    assert row["patch_operation_label"] >= 0
    assert row["patch_template_slot"] >= 0
    assert row["learned_patch_policy_target"]["recorded_patch_text_as_target"] is False
    assert "expected_policy" not in row["state_prompt"]
    assert "learned_patch_descriptor_target" not in row["state_prompt"]
    assert "patch_operation=insert_import" not in row["state_prompt"]
    assert "test_behavioral_module_import_restored_0" not in row["state_prompt"]
    assert "test_behavioral_module_repair_case" in row["state_prompt"]
    assert "Runtime-visible repair evidence:" in row["state_prompt"]
    assert row["state_prompt"].index("Runtime-visible repair evidence:") < row[
        "state_prompt"
    ].index("task_id=")
    assert row["state_prompt"].index("Runtime-visible repair evidence:") < row[
        "state_prompt"
    ].index("Runtime-visible evidence:")
    assert "command_identity_tokens=probe_good" in row["state_prompt"]
    assert "descriptor_failure_family" in row["state_prompt"]
    assert (
        row["nsi_reference"]["descriptor_failure_family"]
        == "missing_import_or_symbol_runtime"
    )
    assert row["nsi_reference"]["command_identity_slot:1"] > 0.0
    assert row["nsi_reference"]["command_identity_confidence"] > 0.0
    assert "NameError" in row["state_prompt"]
    assert "repair_good" in row["state_prompt"]
    assert "repair_bad" in row["state_prompt"]
    assert row["source_task_manifest"]["claim_boundary"] == (
        "phase2av_head_training_rows_not_runtime_delta_evidence"
    )


def test_phase2av_head_builder_can_augment_train_candidate_order_without_touching_val(
    tmp_path: Path,
) -> None:
    train = _write_jsonl(tmp_path / "train.tasks.jsonl", [_task("train")])
    val = _write_jsonl(tmp_path / "val.tasks.jsonl", [_task("val")])

    manifest = build_phase2av_descriptor_runtime_head_dataset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        output_dir=tmp_path / "head",
        manifest_json=tmp_path / "manifest.json",
        augment_train_candidate_order=True,
    )
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "head" / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    val_rows = [
        json.loads(line)
        for line in (tmp_path / "head" / "val.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert manifest["candidate_order_augmentation"] == {
        "train_enabled": True,
        "source_train_rows": 1,
        "effective_train_rows": 2,
        "val_enabled": False,
    }
    assert [row["command_slot"] for row in train_rows] == [0, 1]
    assert len(val_rows) == 1
    assert val_rows[0]["command_slot"] == 1
    assert train_rows[0]["command"] == train_rows[0]["candidate_commands"][0]
    assert train_rows[1]["command"] == train_rows[1]["candidate_commands"][1]
    assert train_rows[0]["learned_patch_candidate_target"]["verification_command_slot"] == 0
    assert train_rows[1]["learned_patch_candidate_target"]["verification_command_slot"] == 1
