import json
from pathlib import Path

from reflexlm.cli.audit_phase2au_pretrain_gate import audit_phase2au_pretrain_gate
from reflexlm.cli.build_phase2au_policy_required_head_dataset import (
    build_phase2au_policy_required_head_dataset,
)
from reflexlm.llm.candidate_features import (
    COMMAND_IDENTITY_FEATURE_START,
    COMMAND_IDENTITY_FEATURE_END,
    command_candidate_feature_rows,
    source_overlap_command_slot_prediction,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _task(index: int, *, split: str) -> dict:
    return {
        "task_id": f"phase2au:{split}:{index:05d}",
        "benchmark_family": "phase2au_policy_required_runtime_delta",
        "split": split,
        "repo_origin": f"https://github.com/example/repo-{index}.git",
        "repo_commit": "a" * 40,
        "problem_statement": "Repair behavior using runtime-visible evidence.",
        "evaluation_commands": ["python -m pytest -q tests/generated/test_behavior.py"],
        "artifact_paths": {"generated_tests": ["tests/generated/test_behavior.py"]},
        "allowed_write_scope": ["src/module.py", "src/helper.py"],
        "difficulty_axes": [
            "ambiguous_nonliteral_semantic",
            "multi_file_interaction",
            "negative_constraint",
            "stateful_verification",
        ],
        "runtime_visible_contract": {
            "policy_required_runtime_delta": True,
            "no_policy_symbolic_control_expected_to_fail": True,
            "no_direct_text_membership_or_ast_attr_oracle": True,
        },
        "expected_policy": {
            "patch_proposal": 1,
            "patch_operation": "apply_patch_and_rerun_tests",
            "patch_template": "behavioral_import_restoration",
            "bounded_edit_scope": 1,
            "rollback_safety": 1,
        },
        "candidate_policy_commands": [
            "phase2au_apply_candidate --repair-action repair_a structural_probe_hash=probe_alpha --verify pytest",
            "phase2au_apply_candidate --repair-action repair_b structural_probe_hash=probe_bravo --verify pytest",
        ],
        "runtime_visible_identity": {
            "command_identity_tokens": ["probe_bravo"],
            "identity_source": "runtime_visible_structural_probe_hashes",
        },
        "expected_repair_action": "repair_b",
        "sealed_feedback_used": False,
        "task_spec_sha256": "b" * 64,
    }


def test_phase2au_head_builder_preserves_claim_boundary(tmp_path: Path) -> None:
    train = _write_jsonl(tmp_path / "train.tasks.jsonl", [_task(0, split="train")])
    val = _write_jsonl(tmp_path / "val.tasks.jsonl", [_task(1, split="val")])

    manifest = build_phase2au_policy_required_head_dataset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        output_dir=tmp_path / "head",
        manifest_json=tmp_path / "manifest.json",
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "head" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["passed"] is True
    assert manifest["runtime_delta_supported"] is False
    assert manifest["package_allowed"] is False
    assert manifest["sealed_eval_allowed"] is False
    assert rows[0]["patch_proposal_label"] == 1
    assert rows[0]["command_slot"] == 1
    assert len(rows[0]["candidate_commands"]) == 2
    assert "command_identity_tokens = probe_bravo" in rows[0]["state_prompt"]
    assert rows[0]["nsi_reference"]["command_identity_slot:1"] > rows[0]["nsi_reference"]["command_identity_slot:0"]
    assert rows[0]["nsi_reference"]["command_identity_confidence"] > 0.0
    feature_rows = command_candidate_feature_rows(
        rows[0]["state_prompt"],
        rows[0]["candidate_commands"],
        nsi_reference=rows[0]["nsi_reference"],
    )
    assert (
        sum(feature_rows[1][COMMAND_IDENTITY_FEATURE_START:COMMAND_IDENTITY_FEATURE_END])
        > sum(feature_rows[0][COMMAND_IDENTITY_FEATURE_START:COMMAND_IDENTITY_FEATURE_END])
    )
    assert source_overlap_command_slot_prediction(rows[0]["state_prompt"], rows[0]["candidate_commands"]) == 0
    assert rows[0]["learned_patch_policy_target"]["recorded_patch_text_as_target"] is False
    assert "patch_diff" not in rows[0]["learned_patch_policy_target"]


def test_phase2au_pretrain_gate_requires_passed_task_gate(tmp_path: Path) -> None:
    train = _write_jsonl(tmp_path / "train.tasks.jsonl", [_task(0, split="train")])
    val = _write_jsonl(tmp_path / "val.tasks.jsonl", [_task(1, split="val")])
    manifest = build_phase2au_policy_required_head_dataset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        output_dir=tmp_path / "head",
        manifest_json=tmp_path / "manifest.json",
    )
    assert manifest["passed"] is True
    task_gate = _write_json(
        tmp_path / "task_gate.json",
        {"passed": True, "metrics": {"row_count": 64}},
    )

    report = audit_phase2au_pretrain_gate(
        task_gate_json=task_gate,
        head_manifest_json=tmp_path / "manifest.json",
        train_jsonl=tmp_path / "head" / "train.jsonl",
        val_jsonl=tmp_path / "head" / "val.jsonl",
    )

    assert report["passed"] is True
    assert report["ready_for_phase2au_smoke_training"] is True
    assert report["checks"]["package_and_sealed_blocked_before_runtime_delta"] is True


def test_phase2au_pretrain_gate_rejects_patch_diff_target(tmp_path: Path) -> None:
    train = _write_jsonl(tmp_path / "train.tasks.jsonl", [_task(0, split="train")])
    val = _write_jsonl(tmp_path / "val.tasks.jsonl", [_task(1, split="val")])
    build_phase2au_policy_required_head_dataset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        output_dir=tmp_path / "head",
        manifest_json=tmp_path / "manifest.json",
    )
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "head" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows[0]["learned_patch_policy_target"]["patch_diff"] = "--- leaked"
    _write_jsonl(tmp_path / "head" / "train.jsonl", train_rows)
    task_gate = _write_json(tmp_path / "task_gate.json", {"passed": True})

    report = audit_phase2au_pretrain_gate(
        task_gate_json=task_gate,
        head_manifest_json=tmp_path / "manifest.json",
        train_jsonl=tmp_path / "head" / "train.jsonl",
        val_jsonl=tmp_path / "head" / "val.jsonl",
    )

    assert report["passed"] is False
    assert report["checks"]["no_recorded_or_symbolic_patch_target"] is False
