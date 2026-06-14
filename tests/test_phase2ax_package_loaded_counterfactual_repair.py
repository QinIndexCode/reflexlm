from __future__ import annotations

import json
from pathlib import Path

from reflexlm.cli.audit_phase2ax_package_loaded_counterfactual_repair import (
    audit_phase2ax_package_loaded_counterfactual_repair,
)
from reflexlm.cli.audit_phase2ax_runtime_pretrain_gate import (
    build_phase2ax_runtime_pretrain_gate,
)
from reflexlm.cli.audit_phase2ax_smoke_postflight import (
    audit_phase2ax_smoke_postflight,
)
from reflexlm.cli.audit_phase2ax_full_postflight import (
    audit_phase2ax_full_postflight,
)
from reflexlm.cli.build_phase2ax_package_loaded_counterfactual_repair import (
    BENCHMARK_FAMILY,
    build_phase2ax_package_loaded_counterfactual_repair,
)
from reflexlm.cli.build_phase2ax_head_dataset import build_phase2ax_head_dataset
from reflexlm.cli.build_phase2ax_full_head_dataset import build_phase2ax_full_head_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source_row(index: int, action: str, *, repo: str = "repo") -> dict:
    return {
        "task_id": f"source-{index}",
        "source_kind": "public_repo",
        "repo_origin": f"https://github.com/example/{repo}.git",
        "repo_commit": "abc123",
        "artifact_paths": {"generated_tests": [f"artifacts/row_{index}/generated_test.py"]},
        "expected_repair_action": action,
        "repair_candidates": [
            {"repair_action": action, "structural_probe_hash": f"probe-{index}"}
        ],
        "runtime_visible_evidence": {
            "changed_files": ["src/mod.py"],
            "watched_files": [f"tests/test_{index}.py"],
            "structural_probe_hashes": [f"probe-{index}"],
            "repair_modes": ["mode"],
        },
        "learned_patch_descriptor_target": {
            "operation": "replace_attribute",
            "after_fragment_template_id": "call_attribute_restoration",
            "target_path": "src/mod.py",
            "literal_or_symbol_payload": {"target_symbol_hash": f"sym-{index}"},
        },
        "runtime_visible_contract": {
            "no_sealed_feedback": True,
            "no_gold_hint": True,
            "no_candidate_slot_marker": True,
            "no_freeform_patch_generation": True,
        },
        "sealed_feedback_used": False,
    }


def _linked_source_row(index: int, probe: str, *, repo: str = "repo") -> dict:
    row = _source_row(index, f"structural_repair_{probe[:12]}", repo=repo)
    row["runtime_visible_evidence"]["structural_probe_hashes"] = [probe]
    row["learned_patch_descriptor_target"]["literal_or_symbol_payload"][
        "candidate_structural_probe_hash"
    ] = probe
    return row


def test_phase2ax_builder_and_audit_accept_counterfactual_pairs(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [
            _source_row(0, "repair_a"),
            _source_row(1, "repair_b"),
            _source_row(2, "repair_c"),
            _source_row(3, "repair_d"),
        ],
    )
    tasks = tmp_path / "phase2ax.jsonl"
    metadata = tmp_path / "phase2ax.metadata.json"

    build_report = build_phase2ax_package_loaded_counterfactual_repair(
        source_tasks_jsonl=source,
        output_jsonl=tasks,
        metadata_json=metadata,
        max_pairs=2,
        min_pairs=2,
    )
    audit = audit_phase2ax_package_loaded_counterfactual_repair(
        tasks_jsonl=tasks,
        metadata_json=metadata,
        min_pairs=2,
    )

    rows = [json.loads(line) for line in tasks.read_text(encoding="utf-8").splitlines()]
    assert build_report["passed"] is True
    assert audit["passed"] is True
    assert audit["checks"]["current_surface_identical_within_pairs"] is True
    assert audit["checks"]["prior_context_differs_within_pairs"] is True
    assert audit["metrics"]["current_only_baseline"]["accuracy"] == 0.5
    assert all(row["benchmark_family"] == BENCHMARK_FAMILY for row in rows)
    assert all(row["sealed_feedback_used"] is False for row in rows)


def test_phase2ax_audit_rejects_current_surface_baseline_too_strong(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [
            _source_row(0, "repair_a"),
            _source_row(1, "repair_b"),
        ],
    )
    tasks = tmp_path / "phase2ax.jsonl"
    metadata = tmp_path / "phase2ax.metadata.json"
    build_phase2ax_package_loaded_counterfactual_repair(
        source_tasks_jsonl=source,
        output_jsonl=tasks,
        metadata_json=metadata,
        max_pairs=1,
        min_pairs=1,
    )

    audit = audit_phase2ax_package_loaded_counterfactual_repair(
        tasks_jsonl=tasks,
        metadata_json=metadata,
        min_pairs=1,
        max_current_only_baseline=0.49,
    )

    assert audit["passed"] is False
    assert audit["checks"]["current_only_baseline_below_threshold"] is False


def test_phase2ax_audit_rejects_visible_gold_marker(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [
            _source_row(0, "repair_a"),
            _source_row(1, "repair_b"),
        ],
    )
    tasks = tmp_path / "phase2ax.jsonl"
    metadata = tmp_path / "phase2ax.metadata.json"
    build_phase2ax_package_loaded_counterfactual_repair(
        source_tasks_jsonl=source,
        output_jsonl=tasks,
        metadata_json=metadata,
        max_pairs=1,
        min_pairs=1,
    )
    rows = [json.loads(line) for line in tasks.read_text(encoding="utf-8").splitlines()]
    rows[0]["current_visible_text"] += " gold_hint=repair_a"
    _write_jsonl(tasks, rows)

    audit = audit_phase2ax_package_loaded_counterfactual_repair(
        tasks_jsonl=tasks,
        metadata_json=metadata,
        min_pairs=1,
    )

    assert audit["passed"] is False
    assert audit["checks"]["no_forbidden_visible_markers"] is False


def test_phase2ax_runtime_pretrain_gate_accepts_prior_resolvable_pairs(
    tmp_path: Path,
) -> None:
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [
            _linked_source_row(0, "aaaabbbbcccc1111"),
            _linked_source_row(1, "ddddeeeeffff2222"),
            _linked_source_row(2, "1111222233334444"),
            _linked_source_row(3, "5555666677778888"),
        ],
    )
    tasks = tmp_path / "phase2ax.jsonl"
    metadata = tmp_path / "phase2ax.metadata.json"
    build_phase2ax_package_loaded_counterfactual_repair(
        source_tasks_jsonl=source,
        output_jsonl=tasks,
        metadata_json=metadata,
        max_pairs=2,
        min_pairs=2,
    )
    data_health = audit_phase2ax_package_loaded_counterfactual_repair(
        tasks_jsonl=tasks,
        metadata_json=metadata,
        min_pairs=2,
    )
    data_health_path = _write_json(tmp_path / "phase2ax.data_health.json", data_health)

    gate = build_phase2ax_runtime_pretrain_gate(
        tasks_jsonl=tasks,
        metadata_json=metadata,
        data_health_json=data_health_path,
        min_prior_resolver_accuracy=1.0,
    )

    assert data_health["passed"] is True
    assert gate["passed"] is True
    assert gate["checks"]["current_only_baseline_nonzero"] is True
    assert gate["metrics"]["current_only"]["accuracy"] == 0.5
    assert gate["metrics"]["prior_runtime_resolver"]["accuracy"] == 1.0
    assert gate["metrics"]["wrong_cache"]["accuracy"] == 0.0


def test_phase2ax_runtime_pretrain_gate_rejects_unlinked_candidates(
    tmp_path: Path,
) -> None:
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [
            _source_row(0, "repair_a"),
            _source_row(1, "repair_b"),
        ],
    )
    tasks = tmp_path / "phase2ax.jsonl"
    metadata = tmp_path / "phase2ax.metadata.json"
    build_phase2ax_package_loaded_counterfactual_repair(
        source_tasks_jsonl=source,
        output_jsonl=tasks,
        metadata_json=metadata,
        max_pairs=1,
        min_pairs=1,
    )

    gate = build_phase2ax_runtime_pretrain_gate(
        tasks_jsonl=tasks,
        metadata_json=metadata,
        min_prior_resolver_accuracy=1.0,
    )

    assert gate["passed"] is False
    assert gate["checks"]["candidate_prior_link_present"] is False
    assert "revise_phase2ax_candidate_prior_link_before_training" in gate["blocked_actions"]


def test_phase2ax_head_builder_keeps_pairs_intact_and_blocks_claim_upgrade(
    tmp_path: Path,
) -> None:
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [
            _linked_source_row(0, "aaaabbbbcccc1111"),
            _linked_source_row(1, "ddddeeeeffff2222"),
            _linked_source_row(2, "1111222233334444"),
            _linked_source_row(3, "5555666677778888"),
        ],
    )
    tasks = tmp_path / "phase2ax.jsonl"
    metadata = tmp_path / "phase2ax.metadata.json"
    build_phase2ax_package_loaded_counterfactual_repair(
        source_tasks_jsonl=source,
        output_jsonl=tasks,
        metadata_json=metadata,
        max_pairs=2,
        min_pairs=2,
    )
    data_health = audit_phase2ax_package_loaded_counterfactual_repair(
        tasks_jsonl=tasks,
        metadata_json=metadata,
        min_pairs=2,
    )
    data_health_path = _write_json(tmp_path / "phase2ax.data_health.json", data_health)
    pretrain = build_phase2ax_runtime_pretrain_gate(
        tasks_jsonl=tasks,
        metadata_json=metadata,
        data_health_json=data_health_path,
        min_prior_resolver_accuracy=1.0,
    )
    pretrain_path = _write_json(tmp_path / "phase2ax.pretrain.json", pretrain)

    manifest = build_phase2ax_head_dataset(
        tasks_jsonl=tasks,
        output_dir=tmp_path / "head",
        manifest_json=tmp_path / "head_manifest.json",
        data_health_json=data_health_path,
        pretrain_gate_json=pretrain_path,
        train_pair_count=1,
    )
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "head" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val_rows = [
        json.loads(line)
        for line in (tmp_path / "head" / "val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["passed"] is True
    assert manifest["smoke_training_allowed"] is True
    assert manifest["full_training_allowed"] is False
    assert manifest["package_allowed"] is False
    assert manifest["sealed_eval_allowed"] is False
    assert manifest["command_slot_distribution"]["train"] == {"0": 1, "1": 1}
    assert manifest["command_slot_distribution"]["val"] == {"0": 1, "1": 1}
    assert {row["source_task_manifest"]["pair_id"] for row in train_rows} == {
        "phase2ax_pair_00000"
    }
    assert {row["source_task_manifest"]["pair_id"] for row in val_rows} == {
        "phase2ax_pair_00001"
    }
    row = train_rows[0]
    assert row["prompt_style"] == "phase2ax_package_loaded_counterfactual_repair_head_v1"
    assert "Prior runtime evidence:" in row["state_prompt"]
    assert "Masked current repair surface:" in row["state_prompt"]
    assert "Runtime-visible repair evidence:" not in row["state_prompt"]
    assert "Structured command identity sidecar:" not in row["state_prompt"]
    assert "expected_repair_action" not in row["state_prompt"]
    assert row["nsi_reference"]["command_identity_confidence"] > 0.0
    assert row["learned_patch_policy_target"]["recorded_patch_text_as_target"] is False


def _phase2ax_training_summary(
    *,
    val_accuracy: float = 1.0,
    source_accuracy: float = 0.5,
) -> dict:
    return {
        "adapter_name": "phase2ax_package_loaded_counterfactual_repair_r16_alpha32_lr1e-4_len256_smoke",
        "train_examples": 32,
        "val_examples": 32,
        "config_hash": "abc123",
        "use_pairwise_command_reranker": False,
        "open_repair_heads_enabled": True,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "effective_split_hashes": {
            "phase2c_head_train": "train-hash",
            "phase2c_head_val": "val-hash",
        },
        "source_overlap_command_slot_baseline": {
            "val": {"accuracy": source_accuracy}
        },
        "pairwise_candidate_encoding": {
            "val": {"pairwise_scored_candidates": 0}
        },
        "history": [
            {
                "train_loss": 0.5,
                "train_elapsed_seconds": 10.0,
                "train_steps_per_second": 1.0,
                "val_metrics": {
                    "loss": 0.2,
                    "command_slot_accuracy": val_accuracy,
                    "command_slot_count": 32,
                    "patch_operation_accuracy": 0.5,
                    "patch_operation_count": 32,
                    "patch_template_slot_accuracy": 0.4,
                    "patch_template_slot_count": 32,
                },
            }
        ],
    }


def _phase2ax_pretrain_gate() -> dict:
    return {
        "passed": True,
        "metrics": {
            "current_only": {"accuracy": 0.5},
            "wrong_cache": {"accuracy": 0.0},
        },
    }


def _phase2ax_head_manifest(*, package_allowed: bool = False) -> dict:
    return {
        "passed": True,
        "package_allowed": package_allowed,
        "sealed_eval_allowed": False,
        "effective_split_hashes": {
            "phase2ax_head_train": "manifest-train",
            "phase2ax_head_val": "manifest-val",
        },
    }


def _phase2ax_full_training_summary(
    *,
    val_accuracy: float = 1.0,
    source_accuracy: float = 0.5,
) -> dict:
    return {
        "adapter_name": "phase2ax_package_loaded_counterfactual_repair_r16_alpha32_full128_val128_seed13",
        "train_examples": 128,
        "val_examples": 128,
        "config_hash": "full123",
        "use_pairwise_command_reranker": False,
        "open_repair_heads_enabled": True,
        "open_repair_training_contract": {
            "sealed_feedback_used": False,
            "freeform_patch_text_target": False,
            "no_json_motor_target": True,
            "low_level_qwen_calls_target": 0,
        },
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "effective_split_hashes": {
            "phase2c_head_train": "train-hash",
            "phase2c_head_val": "val-hash",
        },
        "source_overlap_command_slot_baseline": {
            "val": {"accuracy": source_accuracy}
        },
        "pairwise_candidate_encoding": {
            "val": {"pairwise_scored_candidates": 0}
        },
        "history": [
            {
                "train_loss": 0.4,
                "train_elapsed_seconds": 100.0,
                "train_steps_per_second": 0.5,
                "val_metrics": {
                    "loss": 0.3,
                    "command_slot_accuracy": val_accuracy,
                    "command_slot_count": 128,
                    "patch_operation_accuracy": 0.3359375,
                    "patch_operation_count": 128,
                    "patch_template_slot_accuracy": 0.3203125,
                    "patch_template_slot_count": 128,
                    "patch_target_file_slot_accuracy": 1.0,
                },
            }
        ],
    }


def _phase2ax_full_manifest(
    *,
    package_allowed: bool = False,
    repo_disjoint: bool = True,
    passed: bool = True,
) -> dict:
    overlap = [] if repo_disjoint else ["https://github.com/example/shared.git"]
    return {
        "passed": passed,
        "full_training_allowed": passed,
        "package_allowed": package_allowed,
        "sealed_eval_allowed": False,
        "train_rows": 128,
        "val_rows": 128,
        "checks": {
            "repo_origin_disjoint": repo_disjoint,
        },
        "repo_origins": {
            "train": ["https://github.com/example/train.git"],
            "val": ["https://github.com/example/val.git"],
            "overlap": overlap,
        },
        "effective_split_hashes": {
            "phase2ax_full_head_train": "manifest-train",
            "phase2ax_full_head_val": "manifest-val",
        },
    }


def _phase2ax_full_pretrain_gate() -> dict:
    return {
        "passed": True,
        "metrics": {
            "current_only": {"accuracy": 0.5},
            "wrong_cache": {"accuracy": 0.0},
            "prior_runtime_resolver": {"accuracy": 1.0},
        },
    }


def _phase2ax_smoke_postflight() -> dict:
    return {
        "passed": True,
        "ready_for_phase2ax_full_nonsealed_training": True,
    }


def test_phase2ax_smoke_postflight_accepts_smoke_but_blocks_package(
    tmp_path: Path,
) -> None:
    report = audit_phase2ax_smoke_postflight(
        training_summary_json=_write_json(
            tmp_path / "summary.json", _phase2ax_training_summary()
        ),
        data_health_json=_write_json(tmp_path / "data_health.json", {"passed": True}),
        pretrain_gate_json=_write_json(tmp_path / "pretrain.json", _phase2ax_pretrain_gate()),
        head_manifest_json=_write_json(tmp_path / "manifest.json", _phase2ax_head_manifest()),
    )

    assert report["passed"] is True
    assert report["ready_for_phase2ax_full_nonsealed_training"] is True
    assert report["ready_for_package_or_execution_claim"] is False
    assert report["ready_for_sealed_eval"] is False
    assert report["metrics"]["model_minus_source_overlap"] == 0.5
    assert report["metrics"]["patch_descriptor_evaluable"] is True


def test_phase2ax_smoke_postflight_rejects_source_overlap_tie(
    tmp_path: Path,
) -> None:
    report = audit_phase2ax_smoke_postflight(
        training_summary_json=_write_json(
            tmp_path / "summary.json",
            _phase2ax_training_summary(val_accuracy=0.85, source_accuracy=0.85),
        ),
        data_health_json=_write_json(tmp_path / "data_health.json", {"passed": True}),
        pretrain_gate_json=_write_json(tmp_path / "pretrain.json", _phase2ax_pretrain_gate()),
        head_manifest_json=_write_json(tmp_path / "manifest.json", _phase2ax_head_manifest()),
    )

    assert report["passed"] is False
    assert report["checks"]["model_beats_source_overlap"] is False
    assert "do_not_start_phase2ax_full_training" in report["blocked_actions"]


def test_phase2ax_smoke_postflight_rejects_manifest_package_permission(
    tmp_path: Path,
) -> None:
    report = audit_phase2ax_smoke_postflight(
        training_summary_json=_write_json(
            tmp_path / "summary.json", _phase2ax_training_summary()
        ),
        data_health_json=_write_json(tmp_path / "data_health.json", {"passed": True}),
        pretrain_gate_json=_write_json(tmp_path / "pretrain.json", _phase2ax_pretrain_gate()),
        head_manifest_json=_write_json(
            tmp_path / "manifest.json", _phase2ax_head_manifest(package_allowed=True)
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["package_not_allowed_by_manifest"] is False


def test_phase2ax_full_postflight_accepts_command_slot_evidence_but_blocks_package(
    tmp_path: Path,
) -> None:
    report = audit_phase2ax_full_postflight(
        training_summary_json=_write_json(
            tmp_path / "summary.json", _phase2ax_full_training_summary()
        ),
        full_manifest_json=_write_json(
            tmp_path / "manifest.json", _phase2ax_full_manifest()
        ),
        train_data_health_json=_write_json(tmp_path / "train_data_health.json", {"passed": True}),
        val_data_health_json=_write_json(tmp_path / "val_data_health.json", {"passed": True}),
        train_pretrain_gate_json=_write_json(
            tmp_path / "train_pretrain.json", _phase2ax_full_pretrain_gate()
        ),
        val_pretrain_gate_json=_write_json(
            tmp_path / "val_pretrain.json", _phase2ax_full_pretrain_gate()
        ),
        smoke_postflight_json=_write_json(
            tmp_path / "smoke_postflight.json", _phase2ax_smoke_postflight()
        ),
    )

    assert report["passed"] is True
    assert report["ready_for_phase2ay_runtime_execution_eval"] is True
    assert report["ready_for_phase2ax_package"] is False
    assert report["ready_for_package_or_execution_claim"] is False
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert report["metrics"]["model_minus_source_overlap"] == 0.5
    assert report["metrics"]["model_minus_wrong_cache"] == 1.0
    assert report["metrics"]["patch_descriptor_failure"] is True
    assert report["patch_descriptor_status"] == "weak_not_repair_execution_evidence"


def test_phase2ax_full_postflight_rejects_source_overlap_tie(
    tmp_path: Path,
) -> None:
    report = audit_phase2ax_full_postflight(
        training_summary_json=_write_json(
            tmp_path / "summary.json",
            _phase2ax_full_training_summary(val_accuracy=0.85, source_accuracy=0.85),
        ),
        full_manifest_json=_write_json(
            tmp_path / "manifest.json", _phase2ax_full_manifest()
        ),
        train_data_health_json=_write_json(tmp_path / "train_data_health.json", {"passed": True}),
        val_data_health_json=_write_json(tmp_path / "val_data_health.json", {"passed": True}),
        train_pretrain_gate_json=_write_json(
            tmp_path / "train_pretrain.json", _phase2ax_full_pretrain_gate()
        ),
        val_pretrain_gate_json=_write_json(
            tmp_path / "val_pretrain.json", _phase2ax_full_pretrain_gate()
        ),
        smoke_postflight_json=_write_json(
            tmp_path / "smoke_postflight.json", _phase2ax_smoke_postflight()
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["model_beats_source_overlap"] is False
    assert report["ready_for_phase2ay_runtime_execution_eval"] is False


def test_phase2ax_full_postflight_rejects_missing_repo_disjoint_evidence(
    tmp_path: Path,
) -> None:
    report = audit_phase2ax_full_postflight(
        training_summary_json=_write_json(
            tmp_path / "summary.json", _phase2ax_full_training_summary()
        ),
        full_manifest_json=_write_json(
            tmp_path / "manifest.json", _phase2ax_full_manifest(repo_disjoint=False)
        ),
        train_data_health_json=_write_json(tmp_path / "train_data_health.json", {"passed": True}),
        val_data_health_json=_write_json(tmp_path / "val_data_health.json", {"passed": True}),
        train_pretrain_gate_json=_write_json(
            tmp_path / "train_pretrain.json", _phase2ax_full_pretrain_gate()
        ),
        val_pretrain_gate_json=_write_json(
            tmp_path / "val_pretrain.json", _phase2ax_full_pretrain_gate()
        ),
        smoke_postflight_json=_write_json(
            tmp_path / "smoke_postflight.json", _phase2ax_smoke_postflight()
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["full_manifest_repo_origin_disjoint"] is False
    assert report["metrics"]["repo_origin_overlap"] == ["https://github.com/example/shared.git"]


def test_phase2ax_full_postflight_rejects_package_permission_before_runtime_eval(
    tmp_path: Path,
) -> None:
    report = audit_phase2ax_full_postflight(
        training_summary_json=_write_json(
            tmp_path / "summary.json", _phase2ax_full_training_summary()
        ),
        full_manifest_json=_write_json(
            tmp_path / "manifest.json", _phase2ax_full_manifest(package_allowed=True)
        ),
        train_data_health_json=_write_json(tmp_path / "train_data_health.json", {"passed": True}),
        val_data_health_json=_write_json(tmp_path / "val_data_health.json", {"passed": True}),
        train_pretrain_gate_json=_write_json(
            tmp_path / "train_pretrain.json", _phase2ax_full_pretrain_gate()
        ),
        val_pretrain_gate_json=_write_json(
            tmp_path / "val_pretrain.json", _phase2ax_full_pretrain_gate()
        ),
        smoke_postflight_json=_write_json(
            tmp_path / "smoke_postflight.json", _phase2ax_smoke_postflight()
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["manifest_blocks_package"] is False
    assert "do_not_package_phase2ax_before_runtime_execution_eval" in report["blocked_actions"]


def test_phase2ax_full_head_builder_accepts_repo_disjoint_splits(
    tmp_path: Path,
) -> None:
    train_source = _write_jsonl(
        tmp_path / "train_source.jsonl",
        [
            _linked_source_row(0, "aaaabbbbcccc1111", repo="train-a"),
            _linked_source_row(1, "ddddeeeeffff2222", repo="train-a"),
        ],
    )
    val_source = _write_jsonl(
        tmp_path / "val_source.jsonl",
        [
            _linked_source_row(2, "1111222233334444", repo="val-a"),
            _linked_source_row(3, "5555666677778888", repo="val-a"),
        ],
    )
    smoke_postflight = _write_json(
        tmp_path / "smoke_postflight.json",
        {"passed": True, "ready_for_phase2ax_full_nonsealed_training": True},
    )

    manifest = build_phase2ax_full_head_dataset(
        train_source_jsonl=train_source,
        val_source_jsonl=val_source,
        output_dir=tmp_path / "full_head",
        report_dir=tmp_path / "reports",
        manifest_json=tmp_path / "manifest.json",
        smoke_postflight_json=smoke_postflight,
        max_train_pairs=1,
        max_val_pairs=1,
        min_train_pairs=1,
        min_val_pairs=1,
    )

    assert manifest["passed"] is True
    assert manifest["full_training_allowed"] is True
    assert manifest["package_allowed"] is False
    assert manifest["sealed_eval_allowed"] is False
    assert manifest["checks"]["repo_origin_disjoint"] is True
    assert manifest["command_slot_distribution"]["train"] == {"0": 1, "1": 1}
    assert manifest["command_slot_distribution"]["val"] == {"0": 1, "1": 1}


def test_phase2ax_full_head_builder_rejects_repo_overlap(
    tmp_path: Path,
) -> None:
    train_source = _write_jsonl(
        tmp_path / "train_source.jsonl",
        [
            _linked_source_row(0, "aaaabbbbcccc1111", repo="shared"),
            _linked_source_row(1, "ddddeeeeffff2222", repo="shared"),
        ],
    )
    val_source = _write_jsonl(
        tmp_path / "val_source.jsonl",
        [
            _linked_source_row(2, "1111222233334444", repo="shared"),
            _linked_source_row(3, "5555666677778888", repo="shared"),
        ],
    )
    smoke_postflight = _write_json(
        tmp_path / "smoke_postflight.json",
        {"passed": True, "ready_for_phase2ax_full_nonsealed_training": True},
    )

    manifest = build_phase2ax_full_head_dataset(
        train_source_jsonl=train_source,
        val_source_jsonl=val_source,
        output_dir=tmp_path / "full_head",
        report_dir=tmp_path / "reports",
        manifest_json=tmp_path / "manifest.json",
        smoke_postflight_json=smoke_postflight,
        max_train_pairs=1,
        max_val_pairs=1,
        min_train_pairs=1,
        min_val_pairs=1,
    )

    assert manifest["passed"] is False
    assert manifest["full_training_allowed"] is False
    assert manifest["checks"]["repo_origin_disjoint"] is False
    assert manifest["repo_origins"]["overlap"] == ["https://github.com/example/shared.git"]
