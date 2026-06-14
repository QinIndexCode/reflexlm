import copy
import json
from pathlib import Path

from reflexlm.cli.audit_phase2t_dynamic_repair_traces import (
    build_phase2t_baseline_feasibility_sanity_audit,
    build_phase2t_data_health,
    build_phase2t_full_postflight,
    build_phase2t_holdout_control_postflight,
    build_phase2t_cross_model_summary,
    build_phase2t_multiseed_summary,
    build_phase2t_package_gate,
    build_phase2t_postpackage_gate,
    build_phase2t_pretrain_gate,
    build_phase2t_sealed_zero_baseline_audit,
    build_phase2t_smoke_postflight,
)


TASK_FAMILIES = [
    "dependency_or_import_mismatch",
    "localized_unit_assertion",
    "stale_snapshot_update",
    "config_or_environment_marker",
    "multi_file_traceback_relation",
    "regression_after_partial_repair",
    "safety_blocked_command_temptation",
    "false_completion_trap",
]
FACTOR_LEVELS = {
    "candidate_count": [2, 3, 4],
    "evidence_density": ["low", "medium", "high"],
    "repair_depth": ["one_edit", "two_edits", "stale_state_refresh"],
    "failure_observability": [
        "direct_traceback",
        "indirect_changed_file_relation",
        "ambiguous_same_intent_command",
    ],
    "ambiguity_class": [
        "same_intent_command",
        "same_file_read",
        "stage_transition",
        "patch_location_ambiguity",
    ],
    "safety_pressure": ["none", "unsafe_command_lure", "rollback_required"],
}


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _artifacts(root: Path, split: str, repo_id: str, index: int) -> dict[str, str]:
    artifact_dir = root / "artifacts" / split / repo_id / f"row_{index:05d}"
    paths = {
        "patch_diff": artifact_dir / "patch.diff",
        "command_log": artifact_dir / "command_log.json",
        "test_output": artifact_dir / "test_output.json",
        "rollback_log": artifact_dir / "rollback_log.json",
        "sandbox_integrity_report": artifact_dir / "sandbox_integrity.json",
    }
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{key}\n", encoding="utf-8")
    return {key: path.relative_to(root).as_posix() for key, path in paths.items()}


def _row(root: Path, split: str, index: int) -> dict:
    repo_id = f"{split}_repo_{index:03d}"
    task_family = TASK_FAMILIES[index % len(TASK_FAMILIES)]
    difficulty = {
        "task_family": task_family,
        **{key: values[index % len(values)] for key, values in FACTOR_LEVELS.items()},
    }
    return {
        "phase": "Phase2T",
        "trace_id": f"{split}:{repo_id}:phase2t:{index}",
        "split": split,
        "source_kind": "public_repo",
        "trace_construction_mode": "phase2t_dynamic_public_repo_repair_loop_trace",
        "repo_id": repo_id,
        "repo_url_or_origin": f"https://github.com/example/{repo_id}.git",
        "commit_hash": f"{index + 1:040x}",
        "current_visible_text": "Visible repair-loop task without answer markers.",
        "runtime_visible_evidence": {"failure_observability": difficulty["failure_observability"]},
        "repair_candidates": [
            {"repair_action": "repair_alpha", "description": "bounded source patch"},
            {"repair_action": "repair_bravo", "description": "bounded source patch"},
        ],
        "expected_repair_action": "repair_alpha",
        "difficulty": difficulty,
        "repair_loop_episode": {
            "loop_schema": "phase2t_repair_loop_v1",
            "stages": [
                {"stage": "inspect_runtime_evidence"},
                {"stage": "propose_bounded_patch"},
                {"stage": "run_verification_tests"},
                {"stage": "rollback_failed_or_unsafe_patch"},
                {"stage": "emit_verified_stop"},
            ],
        },
        "architecture_targets": {
            key: {"required": True}
            for key in [
                "patch_proposal_head",
                "test_selection_head",
                "rollback_safety_head",
                "stop_condition_head",
                "verification_state_receptors",
                "incident_timeout_receptors",
                "progress_monitor_receptors",
                "bounded_edit_scope_policy",
            ]
        },
        "safety_controls": {
            "source_repo_read_only_observed": True,
            "bounded_edit_scope_observed": True,
            "command_allowlist_observed": True,
            "rollback_recorded": True,
            "sandbox_cleanup_recorded": True,
            "stop_requires_verification": True,
            "unauthorized_write_count": 0,
            "low_level_qwen_calls": 0,
        },
        "modern_baseline_contract": {
            "required": True,
            "measured_not_declared": True,
            "same_repair_loop_artifacts_required": True,
            "cost_and_command_budget_required": True,
        },
        "baselines": {
            "source_overlap": "repair_bravo",
            "native_head_only_no_cache": "repair_bravo",
            "continuation_only": "repair_bravo",
            "prompt_only": "repair_bravo",
            "react": "repair_bravo",
            "modern_coding_agent_loop": "repair_bravo",
        },
        "baseline_metadata": {
            name: {
                "measured": True,
                "uses_expected_repair_action": False,
                "uses_sealed_feedback": False,
            }
            for name in [
                "source_overlap",
                "native_head_only_no_cache",
                "continuation_only",
                "prompt_only",
                "react",
                "modern_coding_agent_loop",
            ]
        },
        "artifact_paths": _artifacts(root, split, repo_id, index),
        "trace_hash": f"{split}-{index}",
    }


def _rows(root: Path, split: str, count: int) -> list[dict]:
    return [_row(root, split, index) for index in range(count)]


def _write_valid_dataset(root: Path) -> tuple[Path, Path, Path, Path]:
    train_rows = _rows(root, "train", 24)
    val_rows = _rows(root, "val", 16)
    holdout_rows = _rows(root, "holdout", 16)
    manifest = _write(
        root / "manifest.json",
        {
            "collector_family": "phase2t_dynamic_public_repo_repair_loop_trace_collector",
            "claim_bearing_training_ready": False,
            "sealed_v3_used": False,
            "writes_to_source_repos": False,
            "execution_sandbox_used": True,
        },
    )
    return (
        manifest,
        _write_jsonl(root / "train.raw.jsonl", train_rows),
        _write_jsonl(root / "val.raw.jsonl", val_rows),
        _write_jsonl(root / "holdout.raw.jsonl", holdout_rows),
    )


def test_phase2t_data_health_and_pretrain_gate_accept_valid_repair_loop_dataset(
    tmp_path: Path,
) -> None:
    manifest, train, val, holdout = _write_valid_dataset(tmp_path)

    report = build_phase2t_data_health(
        manifest_json=manifest,
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        dataset_root=tmp_path,
    )
    gate = build_phase2t_pretrain_gate(data_health_json=_write(tmp_path / "health.json", report))

    assert report["passed"] is True
    assert report["allowed_next_action"] == "run_phase2t_claim_bearing_smoke_training_only"
    assert report["rollups"]["missing_val_task_families"] == []
    assert report["rollups"]["missing_val_factor_levels"] == {}
    assert gate["passed"] is True
    assert gate["allowed_next_action"] == "run_phase2t_claim_bearing_smoke_training_only"


def test_phase2t_data_health_rejects_training_ready_manifest_and_sealed_marker(
    tmp_path: Path,
) -> None:
    manifest, train, val, holdout = _write_valid_dataset(tmp_path)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["claim_bearing_training_ready"] = True
    manifest_payload["sealed_v3_used"] = True
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    val_rows = [json.loads(line) for line in val.read_text(encoding="utf-8").splitlines()]
    val_rows[0]["current_visible_text"] += " external_trace_v3_semantic_required"
    _write_jsonl(val, val_rows)

    report = build_phase2t_data_health(
        manifest_json=manifest,
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        dataset_root=tmp_path,
    )

    assert report["passed"] is False
    assert "do_not_skip_phase2t_data_health_with_collector_manifest" in report[
        "blocked_actions"
    ]
    assert "do_not_use_sealed_or_sealed_failure_feedback" in report["blocked_actions"]


def test_phase2t_data_health_rejects_missing_schema_artifact_and_pressure_coverage(
    tmp_path: Path,
) -> None:
    manifest, train, val, holdout = _write_valid_dataset(tmp_path)
    val_rows = [json.loads(line) for line in val.read_text(encoding="utf-8").splitlines()]
    val_rows = val_rows[:3]
    val_rows[0]["architecture_targets"]["patch_proposal_head"]["required"] = False
    missing_artifact = tmp_path / val_rows[0]["artifact_paths"]["patch_diff"]
    missing_artifact.unlink()
    _write_jsonl(val, val_rows)

    report = build_phase2t_data_health(
        manifest_json=manifest,
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        dataset_root=tmp_path,
        min_val_rows=3,
    )

    assert report["passed"] is False
    assert "do_not_train_phase2t_without_repair_loop_schema" in report["blocked_actions"]
    assert "do_not_train_phase2t_until_val_pressure_matrix_is_covered" in report[
        "blocked_actions"
    ]


def _postflight_gate_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    split_hashes = {
        "phase2t_train": "train_hash",
        "phase2t_val": "val_hash",
        "phase2t_holdout": "holdout_hash",
    }
    data_health = _write(
        tmp_path / "data_health.json",
        {"passed": True, "effective_split_hashes": split_hashes},
    )
    pretrain_gate = _write(
        tmp_path / "pretrain_gate.json",
        {"passed": True, "effective_split_hashes": split_hashes},
    )
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        {
            "dataset_family": "phase2t_dynamic_repair_head_dataset",
            "source_data_health_passed": True,
            "source_pretrain_gate_passed": True,
            "command_identity_margin_gate_passed": True,
            "effective_split_hashes": split_hashes,
            "splits": {
                "train": {"rows": 24},
                "val": {"rows": 16},
            },
        },
    )
    return data_health, pretrain_gate, head_manifest


def _training_summary(*, val_accuracy: float, source_overlap: float = 0.50) -> dict:
    return {
        "train_examples": 24,
        "val_examples": 16,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "use_pairwise_command_reranker": False,
        "command_candidate_encoder": "features_only",
        "effective_split_hashes": {
            "phase2c_head_train": "head_train_hash",
            "phase2c_head_val": "head_val_hash",
        },
        "source_overlap_command_slot_baseline": {
            "val": {"accuracy": source_overlap, "total": 16}
        },
        "pairwise_candidate_encoding": {
            "train": {"pairwise_scored_candidates": 0},
            "val": {"pairwise_scored_candidates": 0},
        },
        "history": [
            {
                "train_elapsed_seconds": 120.0,
                "val_metrics": {
                    "elapsed_seconds": 8.0,
                    "command_slot_accuracy": val_accuracy,
                    "command_slot_count": 16,
                },
            }
        ],
        "config": {
            "latent_fusion": "additive",
            "use_pairwise_command_reranker": False,
            "command_candidate_encoder": "features_only",
        },
    }


def test_phase2t_smoke_postflight_allows_full_only_after_val_and_delta_gate(
    tmp_path: Path,
) -> None:
    data_health, pretrain_gate, head_manifest = _postflight_gate_inputs(tmp_path)

    report = build_phase2t_smoke_postflight(
        training_summary_json=_write(
            tmp_path / "summary.pass.json",
            _training_summary(val_accuracy=0.875, source_overlap=0.50),
        ),
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
    )

    assert report["passed"] is True
    assert report["ready_for_full_train"] is True
    assert report["ready_for_package"] is False
    assert report["allowed_next_action"] == "run_phase2t_full_nonsealed_training_only"
    assert abs(report["metrics"]["model_minus_source_overlap_accuracy"] - 0.375) < 1e-9


def test_phase2t_smoke_postflight_blocks_full_when_val_gate_fails(
    tmp_path: Path,
) -> None:
    data_health, pretrain_gate, head_manifest = _postflight_gate_inputs(tmp_path)

    report = build_phase2t_smoke_postflight(
        training_summary_json=_write(
            tmp_path / "summary.fail.json",
            _training_summary(val_accuracy=0.8125, source_overlap=0.50),
        ),
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
    )

    assert report["passed"] is False
    assert report["allowed_next_action"] == (
        "freeze_phase2t_smoke_failure_and_analyze_nonsealed_design"
    )
    assert "do_not_claim_phase2t_val_gate" in report["blocked_actions"]
    assert "do_not_run_phase2t_full_train_until_smoke_postflight_passes" in report[
        "blocked_actions"
    ]


def test_phase2t_smoke_postflight_rejects_pairwise_or_hash_drift(
    tmp_path: Path,
) -> None:
    data_health, pretrain_gate, head_manifest = _postflight_gate_inputs(tmp_path)
    head_payload = json.loads(head_manifest.read_text(encoding="utf-8"))
    head_payload["effective_split_hashes"]["phase2t_val"] = "drifted_val_hash"
    head_manifest.write_text(json.dumps(head_payload), encoding="utf-8")
    summary = _training_summary(val_accuracy=0.90, source_overlap=0.50)
    summary["use_pairwise_command_reranker"] = True
    summary["pairwise_candidate_encoding"]["val"]["pairwise_scored_candidates"] = 4

    report = build_phase2t_smoke_postflight(
        training_summary_json=_write(tmp_path / "summary.pairwise.json", summary),
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
    )

    assert report["passed"] is False
    assert "do_not_mix_phase2t_smoke_with_pairwise_mechanism" in report["blocked_actions"]
    assert "do_not_train_or_package_with_phase2t_hash_mismatch" in report["blocked_actions"]


def test_phase2t_full_postflight_allows_holdout_controls_not_package(
    tmp_path: Path,
) -> None:
    data_health, pretrain_gate, head_manifest = _postflight_gate_inputs(tmp_path)
    head_payload = json.loads(head_manifest.read_text(encoding="utf-8"))
    head_payload["splits"]["train"]["rows"] = 96
    head_payload["splits"]["val"]["rows"] = 64
    head_manifest.write_text(json.dumps(head_payload), encoding="utf-8")
    summary = _training_summary(val_accuracy=1.0, source_overlap=0.40625)
    summary["train_examples"] = 96
    summary["val_examples"] = 64
    summary["source_overlap_command_slot_baseline"]["val"]["total"] = 64
    summary["history"][0]["val_metrics"]["command_slot_count"] = 64

    report = build_phase2t_full_postflight(
        training_summary_json=_write(tmp_path / "summary.full.json", summary),
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
        min_train_examples=96,
        min_val_examples=64,
    )

    assert report["passed"] is True
    assert report["ready_for_holdout_controls"] is True
    assert report["ready_for_package"] is False
    assert report["allowed_next_action"] == (
        "run_phase2t_holdout_control_diagnostics_before_package"
    )
    assert abs(report["metrics"]["model_minus_source_overlap_accuracy"] - 0.59375) < 1e-9


def test_phase2t_full_postflight_rejects_small_or_delta_failed_runs(
    tmp_path: Path,
) -> None:
    data_health, pretrain_gate, head_manifest = _postflight_gate_inputs(tmp_path)
    summary = _training_summary(val_accuracy=0.90, source_overlap=0.80)
    summary["train_examples"] = 24
    summary["val_examples"] = 16

    report = build_phase2t_full_postflight(
        training_summary_json=_write(tmp_path / "summary.small.json", summary),
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
        min_train_examples=96,
        min_val_examples=64,
    )

    assert report["passed"] is False
    assert "do_not_claim_phase2t_full_mechanism_delta_from_source_overlap" in report[
        "blocked_actions"
    ]
    assert "do_not_claim_phase2t_full_before_minimum_nonsealed_split_size" in report[
        "blocked_actions"
    ]


def _diagnostics(*, effective: float, source_overlap: float, slot_head: float = 0.20) -> dict:
    total = 96
    return {
        "sealed_data_used_for_training_or_tuning": False,
        "use_pairwise_command_reranker": False,
        "command_candidate_encoder": "features_only",
        "sources": {
            "effective": {"accuracy": effective, "total": total},
            "source_overlap_baseline": {"accuracy": source_overlap, "total": total},
            "slot_head": {"accuracy": slot_head, "total": total},
        },
    }


def test_phase2t_holdout_control_postflight_requires_zero_nsi_delta(
    tmp_path: Path,
) -> None:
    full = _write(tmp_path / "full_postflight.json", {"passed": True})
    holdout = _write(
        tmp_path / "holdout.json",
        _diagnostics(effective=1.0, source_overlap=0.40625, slot_head=0.15625),
    )
    zero_nsi = _write(
        tmp_path / "zero_nsi.json",
        _diagnostics(effective=0.3125, source_overlap=0.40625, slot_head=0.1666667),
    )
    zero_manifest = _write(
        tmp_path / "zero_manifest.json",
        {
            "nsi_reference_erased": True,
            "sealed_v3_used_for_training_or_tuning": False,
        },
    )

    report = build_phase2t_holdout_control_postflight(
        full_postflight_json=full,
        holdout_diagnostics_json=holdout,
        zero_nsi_diagnostics_json=zero_nsi,
        zero_nsi_manifest_json=zero_manifest,
    )

    assert report["passed"] is True
    assert report["ready_for_package"] is False
    assert report["allowed_next_action"] == (
        "run_phase2t_package_gate_or_additional_controls_before_package"
    )
    assert report["metrics"]["full_minus_zero_nsi_holdout_accuracy"] == 0.6875


def test_phase2t_holdout_control_postflight_blocks_when_zero_nsi_matches_full(
    tmp_path: Path,
) -> None:
    full = _write(tmp_path / "full_postflight.json", {"passed": True})
    holdout = _write(
        tmp_path / "holdout.json",
        _diagnostics(effective=0.90, source_overlap=0.50, slot_head=0.40),
    )
    zero_nsi = _write(
        tmp_path / "zero_nsi.json",
        _diagnostics(effective=0.88, source_overlap=0.50, slot_head=0.40),
    )

    report = build_phase2t_holdout_control_postflight(
        full_postflight_json=full,
        holdout_diagnostics_json=holdout,
        zero_nsi_diagnostics_json=zero_nsi,
        min_full_minus_zero_nsi=0.15,
    )

    assert report["passed"] is False
    assert "do_not_claim_phase2t_nsi_latent_necessity" in report["blocked_actions"]


def _holdout_postflight(
    path: Path,
    *,
    passed: bool = True,
    holdout: float = 1.0,
    source_delta: float = 0.59,
    zero_nsi_delta: float = 0.68,
) -> Path:
    return _write(
        path,
        {
            "passed": passed,
            "metrics": {
                "holdout_effective_accuracy": holdout,
                "model_minus_source_overlap_holdout_accuracy": source_delta,
                "full_minus_zero_nsi_holdout_accuracy": zero_nsi_delta,
                "full_minus_raw_slot_head_holdout_accuracy": 0.84,
            },
        },
    )


def test_phase2t_multiseed_summary_requires_three_passing_seeds(tmp_path: Path) -> None:
    reports = [
        _holdout_postflight(tmp_path / "seed17.json"),
        _holdout_postflight(tmp_path / "seed23.json", zero_nsi_delta=0.59),
        _holdout_postflight(tmp_path / "seed29.json", zero_nsi_delta=0.66),
    ]

    report = build_phase2t_multiseed_summary(holdout_postflight_jsons=reports)

    assert report["passed"] is True
    assert report["ready_for_multimodel_reproduction"] is True
    assert report["ready_for_package"] is False
    assert report["metrics"]["seed_count"] == 3
    assert report["metrics"]["min_full_minus_zero_nsi_holdout_accuracy"] == 0.59


def test_phase2t_multiseed_summary_blocks_too_few_or_delta_failed_seeds(
    tmp_path: Path,
) -> None:
    reports = [
        _holdout_postflight(tmp_path / "seed17.json"),
        _holdout_postflight(tmp_path / "seed23.json", zero_nsi_delta=0.01),
    ]

    report = build_phase2t_multiseed_summary(holdout_postflight_jsons=reports)

    assert report["passed"] is False
    assert "do_not_claim_multiseed_with_too_few_seeds" in report["blocked_actions"]
    assert "do_not_claim_phase2t_multiseed_nsi_latent_delta" in report["blocked_actions"]


def _multiseed_summary(
    path: Path,
    *,
    passed: bool = True,
    seed_count: int = 3,
    min_holdout: float = 1.0,
    min_source_delta: float = 0.59,
    min_zero_nsi_delta: float = 0.62,
) -> Path:
    return _write(
        path,
        {
            "passed": passed,
            "metrics": {
                "seed_count": seed_count,
                "min_holdout_effective_accuracy": min_holdout,
                "min_model_minus_source_overlap_holdout_accuracy": min_source_delta,
                "min_full_minus_zero_nsi_holdout_accuracy": min_zero_nsi_delta,
            },
        },
    )


def test_phase2t_cross_model_summary_requires_two_models_with_three_seeds(
    tmp_path: Path,
) -> None:
    summaries = [
        _multiseed_summary(tmp_path / "qwen3b.json", min_zero_nsi_delta=0.625),
        _multiseed_summary(tmp_path / "qwen7b.json", min_zero_nsi_delta=0.59375),
    ]

    report = build_phase2t_cross_model_summary(multiseed_summary_jsons=summaries)

    assert report["passed"] is True
    assert report["ready_for_package"] is False
    assert report["ready_for_package_gate_design"] is True
    assert report["metrics"]["model_count"] == 2
    assert report["metrics"]["min_zero_nsi_delta_across_models"] == 0.59375


def test_phase2t_cross_model_summary_blocks_missing_model_or_seed_count(
    tmp_path: Path,
) -> None:
    summaries = [_multiseed_summary(tmp_path / "qwen3b.json", seed_count=2)]

    report = build_phase2t_cross_model_summary(multiseed_summary_jsons=summaries)

    assert report["passed"] is False
    assert "do_not_claim_cross_model_with_too_few_models" in report["blocked_actions"]
    assert "do_not_claim_cross_model_without_per_model_seed_reproduction" in report[
        "blocked_actions"
    ]


def _package_gate_inputs(tmp_path: Path) -> dict[str, Path]:
    split_hashes = {
        "phase2t_train": "train_hash",
        "phase2t_val": "val_hash",
        "phase2t_holdout": "holdout_hash",
    }
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    cross_model = _write(
        tmp_path / "cross_model.json",
        {
            "passed": True,
            "ready_for_sealed_eval": False,
            "metrics": {
                "model_count": 2,
                "min_seed_count_per_model": 3,
                "min_holdout_effective_accuracy_across_models": 1.0,
                "min_source_overlap_delta_across_models": 0.59375,
                "min_zero_nsi_delta_across_models": 0.59375,
            },
        },
    )
    summary = _write(
        tmp_path / "summary.json",
        {
            "no_json_motor_target": True,
            "low_level_qwen_calls_target": 0,
            "use_pairwise_command_reranker": False,
            "command_candidate_encoder": "features_only",
            "effective_split_hashes": {
                "phase2c_head_train": "head_train_hash",
                "phase2c_head_val": "head_val_hash",
            },
            "config": {
                "latent_fusion": "additive",
                "command_candidate_encoder": "features_only",
            },
            "head_config": {
                "latent_fusion": "additive",
                "use_pairwise_command_reranker": False,
                "command_candidate_encoder": "features_only",
            },
        },
    )
    full_postflight = _write(
        tmp_path / "full_postflight.json",
        {
            "passed": True,
            "ready_for_sealed_eval": False,
            "metrics": {"val_command_slot_accuracy": 1.0},
        },
    )
    holdout_postflight = _write(
        tmp_path / "holdout_postflight.json",
        {
            "passed": True,
            "ready_for_sealed_eval": False,
            "metrics": {
                "holdout_effective_accuracy": 1.0,
                "model_minus_source_overlap_holdout_accuracy": 0.59375,
                "full_minus_zero_nsi_holdout_accuracy": 0.6875,
            },
        },
    )
    data_health = _write(
        tmp_path / "data_health.json",
        {"passed": True, "effective_split_hashes": split_hashes},
    )
    pretrain = _write(
        tmp_path / "pretrain.json",
        {"passed": True, "effective_split_hashes": split_hashes},
    )
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        {
            "source_data_health_passed": True,
            "source_pretrain_gate_passed": True,
            "command_identity_margin_gate_passed": True,
            "effective_split_hashes": split_hashes,
        },
    )
    return {
        "cross_model": cross_model,
        "summary": summary,
        "full_postflight": full_postflight,
        "holdout_postflight": holdout_postflight,
        "data_health": data_health,
        "pretrain": pretrain,
        "head_manifest": head_manifest,
        "adapter_dir": adapter_dir,
    }


def test_phase2t_package_gate_allows_package_but_not_sealed_eval(tmp_path: Path) -> None:
    inputs = _package_gate_inputs(tmp_path)

    report = build_phase2t_package_gate(
        cross_model_summary_json=inputs["cross_model"],
        canonical_training_summary_json=inputs["summary"],
        canonical_full_postflight_json=inputs["full_postflight"],
        canonical_holdout_control_postflight_json=inputs["holdout_postflight"],
        data_health_json=inputs["data_health"],
        pretrain_gate_json=inputs["pretrain"],
        head_manifest_json=inputs["head_manifest"],
        adapter_dir=inputs["adapter_dir"],
    )

    assert report["passed"] is True
    assert report["ready_for_package"] is True
    assert report["ready_for_sealed_eval"] is False
    assert report["allowed_next_action"] == "run_phase2t_package_only_then_sealed_eval_gate"
    assert "phase2t_qwen2_5_3b_7b_three_seed_reproduction_supported" in report[
        "supported_claims"
    ]
    assert "epoch_making_architecture_claim_not_established" in report["unsupported_claims"]


def test_phase2t_package_gate_blocks_hash_drift_or_missing_cross_model(
    tmp_path: Path,
) -> None:
    inputs = _package_gate_inputs(tmp_path)
    cross_payload = json.loads(inputs["cross_model"].read_text(encoding="utf-8"))
    cross_payload["passed"] = False
    inputs["cross_model"].write_text(json.dumps(cross_payload), encoding="utf-8")
    head_payload = json.loads(inputs["head_manifest"].read_text(encoding="utf-8"))
    head_payload["effective_split_hashes"]["phase2t_val"] = "drifted"
    inputs["head_manifest"].write_text(json.dumps(head_payload), encoding="utf-8")

    report = build_phase2t_package_gate(
        cross_model_summary_json=inputs["cross_model"],
        canonical_training_summary_json=inputs["summary"],
        canonical_full_postflight_json=inputs["full_postflight"],
        canonical_holdout_control_postflight_json=inputs["holdout_postflight"],
        data_health_json=inputs["data_health"],
        pretrain_gate_json=inputs["pretrain"],
        head_manifest_json=inputs["head_manifest"],
        adapter_dir=inputs["adapter_dir"],
    )

    assert report["passed"] is False
    assert report["ready_for_package"] is False
    assert "do_not_package_without_phase2t_cross_model_summary" in report["blocked_actions"]
    assert "do_not_package_phase2t_with_split_hash_mismatch" in report["blocked_actions"]


def _write_package_manifest(
    package_dir: Path,
    *,
    label: str,
    zero_nsi: bool = False,
    continuation: bool = True,
    native_heads: bool = True,
    model: str = "model",
    adapter: str = "adapter",
    checkpoint: str = "checkpoint",
) -> Path:
    package_dir.mkdir(parents=True, exist_ok=True)
    return _write(
        package_dir / "native_nervous_package.json",
        {
            "package_family": "phase2d_native_nervous_package",
            "policy_label": label,
            "base_model_name": model,
            "native_head_path": adapter,
            "low_level_checkpoint_path": checkpoint,
            "json_text_target": False,
            "zero_nsi_latent": zero_nsi,
            "native_head_calls_enabled": native_heads,
            "continuation_cache_enabled": continuation,
            "motor_output": "explicit_heads_runtime_serialization",
        },
    )


def test_phase2t_postpackage_gate_allows_sealed_eval_only_after_controls(
    tmp_path: Path,
) -> None:
    package_gate = _write(
        tmp_path / "package_gate.json",
        {"passed": True, "ready_for_package": True, "ready_for_sealed_eval": False},
    )
    label = "phase2t_package"
    full = tmp_path / label
    no_nsi = tmp_path / f"{label}_no_nsi_latent"
    native = tmp_path / f"{label}_native_head_only"
    continuation = tmp_path / f"{label}_continuation_only"
    _write_package_manifest(full, label=label)
    _write_package_manifest(no_nsi, label=f"{label}_no_nsi_latent", zero_nsi=True)
    _write_package_manifest(
        native,
        label=f"{label}_native_head_only",
        continuation=False,
    )
    _write_package_manifest(
        continuation,
        label=f"{label}_continuation_only",
        native_heads=False,
    )

    report = build_phase2t_postpackage_gate(
        package_gate_json=package_gate,
        full_package_path=full,
        no_nsi_package_path=no_nsi,
        native_head_only_package_path=native,
        continuation_only_package_path=continuation,
    )

    assert report["passed"] is True
    assert report["ready_for_sealed_eval"] is True
    assert report["ready_for_claim_upgrade"] is False
    assert report["allowed_next_action"] == "run_phase2t_sealed_eval_only_no_training_feedback"


def test_phase2t_postpackage_gate_rejects_mismatched_controls(tmp_path: Path) -> None:
    package_gate = _write(
        tmp_path / "package_gate.json",
        {"passed": True, "ready_for_package": True, "ready_for_sealed_eval": False},
    )
    label = "phase2t_package"
    full = tmp_path / label
    no_nsi = tmp_path / f"{label}_no_nsi_latent"
    native = tmp_path / f"{label}_native_head_only"
    continuation = tmp_path / f"{label}_continuation_only"
    _write_package_manifest(full, label=label)
    _write_package_manifest(no_nsi, label="wrong_label", zero_nsi=True)
    _write_package_manifest(
        native,
        label=f"{label}_native_head_only",
        continuation=False,
        adapter="drifted_adapter",
    )
    _write_package_manifest(
        continuation,
        label=f"{label}_continuation_only",
        native_heads=False,
    )

    report = build_phase2t_postpackage_gate(
        package_gate_json=package_gate,
        full_package_path=full,
        no_nsi_package_path=no_nsi,
        native_head_only_package_path=native,
        continuation_only_package_path=continuation,
    )

    assert report["passed"] is False
    assert "do_not_run_sealed_eval_with_mismatched_control_packages" in report[
        "blocked_actions"
    ]
    assert "do_not_run_sealed_eval_with_drifted_package_artifacts" in report[
        "blocked_actions"
    ]


def _metric(value: float | None) -> dict | None:
    if value is None:
        return None
    return {"mean": value, "count": 1}


def _write_eval_run(
    tmp_path: Path,
    name: str,
    *,
    completion: float,
    policy: dict,
    action_type: str = "READ_STDOUT",
    oracle_type: str = "READ_STDERR",
    correct: bool = False,
    parse_failures: int = 0,
    hallucinated: bool = False,
) -> Path:
    run_path = tmp_path / "runs" / name
    run_path.mkdir(parents=True, exist_ok=True)
    trace_row = {
        "step_index": 0,
        "action": {"type": action_type},
        "oracle_action": {"type": oracle_type},
        "correct": correct,
        "hallucinated": hallucinated,
        "qwen_called": True,
        "cache_hit": bool(policy.get("continuation_cache_enabled")),
        "cache_reset_reason": "semantic_command_ambiguity"
        if policy.get("continuation_cache_enabled")
        else None,
    }
    _write_jsonl(run_path / "trace_rows.jsonl", [trace_row])
    _write_jsonl(
        run_path / "episode_results.jsonl",
        [{"parse_failures": parse_failures, "task_completion_rate": completion}],
    )
    return _write(
        tmp_path / f"{name}.json",
        {
            "policy": policy,
            "metrics": {
                "aggregate": {
                    "task_completion_rate": _metric(completion),
                    "oracle_step_accuracy": _metric(1.0 if correct else 0.0),
                    "command_decision_accuracy": _metric(1.0 if correct else 0.0)
                    if action_type == "RUN_COMMAND"
                    else None,
                    "read_file_decision_accuracy": _metric(1.0 if correct else 0.0)
                    if action_type.startswith("READ_")
                    else None,
                    "state_hallucination_rate": _metric(1.0 if hallucinated else 0.0),
                    "model_calls": _metric(1.0 if policy.get("native_head_calls_enabled", True) else 0.0),
                }
            },
            "run_path": str(run_path),
        },
    )


def test_phase2t_sealed_zero_baseline_audit_classifies_all_zero_controls(
    tmp_path: Path,
) -> None:
    gate = _write(tmp_path / "external_gate.json", {"passed": True})
    full = _write_eval_run(
        tmp_path,
        "full",
        completion=0.95,
        policy={
            "policy_family": "phase2d_native_nervous_package",
            "zero_nsi_latent": False,
            "native_head_calls_enabled": True,
            "continuation_cache_enabled": True,
        },
        action_type="READ_STDERR",
        oracle_type="READ_STDERR",
        correct=True,
    )
    no_nsi = _write_eval_run(
        tmp_path,
        "no_nsi",
        completion=0.0,
        policy={
            "policy_family": "phase2d_native_nervous_package",
            "zero_nsi_latent": True,
            "native_head_calls_enabled": True,
            "continuation_cache_enabled": True,
        },
    )
    native = _write_eval_run(
        tmp_path,
        "native",
        completion=0.0,
        policy={
            "policy_family": "phase2d_native_nervous_package",
            "zero_nsi_latent": False,
            "native_head_calls_enabled": True,
            "continuation_cache_enabled": False,
        },
    )
    continuation = _write_eval_run(
        tmp_path,
        "continuation",
        completion=0.0,
        policy={
            "policy_family": "phase2d_native_nervous_package",
            "zero_nsi_latent": False,
            "native_head_calls_enabled": False,
            "continuation_cache_enabled": True,
        },
    )
    prompt = _write_eval_run(
        tmp_path,
        "prompt",
        completion=0.0,
        policy={"policy_family": "huggingface_json"},
    )
    react = _write_eval_run(
        tmp_path,
        "react",
        completion=0.0,
        policy={"policy_family": "huggingface_json"},
    )

    report = build_phase2t_sealed_zero_baseline_audit(
        external_gate_json=gate,
        full_eval_json=full,
        prompt_eval_json=prompt,
        react_eval_json=react,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_sealed_claim"] is True
    assert report["ready_for_strong_architecture_claim"] is False
    assert report["all_controls_zero"] is True
    assert "add_graded_sanity_subset_with_nonzero_baseline_feasibility" in report[
        "blocked_actions"
    ]
    assert report["eval_summaries"]["prompt_only"]["zero_classification"][
        "classification"
    ] == "valid_zero_failure"
    assert report["eval_summaries"]["no_nsi"]["zero_classification"][
        "classification"
    ] == "expected_zero_due_to_missing_capability"


def test_phase2t_sealed_zero_baseline_audit_rejects_parse_failure_zero(
    tmp_path: Path,
) -> None:
    gate = _write(tmp_path / "external_gate.json", {"passed": True})
    full = _write_eval_run(
        tmp_path,
        "full",
        completion=0.95,
        policy={"policy_family": "phase2d_native_nervous_package"},
        action_type="READ_STDERR",
        oracle_type="READ_STDERR",
        correct=True,
    )
    prompt = _write_eval_run(
        tmp_path,
        "prompt",
        completion=0.0,
        policy={"policy_family": "huggingface_json"},
        parse_failures=1,
    )
    clean_zero = _write_eval_run(
        tmp_path,
        "clean_zero",
        completion=0.0,
        policy={"policy_family": "huggingface_json"},
    )

    report = build_phase2t_sealed_zero_baseline_audit(
        external_gate_json=gate,
        full_eval_json=full,
        prompt_eval_json=prompt,
        react_eval_json=clean_zero,
        no_nsi_eval_json=clean_zero,
        native_head_only_eval_json=clean_zero,
        continuation_only_eval_json=clean_zero,
    )

    assert report["passed"] is False
    assert report["suspicious_zero_controls"] == ["prompt_only"]
    assert "do_not_claim_phase2t_sealed_delta_until_zero_roots_are_explained" in report[
        "blocked_actions"
    ]


def _baseline_feasibility_inputs(
    tmp_path: Path,
    *,
    source_overlap: float = 0.40625,
    zero_nsi: float = 0.3125,
    slot_head: float = 0.15625,
    data_health_passed: bool = True,
) -> dict[str, Path]:
    sealed_zero = _write(
        tmp_path / "sealed_zero.json",
        {
            "passed": True,
            "all_controls_zero": True,
            "ready_for_bounded_sealed_claim": True,
            "ready_for_strong_architecture_claim": False,
        },
    )
    holdout_postflight = _write(
        tmp_path / "holdout_postflight.json",
        {
            "passed": True,
            "metrics": {
                "holdout_effective_accuracy": 1.0,
                "holdout_effective_total": 96,
            },
        },
    )
    holdout_diagnostics = _write(
        tmp_path / "holdout_diagnostics.json",
        _diagnostics(
            effective=1.0,
            source_overlap=source_overlap,
            slot_head=slot_head,
        ),
    )
    zero_nsi_diagnostics = _write(
        tmp_path / "zero_nsi_diagnostics.json",
        _diagnostics(
            effective=zero_nsi,
            source_overlap=source_overlap,
            slot_head=slot_head,
        ),
    )
    coverage_checks = {
        "phase2t_no_sealed_reference_anywhere": True,
        "phase2t_train_task_family_coverage": True,
        "phase2t_train_factor_level_coverage": True,
        "phase2t_val_task_family_coverage": True,
        "phase2t_val_factor_level_coverage": True,
        "phase2t_holdout_task_family_coverage": True,
        "phase2t_holdout_factor_level_coverage": True,
    }
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": data_health_passed,
            "checks": coverage_checks,
        },
    )
    return {
        "sealed_zero": sealed_zero,
        "holdout_postflight": holdout_postflight,
        "holdout_diagnostics": holdout_diagnostics,
        "zero_nsi_diagnostics": zero_nsi_diagnostics,
        "data_health": data_health,
    }


def test_phase2t_baseline_feasibility_sanity_accepts_nonsealed_nonzero_controls(
    tmp_path: Path,
) -> None:
    inputs = _baseline_feasibility_inputs(tmp_path)

    report = build_phase2t_baseline_feasibility_sanity_audit(
        sealed_zero_audit_json=inputs["sealed_zero"],
        holdout_control_postflight_json=inputs["holdout_postflight"],
        holdout_diagnostics_json=inputs["holdout_diagnostics"],
        zero_nsi_diagnostics_json=inputs["zero_nsi_diagnostics"],
        data_health_json=inputs["data_health"],
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_sealed_claim_with_zero_caveat"] is True
    assert report["ready_for_strong_architecture_claim"] is False
    assert report["nonzero_baselines"] == ["source_overlap", "zero_nsi", "slot_head"]
    assert report["metrics"]["full_minus_source_overlap_holdout_accuracy"] == 0.59375
    assert report["metrics"]["full_minus_zero_nsi_holdout_accuracy"] == 0.6875
    assert report["metrics"]["full_minus_slot_head_holdout_accuracy"] == 0.84375


def test_phase2t_baseline_feasibility_sanity_rejects_all_zero_or_easy_baselines(
    tmp_path: Path,
) -> None:
    zero_inputs = _baseline_feasibility_inputs(
        tmp_path / "zero",
        source_overlap=0.0,
        zero_nsi=0.0,
        slot_head=0.0,
    )

    zero_report = build_phase2t_baseline_feasibility_sanity_audit(
        sealed_zero_audit_json=zero_inputs["sealed_zero"],
        holdout_control_postflight_json=zero_inputs["holdout_postflight"],
        holdout_diagnostics_json=zero_inputs["holdout_diagnostics"],
        zero_nsi_diagnostics_json=zero_inputs["zero_nsi_diagnostics"],
        data_health_json=zero_inputs["data_health"],
    )

    assert zero_report["passed"] is False
    assert "add_nonsealed_sanity_tasks_where_controls_can_score_nonzero" in zero_report[
        "blocked_actions"
    ]

    easy_inputs = _baseline_feasibility_inputs(tmp_path / "easy", source_overlap=0.95)

    easy_report = build_phase2t_baseline_feasibility_sanity_audit(
        sealed_zero_audit_json=easy_inputs["sealed_zero"],
        holdout_control_postflight_json=easy_inputs["holdout_postflight"],
        holdout_diagnostics_json=easy_inputs["holdout_diagnostics"],
        zero_nsi_diagnostics_json=easy_inputs["zero_nsi_diagnostics"],
        data_health_json=easy_inputs["data_health"],
    )

    assert easy_report["passed"] is False
    assert "redesign_nonsealed_sanity_subset_source_overlap_is_too_easy" in easy_report[
        "blocked_actions"
    ]
