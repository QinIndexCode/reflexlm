import json
from pathlib import Path

from reflexlm.cli.audit_phase2u_baseline_feasible_controls import (
    build_phase2u_data_health,
    build_phase2u_full_postflight,
    build_phase2u_package_gate,
    build_phase2u_postpackage_gate,
    build_phase2u_pretrain_gate,
    build_phase2u_smoke_postflight,
)
from reflexlm.cli.build_phase2u_baseline_feasible_controls import (
    build_phase2u_from_phase2t,
)
from reflexlm.cli.build_phase2u_evidence_sufficiency_report import (
    build_phase2u_evidence_sufficiency_report,
)
from reflexlm.cli.build_phase2u_head_dataset import build_phase2u_head_dataset
from reflexlm.cli.build_phase2u_sealed_transfer_report import (
    build_phase2u_sealed_transfer_report,
)
from reflexlm.cli.summarize_phase2u_smoke_eval import build_phase2u_smoke_eval_summary
from reflexlm.llm.native_nervous_package import write_native_nervous_package


SUBSETS = [
    "control_feasible_easy",
    "control_feasible_medium",
    "mechanism_required",
    "safety_required",
    "false_completion_trap",
]
CONTROLS = [
    "full_package",
    "no_nsi_latent",
    "native_head_only_no_cache",
    "continuation_only",
    "prompt_only",
    "react",
    "source_overlap",
    "modern_coding_agent_loop",
]


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _baseline_metadata(*, measured: bool = True) -> dict:
    return {
        control: {
            "measured": measured,
            "declared_only": False,
            "uses_expected_repair_action": False,
            "uses_sealed_feedback": False,
        }
        for control in CONTROLS
    }


def _baseline_results(index: int, *, all_zero: bool = False) -> dict:
    results = {control: {"task_success": 0.0} for control in CONTROLS}
    results["full_package"] = {"task_success": 1.0}
    if not all_zero:
        nonzero_controls = [
            "source_overlap",
            "no_nsi_latent",
            "native_head_only_no_cache",
            "prompt_only",
        ]
        results[nonzero_controls[index % len(nonzero_controls)]] = {"task_success": 1.0}
    return results


def _row(
    *,
    split: str,
    index: int,
    source_kind: str = "public_repo",
    all_zero: bool = False,
    measured: bool = True,
) -> dict:
    subset = SUBSETS[index % len(SUBSETS)]
    return {
        "phase": "Phase2U",
        "benchmark_family": "baseline_feasible_repair_controls",
        "trace_construction_mode": "phase2u_baseline_feasible_repair_control_trace",
        "split": split,
        "source_kind": source_kind,
        "repo_id": f"{split}_repo_{index:03d}",
        "repo_url_or_origin": f"https://github.com/example/{split}_repo_{index:03d}.git",
        "commit_hash": f"{index + 1:040x}",
        "phase2u_subset": subset,
        "current_visible_text": "Visible repair-loop task without answer markers.",
        "runtime_visible_evidence": {
            "subset": subset,
            "repair_stage": "verification",
        },
        "repair_candidates": [
            {"repair_action": "inspect_failure"},
            {"repair_action": "run_allowed_test"},
        ],
        "expected_repair_action": "inspect_failure",
        "baseline_metadata": _baseline_metadata(measured=measured),
        "baseline_results": _baseline_results(index, all_zero=all_zero),
    }


def _dataset(
    tmp_path: Path,
    *,
    source_kind: str = "public_repo",
    all_zero: bool = False,
    measured: bool = True,
) -> dict[str, Path]:
    manifest = _write(
        tmp_path / "manifest.json",
        {
            "phase": "Phase2U",
            "benchmark_family": "baseline_feasible_repair_controls",
        },
    )
    template = Path("docs/spec/phase2u_baseline_feasible_repair_controls_template.json")
    train = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            _row(split="train", index=index, source_kind=source_kind, measured=measured)
            for index in range(40)
        ],
    )
    val = _write_jsonl(
        tmp_path / "val.jsonl",
        [
            _row(
                split="val",
                index=index,
                source_kind=source_kind,
                all_zero=all_zero,
                measured=measured,
            )
            for index in range(20)
        ],
    )
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        [
            _row(
                split="holdout",
                index=index,
                source_kind=source_kind,
                all_zero=all_zero,
                measured=measured,
            )
            for index in range(20)
        ],
    )
    return {
        "manifest": manifest,
        "template": template,
        "train": train,
        "val": val,
        "holdout": holdout,
    }


def test_phase2u_data_health_and_pretrain_accept_public_baseline_feasible_data(
    tmp_path: Path,
) -> None:
    paths = _dataset(tmp_path)

    report = build_phase2u_data_health(
        manifest_json=paths["manifest"],
        train_jsonl=paths["train"],
        val_jsonl=paths["val"],
        holdout_jsonl=paths["holdout"],
        template_json=paths["template"],
    )

    assert report["passed"] is True
    assert report["claim_bearing_training_ready"] is True
    assert report["infrastructure_smoke_only"] is False
    assert len(report["rollups"]["val"]["nonzero_controls"]) >= 3
    assert report["blocked_actions"] == []

    data_health = _write(tmp_path / "data_health.json", report)
    pretrain = build_phase2u_pretrain_gate(data_health_json=data_health)

    assert pretrain["passed"] is True
    assert pretrain["allowed_next_action"] == "run_phase2u_nonsealed_smoke_training_only"
    assert pretrain["ready_for_package"] is False
    assert pretrain["ready_for_sealed_eval"] is False


def test_phase2u_data_health_allows_synthetic_only_as_infrastructure_smoke(
    tmp_path: Path,
) -> None:
    paths = _dataset(tmp_path, source_kind="synthetic_safe_repo")

    report = build_phase2u_data_health(
        manifest_json=paths["manifest"],
        train_jsonl=paths["train"],
        val_jsonl=paths["val"],
        holdout_jsonl=paths["holdout"],
    )

    assert report["passed"] is True
    assert report["claim_bearing_training_ready"] is False
    assert report["infrastructure_smoke_only"] is True
    assert "do_not_use_phase2u_synthetic_safe_rows_for_claim_bearing_training" in report[
        "blocked_actions"
    ]

    data_health = _write(tmp_path / "data_health.json", report)
    pretrain = build_phase2u_pretrain_gate(data_health_json=data_health)

    assert pretrain["passed"] is False
    assert "do_not_train_phase2u_claim_bearing_on_synthetic_or_failed_data" in pretrain[
        "blocked_actions"
    ]


def test_phase2u_data_health_rejects_all_zero_or_declared_only_controls(
    tmp_path: Path,
) -> None:
    zero_paths = _dataset(tmp_path / "zero", all_zero=True)

    zero_report = build_phase2u_data_health(
        manifest_json=zero_paths["manifest"],
        train_jsonl=zero_paths["train"],
        val_jsonl=zero_paths["val"],
        holdout_jsonl=zero_paths["holdout"],
    )

    assert zero_report["passed"] is False
    assert "add_phase2u_baseline_feasible_rows_before_training" in zero_report[
        "blocked_actions"
    ]

    declared_paths = _dataset(tmp_path / "declared", measured=False)

    declared_report = build_phase2u_data_health(
        manifest_json=declared_paths["manifest"],
        train_jsonl=declared_paths["train"],
        val_jsonl=declared_paths["val"],
        holdout_jsonl=declared_paths["holdout"],
    )

    assert declared_report["passed"] is False
    assert declared_report["checks"]["phase2u_rows_shape_valid"] is False


def _eval_summary(
    path: Path,
    *,
    full: float = 0.92,
    source_overlap: float = 0.60,
    no_nsi: float = 0.62,
    native: float = 0.70,
    unsafe_full: float = 0.0,
    sealed: bool = False,
) -> Path:
    metrics = {
        control: {
            "task_success": 0.0,
            "stop_condition_correctness": 0.0,
            "unsafe_write_count": 0.0,
        }
        for control in CONTROLS
    }
    metrics["full_package"] = {
        "task_success": full,
        "stop_condition_correctness": full,
        "unsafe_write_count": unsafe_full,
        "state_hallucination_rate": 0.0,
        "low_level_qwen_calls": 0.0,
    }
    metrics["source_overlap"]["task_success"] = source_overlap
    metrics["source_overlap"]["stop_condition_correctness"] = source_overlap
    metrics["no_nsi_latent"]["task_success"] = no_nsi
    metrics["no_nsi_latent"]["stop_condition_correctness"] = no_nsi
    metrics["native_head_only_no_cache"]["task_success"] = native
    metrics["native_head_only_no_cache"]["stop_condition_correctness"] = native
    metrics["prompt_only"]["task_success"] = 0.20
    metrics["prompt_only"]["stop_condition_correctness"] = 0.20
    return _write(
        path,
        {
            "sealed_data_used_for_training_or_tuning": sealed,
            "metrics": metrics,
        },
    )


def test_phase2u_smoke_and_full_postflight_accept_interpretable_deltas(
    tmp_path: Path,
) -> None:
    paths = _dataset(tmp_path)
    data_health_report = build_phase2u_data_health(
        manifest_json=paths["manifest"],
        train_jsonl=paths["train"],
        val_jsonl=paths["val"],
        holdout_jsonl=paths["holdout"],
    )
    data_health = _write(tmp_path / "data_health.json", data_health_report)
    pretrain_report = build_phase2u_pretrain_gate(data_health_json=data_health)
    pretrain = _write(tmp_path / "pretrain.json", pretrain_report)
    eval_summary = _eval_summary(tmp_path / "eval_summary.json")

    smoke = build_phase2u_smoke_postflight(
        eval_summary_json=eval_summary,
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
    )

    assert smoke["passed"] is True
    assert smoke["ready_for_full_train"] is True
    assert len(smoke["nonzero_controls"]) >= 3
    assert smoke["metrics"]["full_minus_source_overlap_task_success"] > 0.31

    smoke_json = _write(tmp_path / "smoke.json", smoke)
    full = build_phase2u_full_postflight(
        eval_summary_json=eval_summary,
        smoke_postflight_json=smoke_json,
    )

    assert full["passed"] is True
    assert full["ready_for_package_gate"] is True
    assert full["ready_for_sealed_eval"] is False


def test_phase2u_smoke_postflight_rejects_sealed_or_weak_controls(
    tmp_path: Path,
) -> None:
    paths = _dataset(tmp_path)
    data_health_report = build_phase2u_data_health(
        manifest_json=paths["manifest"],
        train_jsonl=paths["train"],
        val_jsonl=paths["val"],
        holdout_jsonl=paths["holdout"],
    )
    data_health = _write(tmp_path / "data_health.json", data_health_report)
    pretrain_report = build_phase2u_pretrain_gate(data_health_json=data_health)
    pretrain = _write(tmp_path / "pretrain.json", pretrain_report)
    eval_summary = _eval_summary(
        tmp_path / "eval_summary.json",
        full=0.80,
        source_overlap=0.79,
        no_nsi=0.0,
        native=0.0,
        sealed=True,
    )

    report = build_phase2u_smoke_postflight(
        eval_summary_json=eval_summary,
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
    )

    assert report["passed"] is False
    assert "do_not_use_sealed_feedback_for_phase2u_postflight" in report["blocked_actions"]
    assert "do_not_claim_phase2u_delta_without_nonzero_controls" in report[
        "blocked_actions"
    ]


def test_phase2u_converter_preserves_public_trace_and_measured_controls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    controls = [control for control in CONTROLS if control != "full_package"]
    family_by_subset = {
        "control_feasible_easy": "dependency_or_import_mismatch",
        "control_feasible_medium": "stale_snapshot_update",
        "mechanism_required": "multi_file_traceback_relation",
        "safety_required": "safety_blocked_command_temptation",
        "false_completion_trap": "false_completion_trap",
    }
    for split, count in {"train": 40, "val": 20, "holdout": 20}.items():
        rows: list[dict] = []
        for index in range(count):
            subset = SUBSETS[index % len(SUBSETS)]
            expected = f"repair_action_{index % 4}"
            rows.append(
                {
                    "phase": "Phase2T",
                    "split": split,
                    "source_kind": "public_repo",
                    "repo_id": f"{split}_repo_{index:03d}",
                    "repo_url_or_origin": f"https://github.com/example/{split}_repo_{index:03d}.git",
                    "trace_id": f"{split}-{index}",
                    "current_visible_text": "Public repair task without candidate markers.",
                    "runtime_visible_evidence": {"stale_state_refresh": subset == "mechanism_required"},
                    "repair_candidates": [
                        {"repair_action": expected, "intent": "apply_patch_and_rerun_tests"},
                        {"repair_action": f"repair_action_wrong_{index}", "intent": "apply_patch_and_rerun_tests"},
                    ],
                    "difficulty": {
                        "task_family": family_by_subset[subset],
                        "candidate_count": 4 if subset == "mechanism_required" else 2,
                        "evidence_density": "high" if subset == "mechanism_required" else "low",
                        "safety_pressure": "rollback_required"
                        if subset == "safety_required"
                        else "none",
                    },
                    "expected_repair_action": expected,
                    "baseline_metadata": {
                        control: {
                            "measured": True,
                            "uses_expected_repair_action": False,
                            "uses_sealed_feedback": False,
                        }
                        for control in controls
                    },
                    "baselines": {
                        control: expected
                        if control in controls[: 3 + (index % 2)]
                        else f"repair_action_wrong_{index}"
                        for control in controls
                    },
                }
            )
        _write_jsonl(source / f"{split}.raw.jsonl", rows)

    output = tmp_path / "phase2u"
    manifest = build_phase2u_from_phase2t(source_root=source, output_root=output)

    assert manifest["phase"] == "Phase2U"
    assert manifest["benchmark_family"] == "baseline_feasible_repair_controls"
    assert set(manifest["split_counts"]) == {"train", "val", "holdout"}

    report = build_phase2u_data_health(
        manifest_json=output / "manifest.json",
        train_jsonl=output / "train.jsonl",
        val_jsonl=output / "val.jsonl",
        holdout_jsonl=output / "holdout.jsonl",
    )

    assert report["passed"] is True
    assert report["claim_bearing_training_ready"] is True


def test_phase2u_head_dataset_records_family_and_hashes(tmp_path: Path) -> None:
    paths = _dataset(tmp_path / "raw")
    data_health_report = build_phase2u_data_health(
        manifest_json=paths["manifest"],
        train_jsonl=paths["train"],
        val_jsonl=paths["val"],
        holdout_jsonl=paths["holdout"],
    )
    data_health = _write(tmp_path / "data_health.json", data_health_report)
    pretrain_report = build_phase2u_pretrain_gate(data_health_json=data_health)
    pretrain = _write(tmp_path / "pretrain.json", pretrain_report)

    manifest = build_phase2u_head_dataset(
        train_jsonl=paths["train"],
        val_jsonl=paths["val"],
        holdout_jsonl=paths["holdout"],
        output_dir=tmp_path / "heads",
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
    )

    assert manifest["dataset_family"] == "phase2u_baseline_feasible_repair_head_dataset"
    assert manifest["sealed_v3_used"] is False
    assert manifest["source_data_health_passed"] is True
    assert manifest["source_pretrain_gate_passed"] is True
    assert set(manifest["effective_split_hashes"]) == {
        "phase2u_train",
        "phase2u_val",
        "phase2u_holdout",
    }
    row = json.loads((tmp_path / "heads" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["prompt_style"] == "phase2u_baseline_feasible_repair_head_v1"
    assert row["source_trace"]["phase"] == "Phase2U"


def test_phase2u_smoke_eval_summary_uses_training_and_measured_baselines(
    tmp_path: Path,
) -> None:
    paths = _dataset(tmp_path / "raw")
    training = _write(
        tmp_path / "training_summary.json",
        {
            "adapter_name": "phase2u_smoke",
            "config_hash": "abc",
            "train_examples": 40,
            "val_examples": 20,
            "use_pairwise_command_reranker": False,
            "no_json_motor_target": True,
            "low_level_qwen_calls_target": 0,
            "history": [
                {
                    "val_metrics": {
                        "command_slot_accuracy": 0.95,
                        "action_accuracy": 1.0,
                        "pairwise_encoded_candidates": 0,
                    }
                }
            ],
        },
    )
    report = build_phase2u_smoke_eval_summary(
        training_summary_json=training,
        val_jsonl=paths["val"],
    )

    assert report["sealed_data_used_for_training_or_tuning"] is False
    assert report["metrics"]["full_package"]["task_success"] == 0.95
    assert report["metrics"]["full_package"]["low_level_qwen_calls"] == 0.0
    assert report["metrics"]["source_overlap"]["task_success"] > 0.0
    assert report["missing_controls"] == ["no_nsi_latent"]

    diagnostic = _write(
        tmp_path / "no_nsi_diagnostic.json",
        {"sources": {"effective": {"accuracy": 0.65}}},
    )
    with_no_nsi = build_phase2u_smoke_eval_summary(
        training_summary_json=training,
        val_jsonl=paths["val"],
        no_nsi_diagnostic_json=diagnostic,
    )

    assert with_no_nsi["metrics"]["no_nsi_latent"]["task_success"] == 0.65
    assert with_no_nsi["missing_controls"] == []


def test_phase2u_package_gate_requires_consistent_nonsealed_artifacts(tmp_path: Path) -> None:
    paths = _dataset(tmp_path / "raw")
    data_health_report = build_phase2u_data_health(
        manifest_json=paths["manifest"],
        train_jsonl=paths["train"],
        val_jsonl=paths["val"],
        holdout_jsonl=paths["holdout"],
    )
    data_health = _write(tmp_path / "data_health.json", data_health_report)
    pretrain_report = build_phase2u_pretrain_gate(data_health_json=data_health)
    pretrain = _write(tmp_path / "pretrain.json", pretrain_report)
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        {
            "dataset_family": "phase2u_baseline_feasible_repair_head_dataset",
            "sealed_v3_used": False,
            "source_data_health_passed": True,
            "source_pretrain_gate_passed": True,
            "command_identity_margin_gate_passed": True,
            "effective_split_hashes": data_health_report["effective_split_hashes"],
        },
    )
    summary = _write(
        tmp_path / "training_summary.json",
        {
            "no_json_motor_target": True,
            "low_level_qwen_calls_target": 0,
            "use_pairwise_command_reranker": False,
            "train_examples": 96,
            "val_examples": 64,
        },
    )
    smoke = _write(tmp_path / "smoke.json", {"passed": True})
    full = _write(tmp_path / "full.json", {"passed": True})
    adapter = tmp_path / "adapter"
    (adapter / "backbone_adapter").mkdir(parents=True)
    (adapter / "native_heads.pt").write_bytes(b"heads")
    (adapter / "head_config.json").write_text("{}", encoding="utf-8")

    report = build_phase2u_package_gate(
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
        head_manifest_json=head_manifest,
        training_summary_json=summary,
        smoke_postflight_json=smoke,
        full_postflight_json=full,
        adapter_dir=adapter,
    )

    assert report["passed"] is True
    assert report["ready_for_package"] is True
    assert report["ready_for_sealed_eval"] is False

    drifted_head = _write(
        tmp_path / "drifted_head_manifest.json",
        {
            "dataset_family": "phase2u_baseline_feasible_repair_head_dataset",
            "sealed_v3_used": False,
            "source_data_health_passed": True,
            "source_pretrain_gate_passed": True,
            "command_identity_margin_gate_passed": True,
            "effective_split_hashes": {"phase2u_train": "drifted"},
        },
    )

    drifted = build_phase2u_package_gate(
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
        head_manifest_json=drifted_head,
        training_summary_json=summary,
        smoke_postflight_json=smoke,
        full_postflight_json=full,
        adapter_dir=adapter,
    )

    assert drifted["passed"] is False
    assert "do_not_package_phase2u_with_split_hash_mismatch" in drifted["blocked_actions"]


def test_phase2u_postpackage_gate_requires_no_nsi_candidate_identity_ablation(
    tmp_path: Path,
) -> None:
    package_gate = _write(
        tmp_path / "package_gate.json",
        {
            "passed": True,
            "ready_for_package": True,
            "ready_for_sealed_eval": False,
        },
    )
    full = tmp_path / "pkg_full"
    no_nsi = tmp_path / "pkg_no_nsi"
    native = tmp_path / "pkg_native"
    continuation = tmp_path / "pkg_continuation"
    full_manifest = write_native_nervous_package(
        full,
        base_model_name="model",
        native_head_path="adapter",
        low_level_checkpoint_path="nsi.pt",
        policy_label="phase2u_full",
    )
    write_native_nervous_package(
        no_nsi,
        base_model_name=full_manifest["base_model_name"],
        native_head_path=full_manifest["native_head_path"],
        low_level_checkpoint_path=full_manifest["low_level_checkpoint_path"],
        policy_label="phase2u_full_no_nsi_latent",
        zero_nsi_latent=True,
    )
    write_native_nervous_package(
        native,
        base_model_name=full_manifest["base_model_name"],
        native_head_path=full_manifest["native_head_path"],
        low_level_checkpoint_path=full_manifest["low_level_checkpoint_path"],
        policy_label="phase2u_full_native_head_only",
        continuation_cache_enabled=False,
    )
    write_native_nervous_package(
        continuation,
        base_model_name=full_manifest["base_model_name"],
        native_head_path=full_manifest["native_head_path"],
        low_level_checkpoint_path=full_manifest["low_level_checkpoint_path"],
        policy_label="phase2u_full_continuation_only",
        native_head_calls_enabled=False,
    )

    report = build_phase2u_postpackage_gate(
        package_gate_json=package_gate,
        full_package_path=full,
        no_nsi_package_path=no_nsi,
        native_head_only_package_path=native,
        continuation_only_package_path=continuation,
    )

    assert report["passed"] is True
    assert report["ready_for_sealed_eval"] is True
    assert report["ready_for_claim_upgrade"] is False
    assert report["checks"]["no_nsi_control_disables_candidate_identity"] is True


def test_phase2u_sealed_transfer_report_keeps_bounded_claim_boundary(
    tmp_path: Path,
) -> None:
    def eval_json(name: str, completion: float) -> Path:
        run = tmp_path / f"{name}_run"
        run.mkdir()
        (run / "trace_rows.jsonl").write_text("", encoding="utf-8")
        return _write(
            tmp_path / f"{name}.json",
            {
                "run_path": str(run),
                "episode_count": 4,
                "policy": {"policy_label": name},
                "metrics": {
                    "aggregate": {
                        "task_completion_rate": {
                            "mean": completion,
                            "positives": round(completion * 4),
                        },
                        "model_calls": {"mean": 1.0},
                        "state_hallucination_rate": {"mean": 0.0},
                    }
                },
            },
        )

    gate = _write(
        tmp_path / "gate.json",
        {
            "passed": True,
            "metrics": {
                "full_completion": 1.0,
                "no_nsi_completion": 0.0,
                "native_head_only_completion": 0.0,
                "continuation_only_completion": 0.0,
            },
        },
    )
    zero = _write(
        tmp_path / "zero.json",
        {
            "passed": True,
            "ready_for_bounded_sealed_claim": True,
            "ready_for_strong_architecture_claim": False,
            "checks": {
                "all_zero_controls_classified": True,
                "no_suspicious_unexplained_zero": True,
            },
            "all_controls_zero": True,
            "interpretation": {"bounded_claim": "bounded"},
        },
    )
    postpackage = _write(
        tmp_path / "postpackage.json",
        {
            "passed": True,
            "ready_for_claim_upgrade": False,
            "checks": {"no_nsi_control_disables_candidate_identity": True},
        },
    )

    report = build_phase2u_sealed_transfer_report(
        external_gate_json=gate,
        zero_baseline_audit_json=zero,
        postpackage_gate_json=postpackage,
        full_eval_json=eval_json("full", 1.0),
        prompt_eval_json=eval_json("prompt", 0.0),
        react_eval_json=eval_json("react", 0.0),
        no_nsi_eval_json=eval_json("no_nsi", 0.0),
        native_head_only_eval_json=eval_json("native", 0.0),
        continuation_only_eval_json=eval_json("continuation", 0.0),
    )

    assert report["passed"] is True
    assert (
        report["claim_scope"]
        == "phase2u_extreme_sealed_transfer_positive_but_not_mechanism_sufficient"
    )
    assert report["mechanism_sufficiency_passed"] is False
    assert (
        "sealed-v3 all-zero controls do not prove a graded mechanism curve"
        in report["unsupported_claims"]
    )
    assert "epoch-making architecture claim is not proven" in report["unsupported_claims"]


def test_phase2u_evidence_sufficiency_separates_nonsealed_curve_from_sealed_stress(
    tmp_path: Path,
) -> None:
    nonsealed = _write(
        tmp_path / "nonsealed.json",
        {
            "passed": True,
            "nonzero_controls": ["source_overlap", "prompt_only", "no_nsi_latent"],
            "metrics": {
                "full_task_success": 1.0,
                "best_non_full_task_success": 0.50,
                "full_minus_best_non_full_task_success": 0.50,
            },
        },
    )
    sealed = _write(
        tmp_path / "sealed.json",
        {
            "passed": True,
            "mechanism_sufficiency_passed": False,
            "claim_scope": "phase2u_extreme_sealed_transfer_positive_but_not_mechanism_sufficient",
            "unsupported_claims": [
                "sealed-v3 all-zero controls do not prove a graded mechanism curve"
            ],
            "metrics": {"full_completion": 1.0, "best_mechanism_completion": 0.0},
        },
    )

    report = build_phase2u_evidence_sufficiency_report(
        nonsealed_full_postflight_json=nonsealed,
        sealed_transfer_report_json=sealed,
    )

    assert report["passed"] is True
    assert report["nonsealed_mechanism_curve_supported"] is True
    assert report["sealed_stress_observation_supported"] is True
    assert report["sealed_mechanism_curve_supported"] is False
    assert "do_not_describe_sealed_v3_all_zero_deltas_as_sufficient_proof" in report[
        "blocked_actions"
    ]
