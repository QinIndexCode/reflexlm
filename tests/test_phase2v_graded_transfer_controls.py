import json
from pathlib import Path

from reflexlm.cli.audit_phase2v_graded_transfer_controls import (
    build_phase2v_data_health,
    build_phase2v_eval_postflight,
    build_phase2v_independence_audit,
    build_phase2v_pretrain_gate,
    build_phase2v_sealed_block_gate,
)
from reflexlm.cli.build_phase2v_evidence_sufficiency_report import (
    build_phase2v_evidence_sufficiency_report,
)
from reflexlm.cli.build_phase2v_eval_summary import build_phase2v_eval_summary
from reflexlm.cli.build_phase2v_graded_transfer_controls import build_phase2v_from_phase2u


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
NON_FULL = [control for control in CONTROLS if control != "full_package"]
TIERS = ["control_feasible", "mixed_mechanism", "mechanism_required"]


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _metadata() -> dict:
    rows = {
        control: {
            "measured": True,
            "declared_only": False,
            "uses_expected_repair_action": False,
            "uses_sealed_feedback": False,
        }
        for control in CONTROLS
    }
    rows["full_package"]["oracle_reference"] = True
    rows["full_package"]["uses_expected_repair_action"] = True
    rows["no_nsi_latent"]["measured"] = False
    rows["no_nsi_latent"]["requires_trained_ablation"] = True
    rows["no_nsi_latent"]["deferred_until_postflight"] = True
    return rows


def _row(split: str, index: int, *, marker: bool = False, all_zero: bool = False, ceiling: bool = False) -> dict:
    tier = TIERS[index % len(TIERS)]
    baseline_results = {
        control: {"task_success": 0.0, "stop_condition_correctness": 0.0, "unsafe_write_count": 0.0}
        for control in CONTROLS
    }
    baseline_results["full_package"]["task_success"] = 1.0
    baseline_results["full_package"]["stop_condition_correctness"] = 1.0
    if ceiling:
        baseline_results["prompt_only"]["task_success"] = 1.0
        baseline_results["prompt_only"]["stop_condition_correctness"] = 1.0
    elif not all_zero:
        for offset, control in enumerate(["prompt_only", "react", "source_overlap", "modern_coding_agent_loop"]):
            if (index + offset) % 4 == 0:
                baseline_results[control]["task_success"] = 1.0
                baseline_results[control]["stop_condition_correctness"] = 1.0
    return {
        "phase": "Phase2V",
        "benchmark_family": "graded_transfer_nonzero_controls",
        "trace_construction_mode": "phase2v_graded_transfer_nonzero_control_trace",
        "split": split,
        "source_kind": "public_repo",
        "repo_id": f"{split}_repo_{index:03d}",
        "repo_url_or_origin": f"https://github.com/example/{split}_repo_{index:03d}.git",
        "phase2v_tier": tier,
        "current_visible_text": "Public repair transfer task." + (" candidate_0" if marker else ""),
        "runtime_visible_evidence": {"tier": tier},
        "repair_candidates": [{"repair_action": "a"}, {"repair_action": "b"}],
        "baseline_metadata": _metadata(),
        "baseline_results": baseline_results,
    }


def _dataset(tmp_path: Path, *, all_zero: bool = False, ceiling: bool = False, marker: bool = False) -> dict[str, Path]:
    manifest = _write(
        tmp_path / "manifest.json",
        {"phase": "Phase2V", "benchmark_family": "graded_transfer_nonzero_controls"},
    )
    return {
        "manifest": manifest,
        "train": _write_jsonl(
            tmp_path / "train.jsonl",
            [_row("train", index, all_zero=all_zero, ceiling=ceiling) for index in range(64)],
        ),
        "val": _write_jsonl(
            tmp_path / "val.jsonl",
            [_row("val", index, marker=marker, all_zero=all_zero, ceiling=ceiling) for index in range(64)],
        ),
        "holdout": _write_jsonl(
            tmp_path / "holdout.jsonl",
            [_row("holdout", index, all_zero=all_zero, ceiling=ceiling) for index in range(64)],
        ),
    }


def test_phase2v_data_health_accepts_nonzero_repo_disjoint_transfer(tmp_path: Path) -> None:
    paths = _dataset(tmp_path)
    report = build_phase2v_data_health(
        manifest_json=paths["manifest"],
        train_jsonl=paths["train"],
        val_jsonl=paths["val"],
        holdout_jsonl=paths["holdout"],
    )
    assert report["passed"] is True
    assert report["rollups"]["val"]["best_nonfull_task_success"] == 0.25
    assert len(report["rollups"]["val"]["nonzero_controls"]) >= 3

    gate = build_phase2v_pretrain_gate(data_health_json=_write(tmp_path / "health.json", report))
    assert gate["passed"] is True
    assert gate["ready_for_sealed_eval"] is False


def test_phase2v_independence_audit_rejects_repo_and_trace_overlap(tmp_path: Path) -> None:
    paths = _dataset(tmp_path / "ok")
    report = build_phase2v_independence_audit(
        manifest_json=paths["manifest"],
        train_jsonl=paths["train"],
        val_jsonl=paths["val"],
        holdout_jsonl=paths["holdout"],
    )
    assert report["passed"] is False
    assert "rewrite_phase2v_trace_ids_with_phase2v_namespace" in report["blocked_actions"]

    phase2v_rows = []
    for index in range(64):
        row = _row("val", index)
        row["trace_id"] = f"phase2v:val:{index}"
        row["phase2v_source_trace_id"] = f"source:val:{index}"
        phase2v_rows.append(row)
    overlap = _row("train", 0)
    overlap["trace_id"] = "phase2v:val:0"
    overlap["phase2v_source_trace_id"] = "source:val:0"
    overlap["repo_url_or_origin"] = phase2v_rows[0]["repo_url_or_origin"]
    bad = {
        "manifest": paths["manifest"],
        "train": _write_jsonl(tmp_path / "bad" / "train.jsonl", [overlap] + phase2v_rows[1:64]),
        "val": _write_jsonl(tmp_path / "bad" / "val.jsonl", phase2v_rows),
        "holdout": _write_jsonl(
            tmp_path / "bad" / "holdout.jsonl",
            [
                {
                    **_row("holdout", index),
                    "trace_id": f"phase2v:holdout:{index}",
                    "phase2v_source_trace_id": f"source:holdout:{index}",
                }
                for index in range(64)
            ],
        ),
    }
    bad_report = build_phase2v_independence_audit(
        manifest_json=bad["manifest"],
        train_jsonl=bad["train"],
        val_jsonl=bad["val"],
        holdout_jsonl=bad["holdout"],
    )
    assert bad_report["passed"] is False
    assert "fix_phase2v_repo_origin_overlap" in bad_report["blocked_actions"]
    assert "fix_phase2v_trace_identity_overlap" in bad_report["blocked_actions"]


def test_phase2v_data_health_rejects_all_zero_ceiling_and_markers(tmp_path: Path) -> None:
    zero = _dataset(tmp_path / "zero", all_zero=True)
    zero_report = build_phase2v_data_health(
        manifest_json=zero["manifest"],
        train_jsonl=zero["train"],
        val_jsonl=zero["val"],
        holdout_jsonl=zero["holdout"],
    )
    assert zero_report["passed"] is False
    assert "do_not_use_phase2v_all_zero_control_transfer" in zero_report["blocked_actions"]

    ceiling = _dataset(tmp_path / "ceiling", ceiling=True)
    ceiling_report = build_phase2v_data_health(
        manifest_json=ceiling["manifest"],
        train_jsonl=ceiling["train"],
        val_jsonl=ceiling["val"],
        holdout_jsonl=ceiling["holdout"],
    )
    assert ceiling_report["passed"] is False
    assert "rebalance_phase2v_controls_before_training" in ceiling_report["blocked_actions"]

    marker = _dataset(tmp_path / "marker", marker=True)
    marker_report = build_phase2v_data_health(
        manifest_json=marker["manifest"],
        train_jsonl=marker["train"],
        val_jsonl=marker["val"],
        holdout_jsonl=marker["holdout"],
    )
    assert marker_report["passed"] is False
    assert marker_report["checks"]["phase2v_no_visible_candidate_or_gold_markers"] is False


def test_phase2v_eval_postflight_requires_full_to_beat_nonzero_controls(tmp_path: Path) -> None:
    paths = _dataset(tmp_path / "data")
    health = _write(
        tmp_path / "health.json",
        build_phase2v_data_health(
            manifest_json=paths["manifest"],
            train_jsonl=paths["train"],
            val_jsonl=paths["val"],
            holdout_jsonl=paths["holdout"],
        ),
    )
    pretrain = _write(tmp_path / "pretrain.json", build_phase2v_pretrain_gate(data_health_json=health))
    summary = _write(
        tmp_path / "summary.json",
        {
            "sealed_data_used_for_training_or_tuning": False,
            "phase2v_row_level_predictions_recomputed": True,
            "missing_controls": [],
            "metrics": {
                "full_package": {
                    "task_success": 1.0,
                    "state_hallucination_rate": 0.0,
                    "low_level_qwen_calls": 0.0,
                },
                "prompt_only": {"task_success": 0.50},
                "react": {"task_success": 0.25},
                "source_overlap": {"task_success": 0.25},
                "modern_coding_agent_loop": {"task_success": 0.25},
                "no_nsi_latent": {"task_success": 0.30},
                "native_head_only_no_cache": {"task_success": 0.40},
                "continuation_only": {"task_success": 0.25},
            },
        },
    )
    report = build_phase2v_eval_postflight(
        eval_summary_json=summary,
        data_health_json=health,
        pretrain_gate_json=pretrain,
    )
    assert report["passed"] is True

    weak_summary = json.loads(summary.read_text(encoding="utf-8"))
    weak_summary["metrics"]["full_package"]["task_success"] = 0.60
    weak = _write(tmp_path / "weak.json", weak_summary)
    weak_report = build_phase2v_eval_postflight(
        eval_summary_json=weak,
        data_health_json=health,
        pretrain_gate_json=pretrain,
    )
    assert weak_report["passed"] is False
    assert "freeze_phase2v_mechanism_insufficiency" in weak_report["blocked_actions"]

    derived_summary = json.loads(summary.read_text(encoding="utf-8"))
    derived_summary["phase2v_row_level_predictions_recomputed"] = False
    derived = _write(tmp_path / "derived.json", derived_summary)
    derived_report = build_phase2v_eval_postflight(
        eval_summary_json=derived,
        data_health_json=health,
        pretrain_gate_json=pretrain,
    )
    assert derived_report["passed"] is False
    assert (
        "do_not_treat_phase2v_derived_summary_as_claim_bearing_eval"
        in derived_report["blocked_actions"]
    )


def test_phase2v_builders_and_evidence_report_keep_sealed_blocked(tmp_path: Path) -> None:
    source = tmp_path / "phase2u"
    for split in ["train", "val", "holdout"]:
        rows = []
        for index in range(64):
            row = _row(split, index)
            row["phase"] = "Phase2U"
            row["benchmark_family"] = "baseline_feasible_repair_controls"
            row["trace_construction_mode"] = "phase2u_baseline_feasible_repair_control_trace"
            row["phase2u_subset"] = "control_feasible_easy" if index % 3 == 0 else "mechanism_required"
            rows.append(row)
        _write_jsonl(source / f"{split}.jsonl", rows)
    manifest = build_phase2v_from_phase2u(source_root=source, output_root=tmp_path / "phase2v")
    assert manifest["phase"] == "Phase2V"
    converted = json.loads((tmp_path / "phase2v" / "val.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert converted["trace_id"].startswith("phase2v:")
    assert converted["trace_id"] != converted["phase2v_source_trace_id"]
    independence = build_phase2v_independence_audit(
        manifest_json=tmp_path / "phase2v" / "manifest.json",
        train_jsonl=tmp_path / "phase2v" / "train.jsonl",
        val_jsonl=tmp_path / "phase2v" / "val.jsonl",
        holdout_jsonl=tmp_path / "phase2v" / "holdout.jsonl",
    )
    assert independence["passed"] is True

    phase2u_evidence = _write(
        tmp_path / "phase2u_evidence.json",
        {"passed": True, "sealed_mechanism_curve_supported": False, "claim_scope": "phase2u_two_layer_bounded_evidence"},
    )
    postflight = _write(
        tmp_path / "postflight.json",
        {
            "passed": True,
            "nonzero_controls": ["prompt_only", "react", "source_overlap"],
            "metrics": {
                "full_task_success": 1.0,
                "best_nonfull_task_success": 0.5,
                "full_minus_best_nonfull_task_success": 0.5,
            },
        },
    )
    sealed_block = _write(
        tmp_path / "sealed_block.json",
        {
            "ready_for_sealed_eval": False,
            "blocked_actions": ["do_not_upgrade_to_production_autonomy_or_epoch_making_claim"],
        },
    )
    report = build_phase2v_evidence_sufficiency_report(
        phase2u_evidence_json=phase2u_evidence,
        phase2v_postflight_json=postflight,
        phase2v_sealed_block_json=sealed_block,
        phase2v_independence_json=_write(
            tmp_path / "independence.json",
            {"passed": True, "identity_hashes": {"phase2v_trace_ids": "abc"}},
        ),
    )
    assert report["passed"] is True
    assert "Phase2V does not prove an epoch-making architecture." in report["unsupported_claims"]

    eval_summary = build_phase2v_eval_summary(
        source_eval_summary_json=_write(
            tmp_path / "source_eval.json",
            {"missing_controls": [], "metrics": {"full_package": {"task_success": 1.0}}},
        ),
        val_jsonl=tmp_path / "phase2v" / "val.jsonl",
        data_health_json=postflight,
        pretrain_gate_json=postflight,
    )
    assert eval_summary["sealed_data_used_for_training_or_tuning"] is False
    assert eval_summary["phase2v_row_level_predictions_recomputed"] is False
    assert (
        "do_not_use_phase2v_derived_summary_as_independent_architecture_evidence"
        in eval_summary["blocked_claims"]
    )
