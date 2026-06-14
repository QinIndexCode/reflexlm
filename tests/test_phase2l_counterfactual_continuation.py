import json
from pathlib import Path

from reflexlm.cli.audit_phase2l_counterfactual_continuation import (
    build_phase2l_data_health,
    build_phase2l_postflight,
    build_phase2l_pretrain_gate,
    build_phase2l_sealed_gate,
)
from reflexlm.cli.generate_debug_cortex_challenge import build_debug_cortex_challenge


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _eval_payload(
    tmp_path: Path,
    name: str,
    completion: float,
    *,
    continuation_control: str = "normal",
    continuation_cache_enabled: bool = True,
    dataset_path: str = "artifacts/datasets/phase2l_counterfactual_continuation_val/challenge.jsonl",
) -> Path:
    run_path = tmp_path / "runs" / name
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "trace_rows.jsonl").write_text("", encoding="utf-8")
    return _write(
        tmp_path / f"{name}.json",
        {
            "run_path": str(run_path),
            "dataset": {"dataset_path": dataset_path},
            "policy": {
                "policy_label": name,
                "continuation_control": continuation_control,
                "continuation_cache_enabled": continuation_cache_enabled,
            },
            "metrics": {
                "aggregate": {
                    "task_completion_rate": {"mean": completion, "count": 12},
                    "command_decision_accuracy": {"mean": completion, "count": 12},
                    "model_calls": {"mean": 0.0, "count": 12},
                    "state_hallucination_rate": {"mean": 0.0, "count": 12},
                }
            },
        },
    )


def test_phase2l_data_health_accepts_counterfactual_continuation_pairs(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2l_counterfactual_continuation_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2l_counterfactual_continuation_val",
        episodes_per_scenario=1,
    )

    report = build_phase2l_data_health(
        train_jsonl=train_dir / "challenge.jsonl",
        val_jsonl=val_dir / "challenge.jsonl",
        train_metadata_json=train_dir / "episode_metadata.json",
        val_metadata_json=val_dir / "episode_metadata.json",
    )

    assert report["passed"] is True
    assert report["checks"]["phase2l_current_visible_state_hash_equal"] is True
    assert report["checks"]["phase2l_prior_context_hash_different"] is True
    assert report["checks"]["phase2l_correct_command_differs_by_pair"] is True
    assert report["rollups"]["pairs"]["val"]["passed_pair_count"] == 6
    assert report["rollups"]["source_overlap"]["val"]["accuracy"] <= 0.60


def test_phase2l_pretrain_gate_allows_only_after_data_health_passes(tmp_path: Path) -> None:
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "checks": {
                "phase2l_val_counterfactual_pairs_complete": True,
                "phase2l_source_overlap_baseline_below_threshold": True,
            },
            "inputs": {
                "val_jsonl": "artifacts/datasets/phase2l_counterfactual_continuation_val/challenge.jsonl"
            },
        },
    )

    report = build_phase2l_pretrain_gate(data_health_json=data_health)

    assert report["passed"] is True
    assert report["allowed_next_action"] == "run_phase2l_smoke_training_only"


def test_phase2l_data_health_rejects_source_overlap_solved_val(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2l_counterfactual_continuation_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2l_counterfactual_continuation_val",
        episodes_per_scenario=1,
    )

    report = build_phase2l_data_health(
        train_jsonl=train_dir / "challenge.jsonl",
        val_jsonl=val_dir / "challenge.jsonl",
        train_metadata_json=train_dir / "episode_metadata.json",
        val_metadata_json=val_dir / "episode_metadata.json",
        max_source_overlap_val_accuracy=-0.01,
    )

    assert report["passed"] is False
    assert "do_not_train_when_source_overlap_solves_phase2l_val" in report["blocked_actions"]


def test_phase2l_postflight_requires_wrong_cache_and_cache_erased_deltas(
    tmp_path: Path,
) -> None:
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "checks": {"phase2l_source_overlap_baseline_below_threshold": True},
            "inputs": {
                "val_jsonl": "artifacts/datasets/phase2l_counterfactual_continuation_val/challenge.jsonl"
            },
        },
    )
    full = _eval_payload(tmp_path, "full", 0.92)
    native = _eval_payload(tmp_path, "native", 0.70, continuation_control="cache_erased")
    wrong = _eval_payload(tmp_path, "wrong", 0.75, continuation_control="wrong_cache")
    erased = _eval_payload(
        tmp_path,
        "erased",
        0.75,
        continuation_control="cache_erased",
        continuation_cache_enabled=False,
    )

    report = build_phase2l_postflight(
        data_health_json=data_health,
        full_eval_json=full,
        native_head_only_eval_json=native,
        wrong_cache_eval_json=wrong,
        cache_erased_eval_json=erased,
    )

    assert report["passed"] is False
    assert report["checks"]["full_beats_wrong_cache_by_required_delta"] is False
    assert report["checks"]["full_beats_cache_erased_by_required_delta"] is False
    assert "do_not_package_without_full_beating_wrong_cache" in report["blocked_actions"]


def test_phase2l_smoke_postflight_cannot_be_package_ready(tmp_path: Path) -> None:
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "checks": {"phase2l_source_overlap_baseline_below_threshold": True},
            "inputs": {
                "val_jsonl": "artifacts/datasets/phase2l_counterfactual_continuation_val/challenge.jsonl"
            },
        },
    )
    full = _eval_payload(tmp_path, "full", 0.95)
    native = _eval_payload(tmp_path, "native", 0.75, continuation_control="cache_erased")
    wrong = _eval_payload(tmp_path, "wrong", 0.60, continuation_control="wrong_cache")
    erased = _eval_payload(
        tmp_path,
        "erased",
        0.60,
        continuation_control="cache_erased",
        continuation_cache_enabled=False,
    )

    report = build_phase2l_postflight(
        data_health_json=data_health,
        full_eval_json=full,
        native_head_only_eval_json=native,
        wrong_cache_eval_json=wrong,
        cache_erased_eval_json=erased,
        postflight_stage="smoke",
    )

    assert report["passed"] is True
    assert report["ready_for_full_train"] is True
    assert report["ready_for_package"] is False
    assert report["allowed_next_action"] == "run_phase2l_full_nonsealed_training_only"


def test_phase2l_local_full512_postflight_is_not_package_ready(tmp_path: Path) -> None:
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "checks": {"phase2l_source_overlap_baseline_below_threshold": True},
            "inputs": {
                "val_jsonl": "artifacts/datasets/phase2l_counterfactual_continuation_val/challenge.jsonl"
            },
        },
    )
    full = _eval_payload(tmp_path, "full", 1.0)
    native = _eval_payload(tmp_path, "native", 0.40, continuation_control="cache_erased")
    wrong = _eval_payload(tmp_path, "wrong", 0.0, continuation_control="wrong_cache")
    erased = _eval_payload(
        tmp_path,
        "erased",
        0.40,
        continuation_control="cache_erased",
        continuation_cache_enabled=False,
    )

    report = build_phase2l_postflight(
        data_health_json=data_health,
        full_eval_json=full,
        native_head_only_eval_json=native,
        wrong_cache_eval_json=wrong,
        cache_erased_eval_json=erased,
        postflight_stage="local_full512",
    )

    assert report["passed"] is True
    assert report["ready_for_full_train"] is False
    assert report["ready_for_larger_gpu_full1024"] is True
    assert report["ready_for_package"] is False
    assert report["allowed_next_action"] == "run_phase2l_full1024_on_larger_gpu_or_repeat_local_full512_seed"


def test_phase2l_sealed_gate_requires_full_to_beat_continuation_controls(
    tmp_path: Path,
) -> None:
    sealed = "artifacts/datasets/phase2i_external_trace_v3_semantic_required/challenge.jsonl"
    full = _eval_payload(tmp_path, "full", 0.90, dataset_path=sealed)
    no_nsi = _eval_payload(tmp_path, "no_nsi", 0.78, dataset_path=sealed)
    native = _eval_payload(tmp_path, "native", 0.80, dataset_path=sealed)
    continuation = _eval_payload(tmp_path, "continuation", 0.82, dataset_path=sealed)
    prompt = _eval_payload(
        tmp_path,
        "prompt",
        0.10,
        continuation_cache_enabled=False,
        dataset_path=sealed,
    )
    react = _eval_payload(
        tmp_path,
        "react",
        0.10,
        continuation_cache_enabled=False,
        dataset_path=sealed,
    )

    report = build_phase2l_sealed_gate(
        full_eval_json=full,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        prompt_only_eval_json=prompt,
        react_eval_json=react,
    )

    assert report["passed"] is False
    assert report["checks"]["full_completion_gate_passed"] is True
    assert report["checks"]["full_beats_no_nsi_by_required_delta"] is False
    assert report["checks"]["full_beats_native_head_only_by_required_delta"] is False
    assert report["checks"]["full_beats_continuation_only_by_required_delta"] is False
    assert (
        report["claim_boundary"]
        == "bounded_claim_only_do_not_upgrade_continuation_memory_necessity"
    )


def test_phase2l_sealed_gate_can_support_stronger_bounded_upgrade_when_all_deltas_pass(
    tmp_path: Path,
) -> None:
    sealed = "artifacts/datasets/phase2i_external_trace_v3_semantic_required/challenge.jsonl"
    full = _eval_payload(tmp_path, "full", 0.92, dataset_path=sealed)
    no_nsi = _eval_payload(tmp_path, "no_nsi", 0.70, dataset_path=sealed)
    native = _eval_payload(tmp_path, "native", 0.70, dataset_path=sealed)
    continuation = _eval_payload(tmp_path, "continuation", 0.60, dataset_path=sealed)
    prompt = _eval_payload(
        tmp_path,
        "prompt",
        0.10,
        continuation_cache_enabled=False,
        dataset_path=sealed,
    )
    react = _eval_payload(
        tmp_path,
        "react",
        0.10,
        continuation_cache_enabled=False,
        dataset_path=sealed,
    )

    report = build_phase2l_sealed_gate(
        full_eval_json=full,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        prompt_only_eval_json=prompt,
        react_eval_json=react,
    )

    assert report["passed"] is True
    assert report["checks"]["sealed_v3_inputs_only"] is True
    assert round(report["metrics"]["deltas"]["full_minus_native_head_only"], 6) == 0.22
    assert (
        report["claim_boundary"]
        == "sealed_v3_supports_counterfactual_continuation_memory_necessity"
    )
