import json
from pathlib import Path

from reflexlm.cli.audit_phase2k_continuation_pressure import (
    build_phase2k_data_health,
    build_phase2k_postflight,
    build_phase2k_sealed_gate,
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
    model_calls: float = 0.0,
    dataset_path: str = "artifacts/datasets/phase2i_external_trace_v3_semantic_required/challenge.jsonl",
) -> Path:
    run_path = tmp_path / "runs" / name
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "trace_rows.jsonl").write_text("", encoding="utf-8")
    return _write(
        tmp_path / f"{name}.json",
        {
            "run_path": str(run_path),
            "dataset": {"dataset_path": dataset_path},
            "policy": {"policy_label": name},
            "metrics": {
                "aggregate": {
                    "task_completion_rate": {"mean": completion, "count": 16},
                    "command_decision_accuracy": {"mean": completion, "count": 16},
                    "model_calls": {"mean": model_calls, "count": 16},
                    "state_hallucination_rate": {"mean": 0.0, "count": 16},
                }
            },
        },
    )


def test_phase2k_data_health_accepts_nonsealed_continuation_pressure_profiles(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2k_continuation_pressure_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2k_continuation_pressure_val",
        episodes_per_scenario=1,
    )

    report = build_phase2k_data_health(
        train_jsonl=train_dir / "challenge.jsonl",
        val_jsonl=val_dir / "challenge.jsonl",
        train_metadata_json=train_dir / "episode_metadata.json",
        val_metadata_json=val_dir / "episode_metadata.json",
    )

    assert report["passed"] is True
    assert report["checks"]["phase2k_prior_command_memory_required"] is True
    assert report["checks"]["phase2k_source_overlap_baseline_below_threshold"] is True
    assert set(report["rollups"]["val"]["continuation_depths"]) == {
        "one_step",
        "two_step",
        "stale_state_refresh",
    }


def test_phase2k_data_health_rejects_source_overlap_solved_val(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2k_continuation_pressure_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2k_continuation_pressure_val",
        episodes_per_scenario=1,
    )

    report = build_phase2k_data_health(
        train_jsonl=train_dir / "challenge.jsonl",
        val_jsonl=val_dir / "challenge.jsonl",
        train_metadata_json=train_dir / "episode_metadata.json",
        val_metadata_json=val_dir / "episode_metadata.json",
        max_source_overlap_val_accuracy=-0.01,
    )

    assert report["passed"] is False
    assert "do_not_train_when_source_overlap_solves_phase2k_val" in report["blocked_actions"]


def test_phase2k_postflight_requires_full_to_beat_native_head_only(
    tmp_path: Path,
) -> None:
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "checks": {"phase2k_source_overlap_baseline_below_threshold": True},
            "inputs": {"val_jsonl": "artifacts/datasets/phase2k_continuation_pressure_val/challenge.jsonl"},
        },
    )
    full = _eval_payload(tmp_path, "full", 0.92)
    native = _eval_payload(tmp_path, "native", 0.88)

    report = build_phase2k_postflight(
        data_health_json=data_health,
        full_eval_json=full,
        native_head_only_eval_json=native,
    )

    assert report["passed"] is False
    assert report["checks"]["full_beats_native_head_only_by_required_delta"] is False
    assert "do_not_package_without_full_beating_native_head_only" in report["blocked_actions"]


def test_phase2k_postflight_allows_package_after_native_delta(
    tmp_path: Path,
) -> None:
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "checks": {"phase2k_source_overlap_baseline_below_threshold": True},
            "inputs": {"val_jsonl": "artifacts/datasets/phase2k_continuation_pressure_val/challenge.jsonl"},
        },
    )
    full = _eval_payload(tmp_path, "full", 0.95)
    native = _eval_payload(tmp_path, "native", 0.80)

    report = build_phase2k_postflight(
        data_health_json=data_health,
        full_eval_json=full,
        native_head_only_eval_json=native,
    )

    assert report["passed"] is True
    assert report["ready_for_package"] is True
    assert report["allowed_next_action"] == "run_phase2k_package_only"


def test_phase2k_smoke_postflight_allows_only_full_training(
    tmp_path: Path,
) -> None:
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "checks": {"phase2k_source_overlap_baseline_below_threshold": True},
            "inputs": {"val_jsonl": "artifacts/datasets/phase2k_continuation_pressure_val/challenge.jsonl"},
        },
    )
    full = _eval_payload(tmp_path, "full", 0.95)
    native = _eval_payload(tmp_path, "native", 0.80)

    report = build_phase2k_postflight(
        data_health_json=data_health,
        full_eval_json=full,
        native_head_only_eval_json=native,
        postflight_stage="smoke",
    )

    assert report["passed"] is True
    assert report["ready_for_full_train"] is True
    assert report["ready_for_package"] is False
    assert report["allowed_next_action"] == "run_phase2k_full_nonsealed_training_only"


def test_phase2k_sealed_gate_keeps_bounded_claim_when_full_ties_native(
    tmp_path: Path,
) -> None:
    full = _eval_payload(tmp_path, "full", 0.33, model_calls=1.0)
    no_nsi = _eval_payload(tmp_path, "no_nsi", 0.05, model_calls=1.0)
    native = _eval_payload(tmp_path, "native", 0.33, model_calls=2.0)
    continuation = _eval_payload(tmp_path, "continuation", 0.0)
    prompt = _eval_payload(tmp_path, "prompt", 0.0, model_calls=1.0)
    react = _eval_payload(tmp_path, "react", 0.0, model_calls=2.0)

    report = build_phase2k_sealed_gate(
        full_eval_json=full,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        prompt_only_eval_json=prompt,
        react_eval_json=react,
    )

    assert report["passed"] is False
    assert report["checks"]["full_beats_no_nsi_by_required_delta"] is True
    assert report["checks"]["full_beats_native_head_only_by_required_delta"] is False
    assert report["checks"]["full_low_level_qwen_calls_zero"] is False
    assert report["claim_boundary"] == "bounded_claim_only_do_not_upgrade_to_full_package_necessity"
