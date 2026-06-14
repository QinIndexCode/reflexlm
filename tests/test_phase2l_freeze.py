import json
from pathlib import Path

from reflexlm.cli.freeze_phase2l_localfull512 import (
    build_phase2l_localfull512_freeze_manifest,
)
from reflexlm.cli.freeze_phase2l_counterfactual_continuation import (
    build_phase2l_freeze_manifest,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _eval(path: Path, completion: float, *, control: str = "normal") -> Path:
    return _write(
        path,
        {
            "policy": {"continuation_control": control},
            "dataset": {
                "dataset_path": "artifacts/datasets/phase2l_counterfactual_continuation_val/challenge.jsonl"
            },
            "metrics": {
                "aggregate": {
                    "task_completion_rate": {"mean": completion, "count": 24},
                    "command_decision_accuracy": {"mean": completion, "count": 24},
                    "model_calls": {"mean": 0.0, "count": 24},
                    "token_equivalent_cost": {"mean": 0.0, "count": 24},
                    "state_hallucination_rate": {"mean": 0.0, "count": 24},
                }
            },
        },
    )


def test_phase2l_localfull512_freeze_keeps_package_and_sealed_blocked(
    tmp_path: Path,
) -> None:
    prereg = _write(tmp_path / "prereg.json", {"status": "local_full512_preregistered"})
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "rollups": {"source_overlap": {"val": {"accuracy": 0.333333}}},
        },
    )
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})
    summary = _write(
        tmp_path / "summary.json",
        {
            "adapter_name": "phase2l_localfull512",
            "adapter_output_dir": "artifacts/adapters/phase2l_localfull512",
            "config_hash": "abc123",
            "effective_split_hashes": {"phase2c_head_train": "t", "phase2c_head_val": "v"},
            "train_examples": 512,
            "val_examples": 72,
            "history": [
                {
                    "train_elapsed_seconds": 4000.0,
                    "train_steps_per_second": 0.12,
                    "val_metrics": {"command_slot_accuracy": 0.416667},
                }
            ],
        },
    )
    full = _eval(tmp_path / "full.json", 1.0)
    native = _eval(tmp_path / "native.json", 0.416667, control="cache_erased")
    wrong = _eval(tmp_path / "wrong.json", 0.0, control="wrong_cache")
    postflight = _write(
        tmp_path / "postflight.json",
        {
            "passed": True,
            "ready_for_package": False,
            "ready_for_sealed_eval": False,
            "allowed_next_action": "run_phase2l_full1024_on_larger_gpu_or_repeat_local_full512_seed",
            "checks": {
                "full_beats_native_head_only_by_required_delta": True,
                "full_beats_wrong_cache_by_required_delta": True,
                "full_beats_cache_erased_by_required_delta": True,
                "full_low_level_qwen_calls_zero": True,
                "sealed_v3_not_used_for_postflight": True,
            },
            "metrics": {
                "cache_erased_completion": 0.416667,
                "full_minus_native_head_only": 0.583333,
                "full_minus_wrong_cache": 1.0,
                "full_minus_cache_erased": 0.583333,
                "full_trace_audit": {"low_level_qwen_calls": 0},
            },
        },
    )

    manifest = build_phase2l_localfull512_freeze_manifest(
        preregistration_json=prereg,
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
        training_summary_json=summary,
        full_eval_json=full,
        native_head_only_eval_json=native,
        wrong_cache_eval_json=wrong,
        postflight_json=postflight,
    )

    assert manifest["passed"] is True
    assert manifest["checks"]["localfull512_not_package_ready"] is True
    assert manifest["checks"]["localfull512_not_sealed_ready"] is True
    assert manifest["metrics"]["full_minus_cache_erased"] == 0.583333
    assert "do_not_package_from_localfull512" in manifest["blocked_actions"]
    assert "Phase2L is package-ready" in manifest["unsupported_claims"]


def test_phase2l_full_freeze_records_nonsealed_positive_and_sealed_failure(
    tmp_path: Path,
) -> None:
    nonsealed = tmp_path / "nonsealed"
    sealed = tmp_path / "sealed"
    packages = tmp_path / "packages"
    adapter = "phase2l_adapter"
    _write(
        nonsealed / "phase2l_full_data_health.json",
        {
            "passed": True,
            "rollups": {"source_overlap": {"val": {"accuracy": 0.333333}}},
        },
    )
    _write(nonsealed / "phase2l_full_pretrain_gate.json", {"passed": True})
    _write(
        nonsealed / "phase2l_full1024_checkpointed.training_summary.json",
        {"adapter_name": adapter, "config_hash": "abc123"},
    )
    _write(
        nonsealed / "phase2l_full1024_checkpointed_postflight.json",
        {"passed": True, "metrics": {"full_minus_native_head_only": 0.5}},
    )
    _write(
        nonsealed / "phase2l_full1024_checkpointed_package_postflight.json",
        {
            "passed": True,
            "ready_for_package": True,
            "metrics": {
                "full_completion": 1.0,
                "native_head_only_completion": 0.5,
                "wrong_cache_completion": 0.0,
                "cache_erased_completion": 0.5,
                "full_minus_native_head_only": 0.5,
                "full_minus_wrong_cache": 1.0,
                "full_minus_cache_erased": 0.5,
            },
        },
    )
    _eval(nonsealed / "phase2l_full1024_checkpointed_package_full_eval.json", 1.0)
    _eval(nonsealed / "phase2l_full1024_checkpointed_package_native_head_only_eval.json", 0.5)
    _eval(
        nonsealed / "phase2l_full1024_checkpointed_package_wrong_cache_eval.json",
        0.0,
        control="wrong_cache",
    )
    _write(
        sealed / "phase2l_sealed_v3_gate.json",
        {
            "passed": False,
            "claim_boundary": "bounded_claim_only_do_not_upgrade_continuation_memory_necessity",
            "checks": {
                "full_completion_gate_passed": False,
                "full_low_level_qwen_calls_zero": False,
                "sealed_v3_inputs_only": True,
                "allowlist_hallucination_zero": True,
            },
            "metrics": {
                "deltas": {
                    "full_minus_no_nsi": 0.0,
                    "full_minus_native_head_only": 0.0,
                    "full_minus_continuation_only": 0.0,
                }
            },
        },
    )
    (sealed / "phase2l_sealed_v3_baseline_table.md").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (sealed / "phase2l_sealed_v3_baseline_table.md").write_text("table", encoding="utf-8")
    for name in [
        "full",
        "no_nsi",
        "native_head_only",
        "continuation_only",
        "prompt_only",
        "react",
    ]:
        _eval(sealed / f"phase2l_{name}_sealed_v3_eval.json", 0.0)
    for suffix in ["", "_no_nsi_latent", "_native_head_only", "_continuation_only"]:
        _write(
            packages / f"{adapter}{suffix}" / "native_nervous_package.json",
            {"adapter_name": adapter},
        )

    manifest = build_phase2l_freeze_manifest(
        nonsealed_report_dir=nonsealed,
        sealed_report_dir=sealed,
        package_root=packages,
        adapter_name=adapter,
    )

    assert manifest["frozen"] is True
    assert manifest["checks"]["package_postflight_passed"] is True
    assert manifest["checks"]["sealed_gate_failed"] is True
    assert manifest["metrics"]["nonsealed_package"]["full_minus_wrong_cache"] == 1.0
    assert "Phase2L proves continuation memory necessity on sealed v3" in manifest[
        "unsupported_claims"
    ]
