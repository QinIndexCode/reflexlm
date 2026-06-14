import json
from pathlib import Path

from reflexlm.cli.freeze_phase2k_continuation_pressure import (
    DEFAULT_ADAPTER_NAME,
    build_phase2k_freeze_manifest,
    build_phase2l_preregistration,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _eval(path: Path, completion: float, *, model_calls: float = 0.0) -> None:
    positives = round(completion * 64)
    _write(
        path,
        {
            "episode_count": 64,
            "dataset": {
                "dataset_path": "artifacts/datasets/phase2i_external_trace_v3_semantic_required/challenge.jsonl"
            },
            "metrics": {
                "aggregate": {
                    "task_completion_rate": {
                        "mean": completion,
                        "positives": positives,
                        "count": 64,
                    },
                    "command_decision_accuracy": {"mean": completion, "count": 64},
                    "read_file_decision_accuracy": {"mean": 1.0, "count": 64},
                    "oracle_step_accuracy": {"mean": completion, "count": 64},
                    "model_calls": {"mean": model_calls, "count": 64},
                    "token_equivalent_cost": {"mean": 0.0, "count": 64},
                    "state_hallucination_rate": {"mean": 0.0, "count": 64},
                }
            },
            "run_path": str(path.parent / f"{path.stem}_run"),
        },
    )


def test_phase2k_freeze_records_nonsealed_positive_and_sealed_failure(
    tmp_path: Path,
) -> None:
    nonsealed = tmp_path / "nonsealed"
    sealed = tmp_path / "sealed"
    packages = tmp_path / "packages"
    _write(
        nonsealed / "phase2k_continuation_pressure_data_health.json",
        {
            "passed": True,
            "rollups": {"source_overlap": {"val": {"accuracy": 0.333333}}},
        },
    )
    _write(nonsealed / "phase2k_head_dataset_manifest.json", {"rows": 144})
    _write(
        nonsealed
        / "phase2k_continuation_pressure_r16_alpha32_lr1e-4_len256_smoke128_val144.training_summary.json",
        {"adapter_name": "smoke"},
    )
    _write(
        nonsealed / "phase2k_smoke_postflight.json",
        {"passed": True, "ready_for_package": False},
    )
    _write(nonsealed / f"{DEFAULT_ADAPTER_NAME}.training_summary.json", {"adapter_name": DEFAULT_ADAPTER_NAME})
    _write(
        nonsealed / "phase2k_full_postflight.json",
        {
            "passed": True,
            "metrics": {
                "full_completion": 1.0,
                "native_head_only_completion": 0.5,
                "full_minus_native_head_only": 0.5,
            },
        },
    )
    _write(
        sealed / "phase2k_sealed_v3_gate.json",
        {
            "passed": False,
            "checks": {
                "full_beats_no_nsi_by_required_delta": True,
                "full_beats_native_head_only_by_required_delta": False,
                "full_low_level_qwen_calls_zero": False,
                "allowlist_hallucination_zero": True,
            },
            "metrics": {
                "deltas": {
                    "full_minus_no_nsi": 0.28125,
                    "full_minus_native_head_only": 0.0,
                    "full_minus_continuation_only": 0.328125,
                }
            },
        },
    )
    _write(sealed / "phase2k_sealed_v3_baseline_table.md", {"table": "ok"})
    for name, completion, calls in [
        ("full", 0.328125, 1.0),
        ("no_nsi", 0.046875, 1.0),
        ("native_head_only", 0.328125, 2.0),
        ("continuation_only", 0.0, 0.0),
        ("prompt_only", 0.0, 1.0),
        ("react", 0.0, 2.0),
    ]:
        filename = f"phase2k_{name}_sealed_v3_eval.json"
        _eval(sealed / filename, completion, model_calls=calls)
    for suffix in ["", "_no_nsi_latent", "_native_head_only", "_continuation_only"]:
        _write(packages / f"{DEFAULT_ADAPTER_NAME}{suffix}" / "native_nervous_package.json", {"ok": True})

    manifest = build_phase2k_freeze_manifest(
        nonsealed_report_dir=nonsealed,
        sealed_report_dir=sealed,
        package_root=packages,
    )

    assert manifest["frozen"] is True
    assert manifest["checks"]["full_nonsealed_beats_native_head_only"] is True
    assert manifest["checks"]["sealed_full_does_not_beat_native_head_only"] is True
    assert manifest["checks"]["sealed_full_low_level_qwen_calls_not_zero"] is True
    assert manifest["metrics"]["nonsealed"]["source_overlap_val_baseline"] == 0.333333
    assert manifest["claim_boundary"].startswith("Freeze Phase2K")


def test_phase2l_preregistration_requires_counterfactual_controls() -> None:
    prereg = build_phase2l_preregistration()

    assert prereg["status"] == "preregistered_not_generated_not_trained"
    assert prereg["data_constraints"]["sealed_inputs_allowed"] is False
    assert "wrong_cache_baseline_measured" in prereg["predeclared_gates"]
    assert prereg["predeclared_gates"]["full_minus_wrong_cache_min"] == 0.25
    assert any("sealed" in item for item in prereg["stop_conditions"]) is False
