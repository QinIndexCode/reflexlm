import json
from pathlib import Path

from reflexlm.cli.check_phase2d_gates import build_phase2d_gate_report
from reflexlm.llm.native_nervous_package import write_native_nervous_package


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _eval_payload(
    tmp_path: Path,
    name: str,
    *,
    completion: float = 1.0,
    task_failure: float = 1.0,
    zero_nsi_latent: bool = False,
) -> Path:
    run_path = tmp_path / name
    run_path.mkdir()
    (run_path / "trace_rows.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "episode_id": "blocking_input_detection-00000",
                        "task_type": "blocking_input_detection",
                        "policy_debug": {"action_source": "low_level_nsi"},
                        "oracle_action": {"type": "WAIT"},
                        "done": True,
                        "reward": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "episode_id": "test_failure_reflex-00000",
                        "task_type": "test_failure_reflex",
                        "policy_debug": {"action_source": "native_head_cortex"},
                        "oracle_action": {
                            "type": "RUN_COMMAND",
                            "command": "python -m pytest -q tests/test_snapshots.py --snapshot-update",
                        },
                        "done": True,
                        "reward": 1.0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    return _write(
        tmp_path / f"{name}.json",
        {
            "run_path": str(run_path),
            "policy": {
                "policy_family": "phase2d_native_nervous_package",
                "json_text_target": False,
                "zero_nsi_latent": zero_nsi_latent,
            },
            "metrics": {
                "aggregate": {
                    "task_completion_rate": {"mean": completion},
                    "dangerous_action_block_rate": {"mean": 1.0},
                    "stale_state_action_rate": {"mean": 0.0},
                    "state_hallucination_rate": {"mean": 0.0},
                    "model_calls": {"mean": 1.0},
                },
                "per_task": {
                    "blocking_input_detection": {"metrics": {"task_completion_rate": {"mean": 1.0}}},
                    "common_error_recovery_routine": {"metrics": {"task_completion_rate": {"mean": 1.0}}},
                    "dangerous_action_interception": {"metrics": {"task_completion_rate": {"mean": 1.0}}},
                    "external_file_change_reflex": {"metrics": {"task_completion_rate": {"mean": 1.0}}},
                    "process_hang_detection": {"metrics": {"task_completion_rate": {"mean": 1.0}}},
                    "test_failure_reflex": {
                        "metrics": {"task_completion_rate": {"mean": task_failure}}
                    },
                },
            },
        },
    )


def test_write_native_nervous_package_records_no_json_and_ablation_flag(tmp_path: Path) -> None:
    manifest = write_native_nervous_package(
        tmp_path / "pkg",
        base_model_name="model",
        native_head_path="adapter",
        low_level_checkpoint_path="nsi.pt",
        zero_nsi_latent=True,
    )

    assert manifest["package_family"] == "phase2d_native_nervous_package"
    assert manifest["json_text_target"] is False
    assert manifest["zero_nsi_latent"] is True
    assert manifest["continuation_cache_enabled"] is True
    assert manifest["continuation_control"] == "normal"
    assert manifest["native_head_calls_enabled"] is True
    assert manifest["disabled_command_candidate_feature_groups"] == ["candidate_identity"]
    saved = json.loads((tmp_path / "pkg" / "native_nervous_package.json").read_text())
    assert saved == manifest


def test_write_native_nervous_package_records_mechanism_ablation_flags(tmp_path: Path) -> None:
    manifest = write_native_nervous_package(
        tmp_path / "pkg",
        base_model_name="model",
        native_head_path="adapter",
        low_level_checkpoint_path="nsi.pt",
        continuation_cache_enabled=False,
        continuation_control="cache_erased",
        native_head_calls_enabled=False,
        disabled_command_candidate_feature_groups=[
            "candidate_identity",
            "slot_position",
        ],
    )

    assert manifest["continuation_cache_enabled"] is False
    assert manifest["debug_continuation_cache"] is False
    assert manifest["continuation_control"] == "cache_erased"
    assert manifest["native_head_calls_enabled"] is False
    assert manifest["disabled_command_candidate_feature_groups"] == [
        "candidate_identity",
        "slot_position",
    ]


def test_write_native_nervous_package_records_reproducible_low_memory_loading(
    tmp_path: Path,
) -> None:
    manifest = write_native_nervous_package(
        tmp_path / "pkg",
        base_model_name="model",
        native_head_path="adapter",
        low_level_checkpoint_path="nsi.pt",
        model_load_strategy="single_device",
        offload_state_dict=True,
    )

    assert manifest["model_load_strategy"] == "single_device"
    assert manifest["offload_state_dict"] is True


def test_write_native_nervous_package_records_packaged_verification_cortex(
    tmp_path: Path,
) -> None:
    manifest = write_native_nervous_package(
        tmp_path / "pkg",
        base_model_name="model",
        native_head_path="adapter",
        low_level_checkpoint_path="nsi.pt",
        verification_cortex_path="verification.pt",
        verification_cortex_model_name="cortex-model",
        verification_cortex_recency_decay=0.5,
    )

    assert manifest["verification_cortex_path"] == "verification.pt"
    assert manifest["verification_cortex_model_name"] == "cortex-model"
    assert manifest["verification_cortex_recency_decay"] == 0.5


def test_phase2d_gate_distinguishes_strong_and_acceptable_pass(tmp_path: Path) -> None:
    fixed = _eval_payload(tmp_path, "fixed", completion=0.96, task_failure=0.91)
    debug = _eval_payload(tmp_path, "debug", completion=0.92, task_failure=1.0)
    quasi = _eval_payload(tmp_path, "quasi", completion=0.80, task_failure=1.0)
    prompt = _write(
        tmp_path / "prompt.json",
        {"metrics": {"aggregate": {"task_completion_rate": {"mean": 0.40}, "model_calls": {"mean": 3.0}}}},
    )
    react = _write(
        tmp_path / "react.json",
        {"metrics": {"aggregate": {"task_completion_rate": {"mean": 0.45}, "model_calls": {"mean": 4.0}}}},
    )
    no_nsi = _eval_payload(
        tmp_path,
        "no_nsi",
        completion=0.70,
        task_failure=0.70,
        zero_nsi_latent=True,
    )
    config = _write(tmp_path / "config.json", {"config_hash": "abc123"})

    report = build_phase2d_gate_report(
        fixed_eval_json=fixed,
        debug_ood_eval_json=debug,
        quasi_real_eval_json=quasi,
        prompt_quasi_eval_json=prompt,
        react_quasi_eval_json=react,
        no_nsi_latent_eval_json=no_nsi,
        config_json=config,
    )

    assert report["strong_pass"] is False
    assert report["acceptable_positive"] is True
    assert report["checks"]["no_nsi_latent_ablation_present"] is True


def test_phase2f_gate_can_require_latent_sensitive_ablation_delta(tmp_path: Path) -> None:
    fixed = _eval_payload(tmp_path, "fixed", completion=1.0, task_failure=1.0)
    debug = _eval_payload(tmp_path, "debug", completion=1.0, task_failure=1.0)
    quasi = _eval_payload(tmp_path, "quasi", completion=1.0, task_failure=1.0)
    latent = _eval_payload(tmp_path, "latent", completion=0.95, task_failure=1.0)
    prompt = _write(
        tmp_path / "prompt.json",
        {"metrics": {"aggregate": {"task_completion_rate": {"mean": 0.40}, "model_calls": {"mean": 3.0}}}},
    )
    react = _write(
        tmp_path / "react.json",
        {"metrics": {"aggregate": {"task_completion_rate": {"mean": 0.45}, "model_calls": {"mean": 4.0}}}},
    )
    no_nsi = _eval_payload(tmp_path, "no_nsi", completion=0.70, zero_nsi_latent=True)
    latent_no_nsi = _eval_payload(
        tmp_path,
        "latent_no_nsi",
        completion=0.75,
        zero_nsi_latent=True,
    )
    config = _write(tmp_path / "config.json", {"config_hash": "abc123"})

    report = build_phase2d_gate_report(
        fixed_eval_json=fixed,
        debug_ood_eval_json=debug,
        quasi_real_eval_json=quasi,
        prompt_quasi_eval_json=prompt,
        react_quasi_eval_json=react,
        no_nsi_latent_eval_json=no_nsi,
        latent_sensitive_eval_json=latent,
        latent_sensitive_no_nsi_eval_json=latent_no_nsi,
        config_json=config,
    )

    assert report["checks"]["latent_sensitive_ablation_delta"] is True
    assert report["metrics"]["latent_sensitive_delta"] == 0.19999999999999996
