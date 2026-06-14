import json
from pathlib import Path

from reflexlm.cli.audit_phase2ar_reproduction import audit_phase2ar_reproduction


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_phase2ar_reproduction_audit_rejects_failed_cross_package_run(tmp_path: Path) -> None:
    rows = [
        {
            "success": False,
            "patch_source": "package_runtime_no_patch_authorized",
            "policy_open_repair_outputs": {"rollback_safety": 0},
        }
        for _ in range(25)
    ] + [
        {
            "success": True,
            "patch_source": "package_runtime_symbolic_structural_patch_proposal",
            "policy_open_repair_outputs": {"rollback_safety": 1},
        }
        for _ in range(7)
    ]

    report = audit_phase2ar_reproduction(
        primary_summary_json=_write(
            tmp_path / "primary.json",
            {
                "rows": 32,
                "success_rate": 1.0,
                "patch_mode": "runtime_symbolic_structural",
                "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
            },
        ),
        reproduction_summary_json=_write(
            tmp_path / "repro.json",
            {
                "rows": 32,
                "success_rate": 0.21875,
                "patch_mode": "runtime_symbolic_structural",
                "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
            },
        ),
        reproduction_results_jsonl=_write_jsonl(tmp_path / "repro.jsonl", rows),
        min_reproduction_success_rate=1.0,
    )

    assert report["passed"] is False
    assert report["checks"]["reproduction_success_rate_met"] is False
    assert report["metrics"]["failure_reasons"]["rollback_safety_head_not_authorized"] == 25
    assert "cross_model_reproduction" in report["unsupported_claims"]


def test_phase2ar_reproduction_audit_accepts_matching_successful_run(tmp_path: Path) -> None:
    rows = [
        {
            "success": True,
            "patch_source": "package_runtime_symbolic_structural_patch_proposal",
            "policy_open_repair_outputs": {"rollback_safety": 1},
        }
        for _ in range(32)
    ]

    report = audit_phase2ar_reproduction(
        primary_summary_json=_write(
            tmp_path / "primary.json",
            {
                "rows": 32,
                "success_rate": 1.0,
                "patch_mode": "runtime_symbolic_structural",
                "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
            },
        ),
        reproduction_summary_json=_write(
            tmp_path / "repro.json",
            {
                "rows": 32,
                "success_rate": 1.0,
                "patch_mode": "runtime_symbolic_structural",
                "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
            },
        ),
        reproduction_results_jsonl=_write_jsonl(tmp_path / "repro.jsonl", rows),
        min_reproduction_success_rate=1.0,
    )

    assert report["passed"] is True
    assert "phase2ar_cross_package_reproduction_supported" in report["supported_claims"]


def test_phase2ar_reproduction_audit_marks_two_seed_smoke_when_contract_matches(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "success": True,
            "patch_source": "package_runtime_symbolic_structural_patch_proposal",
            "policy_open_repair_outputs": {"rollback_safety": 1},
        }
        for _ in range(32)
    ]
    base_training = {
        "effective_split_hashes": {"phase2c_head_train": "a", "phase2c_head_val": "b"},
        "config": {
            "base_model_name": "model",
            "quantization": "4bit",
            "learning_rate": 0.0001,
            "epochs": 5,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_length": 256,
            "lora_rank": 16,
            "lora_alpha": 32,
            "command_candidate_encoder": "features_only",
            "latent_fusion": "additive",
            "open_repair_heads_enabled": True,
            "seed": 13,
        },
    }
    repro_training = json.loads(json.dumps(base_training))
    repro_training["config"]["seed"] = 17

    report = audit_phase2ar_reproduction(
        primary_summary_json=_write(
            tmp_path / "primary.json",
            {
                "rows": 32,
                "success_rate": 1.0,
                "patch_mode": "runtime_symbolic_structural",
                "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
            },
        ),
        reproduction_summary_json=_write(
            tmp_path / "repro.json",
            {
                "rows": 32,
                "success_rate": 1.0,
                "patch_mode": "runtime_symbolic_structural",
                "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
            },
        ),
        reproduction_results_jsonl=_write_jsonl(tmp_path / "repro.jsonl", rows),
        primary_training_summary_json=_write(tmp_path / "primary_training.json", base_training),
        reproduction_training_summary_json=_write(
            tmp_path / "repro_training.json", repro_training
        ),
    )

    assert report["passed"] is True
    assert report["metrics"]["training_contract"]["seed_changed"] is True
    assert report["checks"]["training_contract_matches_except_seed_and_names"] is True
    assert "phase2ar_two_seed_reproduction_smoke_supported" in report["supported_claims"]
    assert "multi_seed_reproduction_3plus" in report["unsupported_claims"]
