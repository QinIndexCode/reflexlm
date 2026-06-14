import json
from pathlib import Path

from reflexlm.cli.audit_phase2av_full_readiness import audit_phase2av_full_readiness


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest(
    tmp_path: Path,
    *,
    train: int = 300,
    val: int = 80,
    holdout: int = 80,
    operation_balanced: bool = True,
    operations: dict[str, int] | None = None,
) -> Path:
    operation_counts = operations or {
        "insert_import": 20,
        "replace_attribute": 20,
        "replace_literal": 20,
    }
    return _write(
        tmp_path / "manifest.json",
        {
            "passed": True,
            "operation_balanced": operation_balanced,
            "split_counts": {"train": train, "val": val, "holdout": holdout},
            "split_reports": {
                split: {"selected_operation_counts": operation_counts}
                for split in ("train", "val", "holdout")
            },
        },
    )


def _summary(tmp_path: Path, *, identity_bias: float = 4.0) -> Path:
    return _write(
        tmp_path / "training_summary.json",
        {
            "train_examples": 422,
            "open_repair_heads_enabled": True,
            "use_pairwise_command_reranker": False,
            "no_json_motor_target": True,
            "low_level_qwen_calls_target": 0,
            "command_identity_logit_bias": identity_bias,
        },
    )


def _postflight(
    tmp_path: Path,
    name: str,
    *,
    passed: bool = True,
    command_slot_accuracy: float = 0.95,
    delta: float = 0.25,
    operation_accuracy: float = 0.95,
    template_accuracy: float = 0.95,
) -> Path:
    return _write(
        tmp_path / f"{name}.json",
        {
            "passed": passed,
            "metrics": {
                "command_slot_accuracy": command_slot_accuracy,
                "model_minus_source_overlap_accuracy": delta,
                "patch_operation_accuracy": operation_accuracy,
                "patch_template_slot_accuracy": template_accuracy,
            },
            "unsupported_claims": [
                "freeform_patch_generation",
                "sealed_cross_model_transfer",
                "open_ended_debugging_generalization",
                "production_autonomy",
                "epoch_making_architecture",
            ],
        },
    )


def _data_health(tmp_path: Path, name: str = "data_health") -> Path:
    return _write(tmp_path / f"{name}.json", {"passed": True})


def test_phase2av_full_readiness_accepts_only_large_diverse_nonsealed_path(
    tmp_path: Path,
) -> None:
    report = audit_phase2av_full_readiness(
        subset_manifest_json=_manifest(tmp_path),
        training_summary_json=_summary(tmp_path),
        smoke_postflight_json=_postflight(tmp_path, "smoke"),
        holdout_postflight_json=_postflight(tmp_path, "holdout"),
        data_health_jsons=[_data_health(tmp_path)],
    )

    assert report["passed"] is True
    assert report["ready_to_start_phase2av_full_training"] is True
    assert report["ready_for_package"] is False
    assert "do_not_run_sealed_eval_for_phase2av" in report["blocked_actions"]
    assert "epoch_making_architecture" in report["unsupported_claims"]


def test_phase2av_full_readiness_accepts_repo_operation_disjoint_manifest(
    tmp_path: Path,
) -> None:
    manifest = _write(
        tmp_path / "repo_operation_disjoint_manifest.json",
        {
            "passed": True,
            "split_construction_family": "phase2av_repo_operation_disjoint_split",
            "split_counts": {"train": 144, "val": 155, "holdout": 156},
            "operation_counts": {
                "train": {
                    "insert_import": 57,
                    "replace_attribute": 34,
                    "replace_literal": 53,
                },
                "val": {
                    "insert_import": 87,
                    "replace_attribute": 8,
                    "replace_literal": 60,
                },
                "holdout": {
                    "insert_import": 86,
                    "replace_attribute": 6,
                    "replace_literal": 64,
                },
            },
        },
    )
    report = audit_phase2av_full_readiness(
        subset_manifest_json=manifest,
        training_summary_json=_summary(tmp_path),
        smoke_postflight_json=_postflight(tmp_path, "smoke"),
        holdout_postflight_json=_postflight(tmp_path, "holdout"),
        data_health_jsons=[_data_health(tmp_path)],
    )

    assert report["passed"] is True
    assert report["checks"]["subset_is_not_operation_balanced_shortcut"] is True
    assert report["checks"]["split_counts_sufficient"] is True
    assert report["metrics"]["effective_train_examples"] == 422
    assert report["metrics"]["split_counts"]["train"] == 144


def test_phase2av_full_readiness_rejects_small_imbalanced_green_smoke(
    tmp_path: Path,
) -> None:
    report = audit_phase2av_full_readiness(
        subset_manifest_json=_manifest(
            tmp_path,
            train=15,
            val=10,
            holdout=16,
            operation_balanced=False,
            operations={"insert_import": 12, "replace_attribute": 3},
        ),
        training_summary_json=_summary(tmp_path),
        smoke_postflight_json=_postflight(tmp_path, "smoke"),
        holdout_postflight_json=_postflight(tmp_path, "holdout"),
        data_health_jsons=[_data_health(tmp_path)],
    )

    assert report["passed"] is False
    assert report["ready_to_start_phase2av_full_training"] is False
    assert "do_not_start_phase2av_full_training" in report["blocked_actions"]
    assert "split_counts_sufficient" in report["blocking_reasons"]
    assert "operation_diversity_sufficient" in report["blocking_reasons"]
    assert "subset_is_not_operation_balanced_shortcut" in report["blocking_reasons"]


def test_phase2av_full_readiness_rejects_missing_identity_prior(
    tmp_path: Path,
) -> None:
    report = audit_phase2av_full_readiness(
        subset_manifest_json=_manifest(tmp_path),
        training_summary_json=_summary(tmp_path, identity_bias=0.0),
        smoke_postflight_json=_postflight(tmp_path, "smoke"),
        holdout_postflight_json=_postflight(tmp_path, "holdout"),
        data_health_jsons=[_data_health(tmp_path)],
    )

    assert report["passed"] is False
    assert report["checks"]["command_identity_prior_recorded"] is False
    assert "command_identity_prior_recorded" in report["blocking_reasons"]


def test_phase2av_full_readiness_rejects_claim_boundary_drift(tmp_path: Path) -> None:
    report = audit_phase2av_full_readiness(
        subset_manifest_json=_manifest(tmp_path),
        training_summary_json=_summary(tmp_path),
        smoke_postflight_json=_postflight(tmp_path, "smoke"),
        holdout_postflight_json=_write(
            tmp_path / "holdout.json",
            {
                "passed": True,
                "metrics": {
                    "command_slot_accuracy": 0.95,
                    "model_minus_source_overlap_accuracy": 0.25,
                    "patch_operation_accuracy": 0.95,
                    "patch_template_slot_accuracy": 0.95,
                },
                "unsupported_claims": ["sealed_cross_model_transfer"],
            },
        ),
        data_health_jsons=[_data_health(tmp_path)],
    )

    assert report["passed"] is False
    assert report["checks"]["claim_boundary_preserved"] is False
