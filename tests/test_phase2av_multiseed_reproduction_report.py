import json
from pathlib import Path

from reflexlm.cli.build_phase2av_multiseed_reproduction_report import (
    build_phase2av_multiseed_reproduction_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(tmp_path: Path, seed: int) -> Path:
    return _write(
        tmp_path / f"summary_{seed}.json",
        {
            "config_hash": f"hash-{seed}",
            "train_examples": 690,
            "val_examples": 155,
            "effective_split_hashes": {"phase2c_head_train": f"train-{seed}"},
        },
    )


def _postflight(
    tmp_path: Path,
    name: str,
    *,
    passed: bool = True,
    holdout_hash: str = "holdout-hash",
    unsupported: list[str] | None = None,
) -> Path:
    return _write(
        tmp_path / f"{name}.json",
        {
            "passed": passed,
            "metrics": {
                "command_slot_accuracy": 1.0,
                "model_minus_source_overlap_accuracy": 0.56,
                "patch_operation_accuracy": 0.98,
                "patch_template_slot_accuracy": 0.98,
                "eval_rows_hash": holdout_hash,
            },
            "unsupported_claims": unsupported
            or [
                "sealed_cross_model_transfer",
                "freeform_patch_generation",
                "open_ended_debugging_generalization",
                "production_autonomy",
                "epoch_making_architecture",
            ],
        },
    )


def _manifest(tmp_path: Path, *, divergent_hash: bool = False, boundary_ok: bool = True) -> Path:
    runs = []
    for seed in (13, 29):
        runs.append(
            {
                "seed": seed,
                "training_summary_json": str(_summary(tmp_path, seed)),
                "val_postflight_json": str(_postflight(tmp_path, f"val_{seed}")),
                "holdout_postflight_json": str(
                    _postflight(
                        tmp_path,
                        f"holdout_{seed}",
                        holdout_hash=f"holdout-hash-{seed}" if divergent_hash else "holdout-hash",
                        unsupported=["sealed_cross_model_transfer"] if not boundary_ok else None,
                    )
                ),
            }
        )
    return _write(
        tmp_path / "manifest.json",
        {"sealed_feedback_used": False, "runs": runs},
    )


def test_phase2av_multiseed_report_accepts_two_seed_nonsealed_reproduction(
    tmp_path: Path,
) -> None:
    report = build_phase2av_multiseed_reproduction_report(
        run_manifest_json=_manifest(tmp_path)
    )

    assert report["passed"] is True
    assert report["metrics"]["unique_seed_count"] == 2
    assert report["metrics"]["holdout_command_slot_accuracy_min"] == 1.0
    assert "do_not_run_sealed_eval_from_multiseed_report" in report["blocked_actions"]


def test_phase2av_multiseed_report_rejects_divergent_holdout_hash(tmp_path: Path) -> None:
    report = build_phase2av_multiseed_reproduction_report(
        run_manifest_json=_manifest(tmp_path, divergent_hash=True)
    )

    assert report["passed"] is False
    assert report["checks"]["shared_holdout_hash"] is False


def test_phase2av_multiseed_report_rejects_boundary_drift(tmp_path: Path) -> None:
    report = build_phase2av_multiseed_reproduction_report(
        run_manifest_json=_manifest(tmp_path, boundary_ok=False)
    )

    assert report["passed"] is False
    assert report["checks"]["all_seed_gates_passed"] is False
