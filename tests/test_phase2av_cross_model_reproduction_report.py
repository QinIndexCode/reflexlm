import json
from pathlib import Path

from reflexlm.cli.build_phase2av_cross_model_reproduction_report import (
    build_phase2av_cross_model_reproduction_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(tmp_path: Path, model: str) -> Path:
    return _write(
        tmp_path / f"{model.replace('/', '_')}.summary.json",
        {"base_model_name": model, "config_hash": model, "train_examples": 690},
    )


def _postflight(tmp_path: Path, name: str, *, boundary_ok: bool = True) -> Path:
    unsupported = [
        "sealed_cross_model_transfer",
        "freeform_patch_generation",
        "open_ended_debugging_generalization",
        "production_autonomy",
        "epoch_making_architecture",
    ]
    if not boundary_ok:
        unsupported = ["sealed_cross_model_transfer"]
    return _write(
        tmp_path / f"{name}.json",
        {
            "passed": True,
            "metrics": {
                "command_slot_accuracy": 1.0,
                "model_minus_source_overlap_accuracy": 0.56,
                "patch_operation_accuracy": 0.98,
                "patch_template_slot_accuracy": 0.98,
                "eval_rows_hash": "holdout-hash",
            },
            "unsupported_claims": unsupported,
        },
    )


def _manifest(tmp_path: Path, *, one_model: bool = False, boundary_ok: bool = True) -> Path:
    models = ["Qwen/Qwen2.5-7B-Instruct"]
    if not one_model:
        models.append("Qwen/Qwen2.5-3B-Instruct")
    runs = [
        {
            "model": model,
            "seed": 13,
            "training_summary_json": str(_summary(tmp_path, model)),
            "val_postflight_json": str(_postflight(tmp_path, f"val_{index}")),
            "holdout_postflight_json": str(
                _postflight(tmp_path, f"holdout_{index}", boundary_ok=boundary_ok)
            ),
        }
        for index, model in enumerate(models)
    ]
    return _write(
        tmp_path / "manifest.json",
        {"sealed_feedback_used": False, "runs": runs},
    )


def test_phase2av_cross_model_report_accepts_two_nonsealed_models(tmp_path: Path) -> None:
    report = build_phase2av_cross_model_reproduction_report(
        run_manifest_json=_manifest(tmp_path)
    )

    assert report["passed"] is True
    assert report["metrics"]["model_count"] == 2
    assert "do_not_claim_sealed_cross_model_transfer" in report["blocked_actions"]


def test_phase2av_cross_model_report_rejects_single_model(tmp_path: Path) -> None:
    report = build_phase2av_cross_model_reproduction_report(
        run_manifest_json=_manifest(tmp_path, one_model=True)
    )

    assert report["passed"] is False
    assert report["checks"]["model_count_sufficient"] is False


def test_phase2av_cross_model_report_rejects_boundary_drift(tmp_path: Path) -> None:
    report = build_phase2av_cross_model_reproduction_report(
        run_manifest_json=_manifest(tmp_path, boundary_ok=False)
    )

    assert report["passed"] is False
    assert report["checks"]["all_model_gates_passed"] is False
