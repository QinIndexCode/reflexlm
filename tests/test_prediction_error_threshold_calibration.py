import json
from dataclasses import asdict
from pathlib import Path

from reflexlm.cli.calibrate_nsi_prediction_error_threshold import (
    calibrate_nsi_prediction_error_threshold,
)
from reflexlm.models.features import StateVectorizer
from reflexlm.models.nsi_model import NSIModelConfig, NSIReflexModel
from reflexlm.llm.native_head_policy import (
    _resolve_prediction_error_escalation_threshold,
)
from reflexlm.train import load_model_checkpoint, save_model_checkpoint


def _checkpoint(tmp_path: Path) -> Path:
    vectorizer = StateVectorizer(hash_bins=0)
    model = NSIReflexModel(NSIModelConfig.smoke(vectorizer.vector_dim))
    summary = {
        "model_kind": "nsi",
        "model_config": asdict(model.config),
        "vectorizer": asdict(vectorizer),
        "training_summary": {},
    }
    return save_model_checkpoint(
        model,
        vectorizer,
        checkpoint_path=tmp_path / "model.pt",
        model_kind="nsi",
        summary=summary,
    )


def test_calibration_attaches_threshold_to_checkpoint_metadata(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    report_path = tmp_path / "calibration.json"
    report_path.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "rows": 200,
                    "recommended_prediction_error_threshold": 0.04,
                    "correct_action_next_state_rmse_p99": 0.032,
                    "prediction_error_rate_above_threshold": 0.0,
                },
                "inputs": {"dataset_path": "heldout.jsonl"},
            }
        ),
        encoding="utf-8",
    )

    result = calibrate_nsi_prediction_error_threshold(
        checkpoint_path=checkpoint,
        calibration_report_json=report_path,
        output_checkpoint_path=tmp_path / "calibrated.pt",
    )
    _model, _vectorizer, payload = load_model_checkpoint(tmp_path / "calibrated.pt")
    calibration = payload["training_summary"]["prediction_error_calibration"]

    assert result["passed"] is True
    assert calibration["threshold"] == 0.04
    assert calibration["calibration_false_positive_rate"] == 0.0


def test_calibration_rejects_failed_report(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    report_path = tmp_path / "failed.json"
    report_path.write_text(json.dumps({"passed": False}), encoding="utf-8")

    try:
        calibrate_nsi_prediction_error_threshold(
            checkpoint_path=checkpoint,
            calibration_report_json=report_path,
            output_checkpoint_path=tmp_path / "calibrated.pt",
        )
    except ValueError as error:
        assert "must pass" in str(error)
    else:
        raise AssertionError("failed calibration report should be rejected")


def test_native_head_threshold_prefers_override_then_checkpoint_calibration() -> None:
    payload = {
        "training_summary": {
            "prediction_error_calibration": {
                "threshold": 0.04,
            }
        }
    }

    calibrated = _resolve_prediction_error_escalation_threshold(None, payload)
    overridden = _resolve_prediction_error_escalation_threshold(0.2, payload)
    legacy = _resolve_prediction_error_escalation_threshold(None, {})

    assert calibrated == (0.04, "checkpoint_calibration")
    assert overridden == (0.2, "configured_override")
    assert legacy == (0.45, "legacy_default")
