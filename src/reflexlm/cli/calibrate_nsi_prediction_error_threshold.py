from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def calibrate_nsi_prediction_error_threshold(
    *,
    checkpoint_path: str | Path,
    calibration_report_json: str | Path,
    output_checkpoint_path: str | Path,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    report_path = Path(calibration_report_json)
    output = Path(output_checkpoint_path)
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("passed") is not True:
        raise ValueError("calibration report must pass before threshold calibration")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("calibration report is missing metrics")
    threshold = float(metrics["recommended_prediction_error_threshold"])
    if not 0.0 < threshold <= 1.0:
        raise ValueError("recommended prediction-error threshold must be in (0, 1]")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_kind") != "nsi":
        raise ValueError("prediction-error calibration requires an NSI checkpoint")
    training_summary = payload.get("training_summary")
    if not isinstance(training_summary, dict):
        training_summary = {}
        payload["training_summary"] = training_summary
    calibration = {
        "schema_version": "reflexlm.prediction_error_calibration.v1",
        "threshold": threshold,
        "method": "heldout_correct_action_next_state_rmse_p99_times_1_25",
        "calibration_rows": int(metrics.get("rows", 0)),
        "calibration_rmse_p99": float(metrics["correct_action_next_state_rmse_p99"]),
        "calibration_false_positive_rate": float(
            metrics["prediction_error_rate_above_threshold"]
        ),
        "calibration_dataset_path": str(report.get("inputs", {}).get("dataset_path", "")),
        "calibration_report_json": str(report_path),
    }
    training_summary["prediction_error_calibration"] = calibration
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "artifact_family": "nsi_prediction_error_threshold_calibration",
        "passed": True,
        "input_checkpoint_path": str(checkpoint),
        "output_checkpoint_path": str(output),
        "calibration": calibration,
    }


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach held-out prediction-error calibration to an NSI checkpoint."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--calibration-report-json", required=True)
    parser.add_argument("--output-checkpoint-path", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    result = calibrate_nsi_prediction_error_threshold(
        checkpoint_path=args.checkpoint_path,
        calibration_report_json=args.calibration_report_json,
        output_checkpoint_path=args.output_checkpoint_path,
    )
    _write(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
