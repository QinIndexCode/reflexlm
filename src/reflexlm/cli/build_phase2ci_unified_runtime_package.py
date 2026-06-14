from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from reflexlm.llm.native_nervous_package import PACKAGE_MANIFEST_NAME
from reflexlm.train import load_model_checkpoint


def build_phase2ci_unified_runtime_package(
    *,
    base_package_path: str | Path,
    structured_runtime_checkpoint_path: str | Path,
    output_package_dir: str | Path,
    output_report_json: str | Path,
    structured_runtime_python_identity: str | None = None,
    require_prediction_error_calibration: bool = False,
) -> dict[str, Any]:
    base_dir = Path(base_package_path)
    manifest_path = base_dir / PACKAGE_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"base package manifest missing: {manifest_path}")
    checkpoint_path = Path(structured_runtime_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"structured runtime checkpoint missing: {checkpoint_path}"
        )
    _, _, checkpoint_payload = load_model_checkpoint(checkpoint_path, device="cpu")
    environment_path = checkpoint_path.parent / "environment.json"
    python_identity = structured_runtime_python_identity
    if python_identity is None and environment_path.exists():
        environment = json.loads(environment_path.read_text(encoding="utf-8-sig"))
        python_identity = (
            environment.get("python", {}).get("executable")
            if isinstance(environment.get("python"), dict)
            else None
        )
    output_dir = Path(output_package_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in base_dir.iterdir():
        destination = output_dir / source.name
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        elif source.name != PACKAGE_MANIFEST_NAME:
            shutil.copy2(source, destination)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["policy_label"] = "phase2ci_unified_package_runtime_cortex"
    manifest["structured_runtime_cortex_checkpoint_path"] = str(checkpoint_path)
    manifest["structured_runtime_cortex_python_identity"] = python_identity
    training_summary = checkpoint_payload.get("training_summary", {})
    prediction_error_calibration = training_summary.get(
        "prediction_error_calibration"
    )
    manifest["structured_runtime_cortex_prediction_error_calibration"] = (
        prediction_error_calibration
        if isinstance(prediction_error_calibration, dict)
        else None
    )
    (output_dir / PACKAGE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    checks = {
        "base_package_has_verification_cortex": bool(
            manifest.get("verification_cortex_path")
        ),
        "structured_runtime_checkpoint_loads": bool(training_summary),
        "structured_runtime_cortex_packaged": bool(
            manifest.get("structured_runtime_cortex_checkpoint_path")
        ),
        "structured_runtime_python_identity_recorded": bool(python_identity),
    }
    if require_prediction_error_calibration:
        checks["structured_runtime_prediction_error_calibrated"] = isinstance(
            prediction_error_calibration,
            dict,
        ) and prediction_error_calibration.get("threshold") is not None
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ci_unified_runtime_package_build",
        "passed": passed,
        "ready_for_unified_package_structured_runtime_claim": passed,
        "checks": checks,
        "package_path": str(output_dir),
        "package_manifest_path": str(output_dir / PACKAGE_MANIFEST_NAME),
        "structured_runtime_cortex_checkpoint_path": str(checkpoint_path),
        "structured_runtime_cortex_python_identity": python_identity,
        "structured_runtime_prediction_error_calibration": prediction_error_calibration,
        "prediction_error_calibration_required": require_prediction_error_calibration,
        "structured_runtime_training_summary": training_summary,
        "supported_claims": [
            "the unified deployment package contains patch-selection, temporal verification, and structured runtime cortical experts"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2ci_unified_package_open_task_family_repo_runtime"
            if passed
            else "repair_phase2ci_unified_runtime_package_build"
        ),
    }
    report_path = Path(output_report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend a native nervous package with a structured runtime cortex."
    )
    parser.add_argument("--base-package-path", required=True)
    parser.add_argument("--structured-runtime-checkpoint-path", required=True)
    parser.add_argument("--output-package-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--structured-runtime-python-identity")
    parser.add_argument("--require-prediction-error-calibration", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2ci_unified_runtime_package(
        base_package_path=args.base_package_path,
        structured_runtime_checkpoint_path=args.structured_runtime_checkpoint_path,
        output_package_dir=args.output_package_dir,
        output_report_json=args.output_report_json,
        structured_runtime_python_identity=args.structured_runtime_python_identity,
        require_prediction_error_calibration=args.require_prediction_error_calibration,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
