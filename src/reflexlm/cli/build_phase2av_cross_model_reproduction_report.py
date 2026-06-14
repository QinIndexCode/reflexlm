from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric(report: dict[str, Any], name: str) -> float | None:
    value = _dict(report.get("metrics")).get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    summary = _read_json(run["training_summary_json"])
    val = _read_json(run["val_postflight_json"])
    holdout = _read_json(run["holdout_postflight_json"])
    return {
        "model": str(run.get("model") or summary.get("base_model_name")),
        "seed": int(run.get("seed")),
        "training_summary_json": str(Path(run["training_summary_json"])),
        "val_postflight_json": str(Path(run["val_postflight_json"])),
        "holdout_postflight_json": str(Path(run["holdout_postflight_json"])),
        "config_hash": summary.get("config_hash"),
        "train_examples": summary.get("train_examples"),
        "val_passed": val.get("passed") is True,
        "holdout_passed": holdout.get("passed") is True,
        "holdout_eval_rows_hash": _dict(holdout.get("metrics")).get("eval_rows_hash"),
        "holdout_command_slot_accuracy": _metric(holdout, "command_slot_accuracy"),
        "holdout_model_minus_source_overlap_accuracy": _metric(
            holdout, "model_minus_source_overlap_accuracy"
        ),
        "holdout_patch_operation_accuracy": _metric(holdout, "patch_operation_accuracy"),
        "holdout_patch_template_slot_accuracy": _metric(
            holdout, "patch_template_slot_accuracy"
        ),
        "unsupported_claims": holdout.get("unsupported_claims", []),
    }


def build_phase2av_cross_model_reproduction_report(
    *,
    run_manifest_json: str | Path,
    min_model_count: int = 2,
    min_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_descriptor_accuracy: float = 0.85,
) -> dict[str, Any]:
    manifest = _read_json(run_manifest_json)
    runs = manifest.get("runs") if isinstance(manifest.get("runs"), list) else []
    reports = [_summarize_run(run) for run in runs if isinstance(run, dict)]
    models = {report["model"] for report in reports}
    holdout_hashes = {
        str(report["holdout_eval_rows_hash"])
        for report in reports
        if report.get("holdout_eval_rows_hash")
    }
    required_unsupported = {
        "sealed_cross_model_transfer",
        "freeform_patch_generation",
        "open_ended_debugging_generalization",
        "production_autonomy",
        "epoch_making_architecture",
    }
    per_model_gates = []
    for report in reports:
        per_model_gates.append(
            report["val_passed"]
            and report["holdout_passed"]
            and isinstance(report["holdout_command_slot_accuracy"], float)
            and report["holdout_command_slot_accuracy"] >= min_command_slot_accuracy
            and isinstance(report["holdout_model_minus_source_overlap_accuracy"], float)
            and report["holdout_model_minus_source_overlap_accuracy"]
            >= min_model_minus_source_overlap
            and isinstance(report["holdout_patch_operation_accuracy"], float)
            and report["holdout_patch_operation_accuracy"] >= min_descriptor_accuracy
            and isinstance(report["holdout_patch_template_slot_accuracy"], float)
            and report["holdout_patch_template_slot_accuracy"] >= min_descriptor_accuracy
            and required_unsupported.issubset(
                {str(claim) for claim in report["unsupported_claims"]}
            )
        )
    commands = [
        report["holdout_command_slot_accuracy"]
        for report in reports
        if isinstance(report["holdout_command_slot_accuracy"], float)
    ]
    deltas = [
        report["holdout_model_minus_source_overlap_accuracy"]
        for report in reports
        if isinstance(report["holdout_model_minus_source_overlap_accuracy"], float)
    ]
    checks = {
        "manifest_present": bool(manifest),
        "model_count_sufficient": len(models) >= min_model_count,
        "all_model_gates_passed": bool(per_model_gates) and all(per_model_gates),
        "shared_holdout_hash": len(holdout_hashes) == 1,
        "sealed_feedback_not_used": manifest.get("sealed_feedback_used") is False,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2av_cross_model_reproduction_report",
        "passed": passed,
        "claim_scope": (
            "phase2av_bounded_nonsealed_cross_model_descriptor_runtime_reproduction"
            if passed
            else "phase2av_cross_model_reproduction_incomplete"
        ),
        "claim_boundary": (
            "This report supports only non-sealed cross-model reproduction of "
            "bounded descriptor-runtime candidate selection. It does not prove "
            "sealed cross-model transfer, freeform patch generation, production "
            "autonomy, open-ended debugging generalization, or an epoch-making "
            "architecture."
        ),
        "checks": checks,
        "metrics": {
            "model_count": len(models),
            "models": sorted(models),
            "holdout_eval_rows_hashes": sorted(holdout_hashes),
            "holdout_command_slot_accuracy_mean": mean(commands) if commands else None,
            "holdout_command_slot_accuracy_min": min(commands) if commands else None,
            "holdout_model_minus_source_overlap_mean": mean(deltas) if deltas else None,
            "holdout_model_minus_source_overlap_min": min(deltas) if deltas else None,
            "runs": reports,
            "thresholds": {
                "min_model_count": min_model_count,
                "min_command_slot_accuracy": min_command_slot_accuracy,
                "min_model_minus_source_overlap": min_model_minus_source_overlap,
                "min_descriptor_accuracy": min_descriptor_accuracy,
            },
        },
        "supported_claims": [
            "phase2av_nonsealed_cross_model_descriptor_runtime_selection_reproduced"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "sealed_cross_model_transfer",
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": [
            "do_not_package_phase2av_from_cross_model_report",
            "do_not_run_sealed_eval_from_cross_model_report",
            "do_not_claim_sealed_cross_model_transfer",
            "do_not_claim_epoch_making_architecture",
        ],
        "next_required_evidence": [
            "runtime_patch_execution_delta_beyond_descriptor_selection",
            "package_gate_after_runtime_delta_not_after_cross_model_descriptor_selection_alone",
        ],
        "inputs": {"run_manifest_json": str(Path(run_manifest_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AV non-sealed cross-model reproduction report."
    )
    parser.add_argument("--run-manifest-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-model-count", type=int, default=2)
    parser.add_argument("--min-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-descriptor-accuracy", type=float, default=0.85)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2av_cross_model_reproduction_report(
        run_manifest_json=args.run_manifest_json,
        min_model_count=args.min_model_count,
        min_command_slot_accuracy=args.min_command_slot_accuracy,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_descriptor_accuracy=args.min_descriptor_accuracy,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
