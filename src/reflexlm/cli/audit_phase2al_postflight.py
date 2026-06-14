from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric_accuracy(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if not isinstance(value, dict):
        return None
    accuracy = value.get("accuracy")
    return float(accuracy) if isinstance(accuracy, (int, float)) else None


def _model_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    eval_metrics = summary.get("eval_metrics")
    if isinstance(eval_metrics, dict):
        return eval_metrics
    history = summary.get("history")
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict) and isinstance(latest.get("val_metrics"), dict):
            return latest["val_metrics"]
    return {}


def _source_overlap(summary: dict[str, Any], split: str) -> float | None:
    payload = _dict(_dict(summary.get("source_overlap_command_slot_baseline")).get(split))
    accuracy = payload.get("accuracy")
    return float(accuracy) if isinstance(accuracy, (int, float)) else None


def audit_phase2al_postflight(
    *,
    pretrain_gate_json: str | Path,
    summary_json: str | Path,
    split: str,
    output_json: str | Path | None = None,
    min_accuracy: float = 0.85,
    min_rows: int = 16,
    min_model_minus_source_overlap: float = 0.15,
    min_source_overlap_accuracy: float = 0.20,
    max_source_overlap_accuracy: float = 0.75,
    min_runtime_identity_accuracy: float = 0.95,
    max_model_runtime_identity_gap: float = 0.05,
) -> dict[str, Any]:
    pretrain = _read_json(pretrain_gate_json)
    summary = _read_json(summary_json)
    model_metrics = _model_metrics(summary)
    accuracy = model_metrics.get("command_slot_accuracy")
    rows = model_metrics.get("command_slot_count")
    model_accuracy = float(accuracy) if isinstance(accuracy, (int, float)) else None
    row_count = int(rows) if isinstance(rows, (int, float)) else 0
    source_overlap = _source_overlap(summary, split)
    pretrain_metrics = _dict(_dict(pretrain.get("split_metrics")).get(split))
    pretrain_source = _metric_accuracy(pretrain_metrics, "identity_text_ablated_source_overlap")
    runtime_identity = _metric_accuracy(pretrain_metrics, "runtime_identity_heuristic")
    if source_overlap is None:
        source_overlap = pretrain_source
    model_minus_source = (
        model_accuracy - source_overlap
        if model_accuracy is not None and source_overlap is not None
        else None
    )
    identity_gap = (
        runtime_identity - model_accuracy
        if runtime_identity is not None and model_accuracy is not None
        else None
    )
    requested_device = _dict(summary.get("config")).get("device")
    actual_device = str(summary.get("device") or requested_device or "")
    checks = {
        "pretrain_passed": pretrain.get("passed") is True,
        "controlled_pressure_only": pretrain.get("claim_bearing_natural_trace_evidence") is False,
        "model_accuracy_present": isinstance(model_accuracy, float),
        "model_accuracy_min": isinstance(model_accuracy, float) and model_accuracy >= min_accuracy,
        "row_count_min": row_count >= min_rows,
        "source_overlap_present": isinstance(source_overlap, float),
        "source_overlap_matches_pretrain": isinstance(source_overlap, float)
        and isinstance(pretrain_source, float)
        and abs(source_overlap - pretrain_source) <= 1e-9,
        "source_overlap_nonzero": isinstance(source_overlap, float)
        and source_overlap >= min_source_overlap_accuracy,
        "source_overlap_not_ceiling": isinstance(source_overlap, float)
        and source_overlap <= max_source_overlap_accuracy,
        "runtime_identity_reference_high": isinstance(runtime_identity, float)
        and runtime_identity >= min_runtime_identity_accuracy,
        "model_beats_source_overlap": isinstance(model_minus_source, float)
        and model_minus_source >= min_model_minus_source_overlap,
        "model_near_runtime_identity_upper_bound": isinstance(identity_gap, float)
        and identity_gap <= max_model_runtime_identity_gap,
        "evaluated_or_trained_on_cuda": actual_device.startswith("cuda")
        or requested_device == "cuda",
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "pairwise_disabled": summary.get("use_pairwise_command_reranker") is False,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2al_postflight",
        "passed": passed,
        "split": split,
        "checks": checks,
        "metrics": {
            "model_command_slot_accuracy": model_accuracy,
            "row_count": row_count,
            "source_overlap_accuracy": source_overlap,
            "pretrain_source_overlap_accuracy": pretrain_source,
            "runtime_identity_heuristic_accuracy": runtime_identity,
            "model_minus_source_overlap_accuracy": model_minus_source,
            "runtime_identity_minus_model_accuracy": identity_gap,
        },
        "thresholds": {
            "min_accuracy": min_accuracy,
            "min_rows": min_rows,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_source_overlap_accuracy": min_source_overlap_accuracy,
            "max_source_overlap_accuracy": max_source_overlap_accuracy,
            "min_runtime_identity_accuracy": min_runtime_identity_accuracy,
            "max_model_runtime_identity_gap": max_model_runtime_identity_gap,
        },
        "blocked_actions": (
            []
            if passed
            else [
                "do_not_upgrade_claim_from_phase2al",
                "do_not_package_phase2al_for_sealed_eval",
            ]
        ),
        "claim_boundary": (
            "Passing Phase2AL supports a controlled non-sealed structural-identity "
            "pressure result where the model beats a non-ceiling source-overlap baseline. "
            "It does not establish natural trace distribution, sealed transfer, production "
            "autonomy, open-ended debugging generalization, or an epoch-making architecture."
        ),
    }
    if output_json:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AL controlled-pressure postflight.")
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-json")
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-rows", type=int, default=16)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-source-overlap-accuracy", type=float, default=0.20)
    parser.add_argument("--max-source-overlap-accuracy", type=float, default=0.75)
    parser.add_argument("--min-runtime-identity-accuracy", type=float, default=0.95)
    parser.add_argument("--max-model-runtime-identity-gap", type=float, default=0.05)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = audit_phase2al_postflight(
        pretrain_gate_json=args.pretrain_gate_json,
        summary_json=args.summary_json,
        split=args.split,
        output_json=args.output_json,
        min_accuracy=args.min_accuracy,
        min_rows=args.min_rows,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_source_overlap_accuracy=args.min_source_overlap_accuracy,
        max_source_overlap_accuracy=args.max_source_overlap_accuracy,
        min_runtime_identity_accuracy=args.min_runtime_identity_accuracy,
        max_model_runtime_identity_gap=args.max_model_runtime_identity_gap,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
