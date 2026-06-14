from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _eval_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = _dict(summary.get("eval_metrics"))
    if metrics:
        return metrics
    history = summary.get("history")
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict):
            return _dict(latest.get("val_metrics"))
    return {}


def _metric_accuracy(pretrain: dict[str, Any], split: str, metric: str) -> float | None:
    payload = _dict(_dict(_dict(pretrain.get("split_metrics")).get(split)).get(metric))
    return _number(payload.get("accuracy"))


def _baseline_accuracy(summary: dict[str, Any], split: str) -> float | None:
    payload = _dict(_dict(summary.get("source_overlap_command_slot_baseline")).get(split))
    return _number(payload.get("accuracy"))


def _encoded_candidates(summary: dict[str, Any], split: str) -> float | None:
    payload = _dict(_dict(summary.get("pairwise_candidate_encoding")).get(split))
    return _number(payload.get("pairwise_scored_candidates"))


def build_phase2af_holdout_postflight(
    *,
    pretrain_gate_json: str | Path,
    eval_summary_json: str | Path,
    split: str = "holdout",
    adapter_dir: str | Path | None = None,
    min_accuracy: float = 0.85,
    min_rows: int = 64,
    min_model_minus_source_overlap: float = 0.15,
    min_model_minus_runtime_identity: float = 0.10,
    min_source_overlap_accuracy: float = 0.05,
    max_source_overlap_accuracy: float = 0.75,
    max_runtime_identity_accuracy: float = 0.90,
) -> dict[str, Any]:
    pretrain = _read_json(pretrain_gate_json)
    summary = _read_json(eval_summary_json)
    metrics = _eval_metrics(summary)

    accuracy = _number(metrics.get("command_slot_accuracy"))
    count = _number(metrics.get("command_slot_count"))
    source_overlap = _baseline_accuracy(summary, split)
    pretrain_source_overlap = _metric_accuracy(
        pretrain, split, "identity_text_ablated_source_overlap"
    )
    runtime_identity = _metric_accuracy(pretrain, split, "runtime_identity_heuristic")
    raw_source = _metric_accuracy(pretrain, split, "raw_source_overlap")
    model_minus_source = (
        accuracy - source_overlap
        if isinstance(accuracy, float) and isinstance(source_overlap, float)
        else None
    )
    model_minus_identity = (
        accuracy - runtime_identity
        if isinstance(accuracy, float) and isinstance(runtime_identity, float)
        else None
    )
    adapter_path = Path(adapter_dir or summary.get("adapter_output_dir") or "")
    adapter_files_present = (
        bool(str(adapter_path))
        and (adapter_path / "native_heads.pt").exists()
        and (adapter_path / "head_config.json").exists()
    )
    effective_hashes = _dict(summary.get("effective_split_hashes"))
    checks = {
        "pretrain_gate_passed": pretrain.get("passed") is True,
        "eval_split_matches": summary.get("eval_split") == split,
        "command_slot_count_min": isinstance(count, float) and count >= min_rows,
        "command_slot_accuracy_min": isinstance(accuracy, float) and accuracy >= min_accuracy,
        "source_overlap_baseline_present": isinstance(source_overlap, float),
        "source_overlap_matches_pretrain_gate": isinstance(source_overlap, float)
        and isinstance(pretrain_source_overlap, float)
        and abs(source_overlap - pretrain_source_overlap) <= 1e-9,
        "source_overlap_nonzero": isinstance(source_overlap, float)
        and source_overlap >= min_source_overlap_accuracy,
        "source_overlap_not_ceiling": isinstance(source_overlap, float)
        and source_overlap <= max_source_overlap_accuracy,
        "raw_source_not_ceiling": isinstance(raw_source, float)
        and raw_source <= max_source_overlap_accuracy,
        "runtime_identity_not_sufficient": isinstance(runtime_identity, float)
        and runtime_identity <= max_runtime_identity_accuracy,
        "model_beats_source_overlap": isinstance(model_minus_source, float)
        and model_minus_source >= min_model_minus_source_overlap,
        "model_beats_runtime_identity": isinstance(model_minus_identity, float)
        and model_minus_identity + 1e-12 >= min_model_minus_runtime_identity,
        "pairwise_disabled": summary.get("use_pairwise_command_reranker") is False,
        "pairwise_encoded_candidates_zero": _encoded_candidates(summary, split) == 0.0,
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "effective_split_hash_present": bool(effective_hashes.get(f"phase2c_head_{split}")),
        "evaluated_on_cuda_request": _dict(summary.get("config")).get("device") == "cuda",
        "adapter_files_present": adapter_files_present,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.extend(
            [
                "do_not_package_phase2af",
                "do_not_run_sealed_phase2af",
                "freeze_phase2af_holdout_failure_before_new_training",
            ]
        )
    return {
        "artifact_family": "phase2af_hardened_structural_sidecar_holdout_postflight",
        "passed": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "blocked_actions": sorted(set(blocked_actions)),
        "metrics": {
            "split": split,
            "command_slot_accuracy": accuracy,
            "command_slot_count": count,
            "source_overlap_accuracy": source_overlap,
            "pretrain_identity_text_ablated_source_overlap_accuracy": pretrain_source_overlap,
            "raw_source_overlap_accuracy": raw_source,
            "runtime_identity_heuristic_accuracy": runtime_identity,
            "model_minus_source_overlap_accuracy": model_minus_source,
            "model_minus_runtime_identity_accuracy": model_minus_identity,
        },
        "thresholds": {
            "min_accuracy": min_accuracy,
            "min_rows": min_rows,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_model_minus_runtime_identity": min_model_minus_runtime_identity,
            "min_source_overlap_accuracy": min_source_overlap_accuracy,
            "max_source_overlap_accuracy": max_source_overlap_accuracy,
            "max_runtime_identity_accuracy": max_runtime_identity_accuracy,
        },
        "claim_boundary": (
            "A passing holdout postflight supports only a non-sealed repo-disjoint "
            "holdout mechanism signal. It is still not a package, sealed-transfer, "
            "production-autonomy, or epoch-making architecture claim."
        ),
        "allowed_next_action": (
            "run_phase2af_ablation_controls_before_package"
            if passed
            else "freeze_phase2af_holdout_failure_and_diagnose"
        ),
        "inputs": {
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "eval_summary_json": str(Path(eval_summary_json)),
            "adapter_dir": str(adapter_path) if str(adapter_path) else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AF holdout evaluation gates.")
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--eval-summary-json", required=True)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-model-minus-runtime-identity", type=float, default=0.10)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2af_holdout_postflight(
        pretrain_gate_json=args.pretrain_gate_json,
        eval_summary_json=args.eval_summary_json,
        split=args.split,
        adapter_dir=args.adapter_dir,
        min_accuracy=args.min_accuracy,
        min_rows=args.min_rows,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_model_minus_runtime_identity=args.min_model_minus_runtime_identity,
    )
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
