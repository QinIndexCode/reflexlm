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


def _latest_val_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history")
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict):
            return _dict(latest.get("val_metrics"))
    return _dict(summary.get("val_metrics"))


def _metric_accuracy(pretrain: dict[str, Any], split: str, metric: str) -> float | None:
    payload = _dict(_dict(_dict(pretrain.get("split_metrics")).get(split)).get(metric))
    return _number(payload.get("accuracy"))


def _baseline_accuracy(summary: dict[str, Any], split: str) -> float | None:
    payload = _dict(
        _dict(summary.get("source_overlap_command_slot_baseline")).get(split)
    )
    return _number(payload.get("accuracy"))


def _encoded_candidates(summary: dict[str, Any], split: str) -> float | None:
    payload = _dict(_dict(summary.get("pairwise_candidate_encoding")).get(split))
    return _number(payload.get("pairwise_scored_candidates"))


def build_phase2af_smoke_postflight(
    *,
    pretrain_gate_json: str | Path,
    training_summary_json: str | Path,
    head_manifest_json: str | Path | None = None,
    adapter_dir: str | Path | None = None,
    min_val_accuracy: float = 0.85,
    min_val_rows: int = 30,
    min_model_minus_source_overlap: float = 0.15,
    min_model_minus_runtime_identity: float = 0.10,
    min_source_overlap_accuracy: float = 0.05,
    max_source_overlap_accuracy: float = 0.75,
    max_runtime_identity_accuracy: float = 0.90,
    min_full_train_rows: int = 96,
    min_full_val_rows: int = 64,
) -> dict[str, Any]:
    pretrain = _read_json(pretrain_gate_json)
    summary = _read_json(training_summary_json)
    head_manifest = _read_json(head_manifest_json) if head_manifest_json else {}
    val_metrics = _latest_val_metrics(summary)

    val_accuracy = _number(val_metrics.get("command_slot_accuracy"))
    val_count = _number(val_metrics.get("command_slot_count"))
    source_overlap = _baseline_accuracy(summary, "val")
    pretrain_source_overlap = _metric_accuracy(
        pretrain, "val", "identity_text_ablated_source_overlap"
    )
    runtime_identity = _metric_accuracy(pretrain, "val", "runtime_identity_heuristic")
    raw_source = _metric_accuracy(pretrain, "val", "raw_source_overlap")
    model_minus_source = (
        val_accuracy - source_overlap
        if isinstance(val_accuracy, float) and isinstance(source_overlap, float)
        else None
    )
    model_minus_identity = (
        val_accuracy - runtime_identity
        if isinstance(val_accuracy, float) and isinstance(runtime_identity, float)
        else None
    )

    effective_hashes = _dict(summary.get("effective_split_hashes"))
    train_examples = int(summary.get("train_examples") or 0)
    val_examples = int(summary.get("val_examples") or 0)
    adapter_path = Path(adapter_dir or summary.get("adapter_output_dir") or "")
    adapter_files_present = (
        bool(str(adapter_path))
        and (adapter_path / "native_heads.pt").exists()
        and (adapter_path / "head_config.json").exists()
    )

    smoke_checks = {
        "pretrain_gate_passed": pretrain.get("passed") is True,
        "val_command_slot_count_min": isinstance(val_count, float) and val_count >= min_val_rows,
        "val_command_slot_accuracy_min": isinstance(val_accuracy, float)
        and val_accuracy >= min_val_accuracy,
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
        "pairwise_encoded_candidates_zero": (_encoded_candidates(summary, "train") == 0.0)
        and (_encoded_candidates(summary, "val") == 0.0),
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "effective_split_hashes_present": bool(
            effective_hashes.get("phase2c_head_train")
            and effective_hashes.get("phase2c_head_val")
        ),
        "trained_on_cuda": _dict(summary.get("config")).get("device") == "cuda",
        "adapter_files_present": adapter_files_present,
    }
    smoke_passed = all(smoke_checks.values())

    full_scale_checks = {
        "smoke_postflight_passed": smoke_passed,
        "train_rows_sufficient_for_full_gate": train_examples >= min_full_train_rows,
        "val_rows_sufficient_for_full_gate": val_examples >= min_full_val_rows,
    }
    ready_for_full_train = all(full_scale_checks.values())

    warning_flags = {
        "small_smoke_only": train_examples < min_full_train_rows
        or val_examples < min_full_val_rows,
        "single_command_intent_observed": len(
            _dict(
                _dict(_dict(summary.get("slot_intent_distribution")).get("val")).get(
                    "command_intents"
                )
            )
        )
        <= 1,
        "head_manifest_family_reused_from_phase2s": head_manifest.get("dataset_family")
        == "phase2s_public_repair_head_dataset",
    }

    blocked_actions: list[str] = []
    if not smoke_passed:
        blocked_actions.extend(
            [
                "do_not_scale_phase2af_training",
                "do_not_package_phase2af",
                "do_not_claim_phase2af_hardened_structural_sidecar_mechanism",
            ]
        )
    if smoke_passed and not ready_for_full_train:
        blocked_actions.extend(
            [
                "do_not_package_phase2af_from_smoke",
                "do_not_claim_full_phase2af_mechanism_from_smoke",
                "expand_or_collect_nonsealed_phase2af_rows_before_full_train",
            ]
        )

    return {
        "artifact_family": "phase2af_hardened_structural_sidecar_smoke_postflight",
        "passed": smoke_passed,
        "ready_for_full_train": ready_for_full_train,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": smoke_checks,
        "full_scale_checks": full_scale_checks,
        "warning_flags": warning_flags,
        "blocked_actions": sorted(set(blocked_actions)),
        "metrics": {
            "val_command_slot_accuracy": val_accuracy,
            "val_command_slot_count": val_count,
            "source_overlap_accuracy": source_overlap,
            "pretrain_identity_text_ablated_source_overlap_accuracy": pretrain_source_overlap,
            "raw_source_overlap_accuracy": raw_source,
            "runtime_identity_heuristic_accuracy": runtime_identity,
            "model_minus_source_overlap_accuracy": model_minus_source,
            "model_minus_runtime_identity_accuracy": model_minus_identity,
            "train_examples": train_examples,
            "val_examples": val_examples,
        },
        "thresholds": {
            "min_val_accuracy": min_val_accuracy,
            "min_val_rows": min_val_rows,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_model_minus_runtime_identity": min_model_minus_runtime_identity,
            "min_source_overlap_accuracy": min_source_overlap_accuracy,
            "max_source_overlap_accuracy": max_source_overlap_accuracy,
            "max_runtime_identity_accuracy": max_runtime_identity_accuracy,
            "min_full_train_rows": min_full_train_rows,
            "min_full_val_rows": min_full_val_rows,
        },
        "claim_boundary": (
            "This postflight can validate a non-sealed smoke mechanism signal only. "
            "It does not establish sealed transfer, production autonomy, open-ended debugging, "
            "or an epoch-making architecture claim."
        ),
        "allowed_next_action": (
            "run_phase2af_full_nonsealed_training_only"
            if ready_for_full_train
            else (
                "expand_nonsealed_phase2af_pool_then_rebuild_full_scale_split"
                if smoke_passed
                else "freeze_phase2af_smoke_failure_and_fix_nonsealed_design"
            )
        ),
        "inputs": {
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "training_summary_json": str(Path(training_summary_json)),
            "head_manifest_json": str(Path(head_manifest_json)) if head_manifest_json else None,
            "adapter_dir": str(adapter_path) if str(adapter_path) else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AF smoke postflight gates.")
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--head-manifest-json")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--min-val-accuracy", type=float, default=0.85)
    parser.add_argument("--min-val-rows", type=int, default=30)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-model-minus-runtime-identity", type=float, default=0.10)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2af_smoke_postflight(
        pretrain_gate_json=args.pretrain_gate_json,
        training_summary_json=args.training_summary_json,
        head_manifest_json=args.head_manifest_json,
        adapter_dir=args.adapter_dir,
        min_val_accuracy=args.min_val_accuracy,
        min_val_rows=args.min_val_rows,
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
