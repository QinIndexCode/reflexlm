from __future__ import annotations

import argparse
import json
import math
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


def _baseline_accuracy(payload: dict[str, Any], split: str) -> float | None:
    baselines = _dict(payload.get("source_overlap_command_slot_baseline"))
    direct = _number(_dict(baselines.get(split)).get("accuracy"))
    if direct is not None:
        return direct
    if len(baselines) == 1:
        only = next(iter(baselines.values()))
        return _number(_dict(only).get("accuracy"))
    return None


def _encoded_candidates(payload: dict[str, Any], split: str) -> float | None:
    encodings = _dict(payload.get("pairwise_candidate_encoding"))
    direct = _number(_dict(encodings.get(split)).get("pairwise_scored_candidates"))
    if direct is not None:
        return direct
    if len(encodings) == 1:
        only = next(iter(encodings.values()))
        return _number(_dict(only).get("pairwise_scored_candidates"))
    return None


def _split_hash(payload: dict[str, Any], split: str) -> str | None:
    hashes = _dict(payload.get("effective_split_hashes"))
    direct = hashes.get(split)
    if isinstance(direct, str) and direct:
        return direct
    prefix = f"{split}_"
    matches = [value for key, value in hashes.items() if key.startswith(prefix)]
    if len(matches) == 1 and isinstance(matches[0], str) and matches[0]:
        return matches[0]
    if len(hashes) == 1:
        only = next(iter(hashes.values()))
        if isinstance(only, str) and only:
            return only
    return None


def _audit_passed(path: str | Path | None) -> bool:
    if not path:
        return False
    return _read_json(path).get("passed") is True


def _metric(payload: dict[str, Any], key: str) -> float | None:
    return _number(payload.get(key))


def build_phase2ag_smoke_postflight(
    *,
    split_manifest_json: str | Path,
    train_audit_json: str | Path,
    val_audit_json: str | Path,
    holdout_audit_json: str | Path,
    head_manifest_json: str | Path,
    holdout_head_manifest_json: str | Path,
    training_summary_json: str | Path,
    holdout_eval_json: str | Path,
    adapter_dir: str | Path | None = None,
    min_val_accuracy: float = 0.85,
    min_holdout_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_val_rows: int = 48,
    min_holdout_rows: int = 48,
    require_val_model_minus_source_overlap: bool = True,
) -> dict[str, Any]:
    split_manifest = _read_json(split_manifest_json)
    head_manifest = _read_json(head_manifest_json)
    holdout_head_manifest = _read_json(holdout_head_manifest_json)
    summary = _read_json(training_summary_json)
    holdout_eval = _read_json(holdout_eval_json)

    val_metrics = _latest_val_metrics(summary)
    holdout_metrics = _dict(holdout_eval.get("eval_metrics"))
    val_accuracy = _metric(val_metrics, "command_slot_accuracy")
    holdout_accuracy = _metric(holdout_metrics, "command_slot_accuracy")
    val_count = _metric(val_metrics, "command_slot_count")
    holdout_count = _metric(holdout_metrics, "command_slot_count")
    val_source = _baseline_accuracy(summary, "val")
    holdout_source = _baseline_accuracy(holdout_eval, "holdout")
    val_minus_source = (
        val_accuracy - val_source
        if isinstance(val_accuracy, float) and isinstance(val_source, float)
        else None
    )
    holdout_minus_source = (
        holdout_accuracy - holdout_source
        if isinstance(holdout_accuracy, float) and isinstance(holdout_source, float)
        else None
    )

    summary_hashes = _dict(summary.get("effective_split_hashes"))
    holdout_eval_hash = _split_hash(holdout_eval, "phase2c_head_holdout")
    head_splits = _dict(head_manifest.get("splits"))
    holdout_head_splits = _dict(holdout_head_manifest.get("splits"))
    head_train_hash = _dict(head_splits.get("train")).get("sha256")
    head_val_hash = _dict(head_splits.get("val")).get("sha256")
    holdout_head_val_hash = _dict(holdout_head_splits.get("val")).get("sha256")
    adapter_path = Path(adapter_dir or summary.get("adapter_output_dir") or "")
    adapter_files_present = (
        bool(str(adapter_path))
        and (adapter_path / "native_heads.pt").exists()
        and (adapter_path / "head_config.json").exists()
    )

    checks = {
        "split_manifest_passed": split_manifest.get("passed") is True,
        "split_repo_disjoint": _dict(split_manifest.get("checks")).get("repo_disjoint")
        is True,
        "split_slot_coverage": _dict(split_manifest.get("checks")).get(
            "train_covers_val_and_holdout_slots"
        )
        is True,
        "train_audit_passed": _audit_passed(train_audit_json),
        "val_audit_passed": _audit_passed(val_audit_json),
        "holdout_audit_passed": _audit_passed(holdout_audit_json),
        "head_manifests_passed_source_gates": head_manifest.get(
            "source_data_health_passed"
        )
        is True
        and head_manifest.get("source_pretrain_gate_passed") is True
        and holdout_head_manifest.get("source_data_health_passed") is True
        and holdout_head_manifest.get("source_pretrain_gate_passed") is True,
        "training_hashes_present": bool(
            summary_hashes.get("phase2c_head_train")
            and summary_hashes.get("phase2c_head_val")
        ),
        "holdout_hash_present": bool(holdout_eval_hash),
        "training_hashes_match_head_manifest": summary_hashes.get(
            "phase2c_head_train"
        )
        == head_train_hash
        and summary_hashes.get("phase2c_head_val") == head_val_hash,
        "holdout_hash_matches_eval_rows": holdout_eval_hash
        == holdout_eval.get("eval_rows_hash")
        == holdout_head_val_hash,
        "val_command_slot_count_min": isinstance(val_count, float)
        and val_count >= min_val_rows,
        "holdout_command_slot_count_min": isinstance(holdout_count, float)
        and holdout_count >= min_holdout_rows,
        "val_command_slot_accuracy_min": isinstance(val_accuracy, float)
        and val_accuracy >= min_val_accuracy,
        "holdout_command_slot_accuracy_min": isinstance(holdout_accuracy, float)
        and holdout_accuracy >= min_holdout_accuracy,
        "val_source_overlap_present": isinstance(val_source, float),
        "holdout_source_overlap_present": isinstance(holdout_source, float),
        "val_model_beats_source_overlap": (
            True
            if not require_val_model_minus_source_overlap
            else isinstance(val_minus_source, float)
            and val_minus_source >= min_model_minus_source_overlap
        ),
        "holdout_model_beats_source_overlap": isinstance(holdout_minus_source, float)
        and holdout_minus_source >= min_model_minus_source_overlap,
        "pairwise_disabled": summary.get("use_pairwise_command_reranker") is False
        and holdout_eval.get("use_pairwise_command_reranker") is False,
        "pairwise_encoded_candidates_zero": (_encoded_candidates(summary, "train") == 0.0)
        and (_encoded_candidates(summary, "val") == 0.0)
        and (_encoded_candidates(holdout_eval, "holdout") == 0.0),
        "no_json_motor_target": summary.get("no_json_motor_target") is True
        and holdout_eval.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target")
        == 0
        and holdout_eval.get("low_level_qwen_calls_target") == 0,
        "trained_and_evaluated_on_cuda": _dict(summary.get("config")).get("device")
        == "cuda"
        and str(holdout_eval.get("device") or "").startswith("cuda"),
        "adapter_files_present": adapter_files_present,
    }
    passed = all(checks.values())

    blocked_actions = [
        "do_not_package_phase2ag_from_smoke",
        "do_not_run_sealed_phase2ag_from_smoke",
        "do_not_claim_epoch_making_architecture",
        "do_not_claim_open_ended_debugging_generalization",
    ]
    if not passed:
        blocked_actions.insert(0, "do_not_scale_phase2ag_training")
    else:
        blocked_actions.append("require_sidecar_erased_and_wrong_sidecar_controls_before_claim")

    report = {
        "artifact_family": "phase2ag_verifiable_candidate_sidecar_smoke_postflight",
        "passed": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "claim_bearing_mechanism_evidence": False,
        "checks": checks,
        "metrics": {
            "train_examples": summary.get("train_examples"),
            "val_examples": summary.get("val_examples"),
            "holdout_examples": holdout_eval.get("eval_examples"),
            "val_command_slot_accuracy": val_accuracy,
            "holdout_command_slot_accuracy": holdout_accuracy,
            "val_command_slot_count": val_count,
            "holdout_command_slot_count": holdout_count,
            "val_source_overlap_accuracy": val_source,
            "holdout_source_overlap_accuracy": holdout_source,
            "val_model_minus_source_overlap_accuracy": val_minus_source,
            "holdout_model_minus_source_overlap_accuracy": holdout_minus_source,
        },
        "thresholds": {
            "min_val_accuracy": min_val_accuracy,
            "min_holdout_accuracy": min_holdout_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "require_val_model_minus_source_overlap": require_val_model_minus_source_overlap,
        },
        "blocked_actions": sorted(set(blocked_actions)),
        "allowed_next_action": (
            "run_nonsealed_sidecar_erased_and_wrong_sidecar_controls"
            if passed
            else "freeze_phase2ag_smoke_failure_and_fix_nonsealed_design"
        ),
        "claim_boundary": (
            "This smoke validates that a runtime-visible verifiable candidate sidecar can be "
            "learned on a small repo-disjoint non-sealed split. Because the split is selected "
            "for unique probe resolvability, it is not sufficient evidence for sealed transfer, "
            "production autonomy, open-ended debugging, or an epoch-making architecture claim."
        ),
        "inputs": {
            "split_manifest_json": str(Path(split_manifest_json)),
            "train_audit_json": str(Path(train_audit_json)),
            "val_audit_json": str(Path(val_audit_json)),
            "holdout_audit_json": str(Path(holdout_audit_json)),
            "head_manifest_json": str(Path(head_manifest_json)),
            "holdout_head_manifest_json": str(Path(holdout_head_manifest_json)),
            "training_summary_json": str(Path(training_summary_json)),
            "holdout_eval_json": str(Path(holdout_eval_json)),
            "adapter_dir": str(adapter_path) if str(adapter_path) else None,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AG smoke postflight gates.")
    parser.add_argument("--split-manifest-json", required=True)
    parser.add_argument("--train-audit-json", required=True)
    parser.add_argument("--val-audit-json", required=True)
    parser.add_argument("--holdout-audit-json", required=True)
    parser.add_argument("--head-manifest-json", required=True)
    parser.add_argument("--holdout-head-manifest-json", required=True)
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--holdout-eval-json", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--min-val-accuracy", type=float, default=0.85)
    parser.add_argument("--min-holdout-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-val-rows", type=int, default=48)
    parser.add_argument("--min-holdout-rows", type=int, default=48)
    parser.add_argument(
        "--no-require-val-model-minus-source-overlap",
        action="store_true",
        help=(
            "Use val only as an accuracy/hash gate and require mechanism delta on "
            "holdout/control splits. This is intended for fullscale runs where val is "
            "near source-overlap ceiling."
        ),
    )
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2ag_smoke_postflight(
        split_manifest_json=args.split_manifest_json,
        train_audit_json=args.train_audit_json,
        val_audit_json=args.val_audit_json,
        holdout_audit_json=args.holdout_audit_json,
        head_manifest_json=args.head_manifest_json,
        holdout_head_manifest_json=args.holdout_head_manifest_json,
        training_summary_json=args.training_summary_json,
        holdout_eval_json=args.holdout_eval_json,
        adapter_dir=args.adapter_dir,
        min_val_accuracy=args.min_val_accuracy,
        min_holdout_accuracy=args.min_holdout_accuracy,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
        require_val_model_minus_source_overlap=not args.no_require_val_model_minus_source_overlap,
    )
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
