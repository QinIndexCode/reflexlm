from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.native_head_training import Phase2CHeadJsonlDataset, _canonical_rows_sha256


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8-sig"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _last_epoch(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history") or []
    return history[-1] if history else {}


def _val_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return _last_epoch(summary).get("val_metrics") or {}


def _metric(payload: dict[str, Any], name: str) -> float | None:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    value = metrics.get(name, payload.get(name))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _effective_head_hash(
    *,
    manifest: dict[str, Any],
    split: str,
    limit: int | None,
    debug_command_oversample: int,
    balance_debug_command_intents: bool,
) -> str | None:
    split_payload = manifest.get("splits", {}).get(split, {})
    path = split_payload.get("path")
    if path:
        dataset = Phase2CHeadJsonlDataset(
            path,
            limit=limit,
            debug_command_oversample=debug_command_oversample,
            balance_debug_command_intents=balance_debug_command_intents,
        )
        return _canonical_rows_sha256(dataset.rows)
    return split_payload.get("sha256")


def build_phase2m_v2_postflight(
    *,
    training_summary_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    design_maturity_json: str | Path,
    head_manifest_json: str | Path,
    native_head_only_eval_json: str | Path | None = None,
    postflight_stage: str = "smoke",
    min_val_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.10,
    min_full_minus_native_head_only: float = 0.10,
    max_smoke_train_records: int = 128,
    max_smoke_val_records: int = 512,
) -> dict[str, Any]:
    if postflight_stage not in {"smoke", "full"}:
        raise ValueError("postflight_stage must be smoke or full")
    summary = _load_json(training_summary_json)
    data_health = _load_json(data_health_json)
    pretrain_gate = _load_json(pretrain_gate_json)
    design = _load_json(design_maturity_json)
    head_manifest = _load_json(head_manifest_json)
    native_eval = _load_json(native_head_only_eval_json)
    val_metrics = _val_metrics(summary)
    run_manifest = summary.get("run_manifest") or {}
    source_overlap = summary.get("source_overlap_command_slot_baseline", {}).get("val", {})
    val_accuracy = _float(val_metrics.get("command_slot_accuracy"))
    source_overlap_accuracy = _float(source_overlap.get("accuracy"))
    model_minus_source = val_accuracy - source_overlap_accuracy
    summary_hashes = summary.get("effective_split_hashes") or {}
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    train_hash = _effective_head_hash(
        manifest=head_manifest,
        split="train",
        limit=config.get("max_train_records"),
        debug_command_oversample=int(config.get("debug_command_oversample") or 1),
        balance_debug_command_intents=bool(config.get("balance_debug_command_intents")),
    )
    val_hash = _effective_head_hash(
        manifest=head_manifest,
        split="val",
        limit=config.get("max_val_records"),
        debug_command_oversample=1,
        balance_debug_command_intents=False,
    )
    full_completion = _metric(summary, "task_completion_rate")
    full_completion_source = "task_completion_rate"
    if full_completion is None and postflight_stage == "full":
        full_completion = val_accuracy
        full_completion_source = "val_command_slot_accuracy"
    native_completion = _metric(native_eval, "task_completion_rate")
    full_minus_native = (
        full_completion - native_completion
        if isinstance(full_completion, float) and isinstance(native_completion, float)
        else None
    )
    last_epoch = _last_epoch(summary)
    train_pairwise = int(last_epoch.get("train_pairwise_encoded_candidates") or 0)
    val_pairwise = int(_float(val_metrics.get("pairwise_encoded_candidates")))
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain_gate.get("passed") is True,
        "design_maturity_claim_bearing": (
            design.get("ready_for_claim_bearing_training") is True
            and design.get("passed") is True
        ),
        "summary_exists_and_completed": bool(summary and run_manifest.get("finished_at_utc")),
        "head_hashes_match_training_summary": (
            bool(train_hash)
            and bool(val_hash)
            and summary_hashes.get("phase2c_head_train") == train_hash
            and summary_hashes.get("phase2c_head_val") == val_hash
        ),
        "smoke_limits_recorded": (
            True
            if postflight_stage == "full"
            else int(summary.get("train_examples") or 0) <= max_smoke_train_records
            and int(summary.get("val_examples") or 0) <= max_smoke_val_records
            and int(summary.get("config", {}).get("max_train_records") or 0)
            <= max_smoke_train_records
            and int(summary.get("config", {}).get("max_val_records") or 0)
            <= max_smoke_val_records
        ),
        "val_command_slot_accuracy_gate": val_accuracy >= min_val_command_slot_accuracy,
        "source_overlap_baseline_recorded": source_overlap.get("total", 0) > 0,
        "model_beats_source_overlap": model_minus_source >= min_model_minus_source_overlap,
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "pairwise_disabled_for_phase2m_isolation": (
            summary.get("use_pairwise_command_reranker") is False
            and summary.get("head_config", {}).get("use_pairwise_command_reranker") is False
            and train_pairwise == 0
            and val_pairwise == 0
        ),
        "additive_latent_fusion": (
            summary.get("config", {}).get("latent_fusion") == "additive"
            and summary.get("head_config", {}).get("latent_fusion") == "additive"
        ),
        "native_head_only_eval_required_for_full": (
            True
            if postflight_stage == "smoke"
            else isinstance(native_completion, float)
        ),
        "full_beats_native_head_only_for_package": (
            True
            if postflight_stage == "smoke"
            else isinstance(full_minus_native, float)
            and full_minus_native >= min_full_minus_native_head_only
        ),
    }
    passed = all(checks.values())
    ready_for_full_train = passed and postflight_stage == "smoke"
    ready_for_package = passed and postflight_stage == "full"
    blocked_actions: list[str] = []
    if not checks["design_maturity_claim_bearing"]:
        blocked_actions.append("do_not_treat_phase2m_training_as_claim_bearing_evidence")
    if not checks["model_beats_source_overlap"]:
        blocked_actions.append("do_not_continue_without_beating_source_overlap")
    if not checks["full_beats_native_head_only_for_package"]:
        blocked_actions.append("do_not_package_without_full_beating_native_head_only")
    if not checks["val_command_slot_accuracy_gate"]:
        blocked_actions.append("do_not_continue_until_phase2m_val_gate_passes")
    return {
        "audit_family": "phase2m_v2_postflight",
        "postflight_stage": postflight_stage,
        "passed": passed,
        "ready_for_full_train": ready_for_full_train,
        "ready_for_package": ready_for_package,
        "ready_for_sealed_eval": False,
        "allowed_next_action": (
            "run_phase2m_v2_full_nonsealed_training"
            if ready_for_full_train
            else "run_phase2m_v2_package_only"
            if ready_for_package
            else "revise_phase2m_v2_before_continuing"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "metrics": {
            "val_command_slot_accuracy": val_accuracy,
            "val_command_slot_count": _float(val_metrics.get("command_slot_count")),
            "source_overlap_val_accuracy": source_overlap_accuracy,
            "model_minus_source_overlap_accuracy": model_minus_source,
            "duration_seconds": _float(run_manifest.get("duration_seconds")),
            "train_pairwise_encoded_candidates": train_pairwise,
            "val_pairwise_encoded_candidates": val_pairwise,
            "full_completion": full_completion,
            "full_completion_source": full_completion_source if full_completion is not None else None,
            "native_head_only_completion": native_completion,
            "full_minus_native_head_only": full_minus_native,
        },
        "thresholds": {
            "min_val_command_slot_accuracy": min_val_command_slot_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_full_minus_native_head_only": min_full_minus_native_head_only,
            "max_smoke_train_records": max_smoke_train_records,
            "max_smoke_val_records": max_smoke_val_records,
        },
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "data_health_json": str(Path(data_health_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "design_maturity_json": str(Path(design_maturity_json)),
            "head_manifest_json": str(Path(head_manifest_json)),
            "native_head_only_eval_json": (
                str(Path(native_head_only_eval_json)) if native_head_only_eval_json else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase2M-v2 smoke/full postflight gate.")
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--design-maturity-json", required=True)
    parser.add_argument("--head-manifest-json", required=True)
    parser.add_argument("--native-head-only-eval-json")
    parser.add_argument("--output-json")
    parser.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.10)
    parser.add_argument("--min-full-minus-native-head-only", type=float, default=0.10)
    parser.add_argument("--max-smoke-train-records", type=int, default=128)
    parser.add_argument("--max-smoke-val-records", type=int, default=512)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2m_v2_postflight(
        training_summary_json=args.training_summary_json,
        data_health_json=args.data_health_json,
        pretrain_gate_json=args.pretrain_gate_json,
        design_maturity_json=args.design_maturity_json,
        head_manifest_json=args.head_manifest_json,
        native_head_only_eval_json=args.native_head_only_eval_json,
        postflight_stage=args.stage,
        min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_full_minus_native_head_only=args.min_full_minus_native_head_only,
        max_smoke_train_records=args.max_smoke_train_records,
        max_smoke_val_records=args.max_smoke_val_records,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
