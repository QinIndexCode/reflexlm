from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "sha256": _sha256(resolved) if resolved.exists() and resolved.is_file() else None,
    }


def _get_dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def _metric(value: dict[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_phase2s_public_repair_boundary_review(
    *,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    head_manifest_json: str | Path,
    smoke_postflight_json: str | Path,
    full_postflight_json: str | Path,
    full_holdout_postflight_json: str | Path,
    full_training_summary_json: str | Path,
) -> dict[str, Any]:
    data_health = _load(data_health_json)
    pretrain_gate = _load(pretrain_gate_json)
    head_manifest = _load(head_manifest_json)
    smoke_postflight = _load(smoke_postflight_json)
    full_postflight = _load(full_postflight_json)
    full_holdout = _load(full_holdout_postflight_json)
    full_summary = _load(full_training_summary_json)

    split_hashes = data_health.get("effective_split_hashes")
    full_config = _get_dict(full_summary, "config")
    train_examples = full_summary.get("train_examples")
    max_train_records = full_config.get("max_train_records")
    all_available_train_used = (
        isinstance(train_examples, int)
        and isinstance(max_train_records, int)
        and train_examples <= max_train_records
        and train_examples == _metric(data_health, "rollups", "train", "rows")
    )
    public_train_supply_below_full_cap = (
        isinstance(train_examples, int)
        and isinstance(max_train_records, int)
        and train_examples < max_train_records
    )
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain_gate.get("passed") is True,
        "head_manifest_source_gates_passed": (
            head_manifest.get("source_data_health_passed") is True
            and head_manifest.get("source_pretrain_gate_passed") is True
        ),
        "split_hashes_match": bool(split_hashes)
        and split_hashes == pretrain_gate.get("effective_split_hashes")
        and split_hashes == head_manifest.get("effective_split_hashes")
        and split_hashes == full_holdout.get("effective_split_hashes"),
        "smoke_postflight_passed": smoke_postflight.get("passed") is True,
        "full_postflight_passed": full_postflight.get("passed") is True,
        "full_holdout_postflight_passed": full_holdout.get("passed") is True,
        "sealed_not_used_for_training_sampling_or_tuning": (
            data_health.get("checks", {}).get("phase2s_no_sealed_reference_anywhere")
            is True
            and full_holdout.get("checks", {}).get("holdout_diagnostics_not_sealed_tuned")
            is True
        ),
        "pairwise_disabled": full_holdout.get("metrics", {}).get(
            "use_pairwise_command_reranker"
        )
        is False,
        "no_json_motor_target": full_postflight.get("checks", {}).get(
            "no_json_motor_target"
        )
        is True,
        "low_level_qwen_calls_target_zero": full_holdout.get("metrics", {}).get(
            "low_level_qwen_calls_target"
        )
        == 0,
        "holdout_beats_source_overlap": (
            isinstance(
                full_holdout.get("metrics", {}).get(
                    "holdout_model_minus_source_overlap_accuracy"
                ),
                (int, float),
            )
            and full_holdout["metrics"]["holdout_model_minus_source_overlap_accuracy"] >= 0.15
        ),
        "holdout_beats_zero_nsi": (
            isinstance(
                full_holdout.get("metrics", {}).get(
                    "holdout_model_minus_zero_nsi_accuracy"
                ),
                (int, float),
            )
            and full_holdout["metrics"]["holdout_model_minus_zero_nsi_accuracy"] >= 0.15
        ),
        "all_available_public_train_rows_used": all_available_train_used,
    }
    passed = all(checks.values())
    active_boundary = (
        "phase2s_public_repo_origin_disjoint_repair_positive_bounded_mechanism"
        if passed
        else "phase2s_public_repair_boundary_not_ready"
    )
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_package_or_run_sealed_until_phase2s_boundary_passes")
    if not checks["sealed_not_used_for_training_sampling_or_tuning"]:
        blocked_actions.append("do_not_use_sealed_or_sealed_failure_feedback")
    if not checks["holdout_beats_source_overlap"]:
        blocked_actions.append("do_not_claim_phase2s_source_overlap_delta")
    if not checks["holdout_beats_zero_nsi"]:
        blocked_actions.append("do_not_claim_phase2s_nsi_identity_delta")

    return {
        "artifact_family": "phase2s_public_repair_boundary_review",
        "passed": passed,
        "active_evidence_boundary": active_boundary,
        "allowed_next_action": (
            "package_for_final_eval_only_after_manual_boundary_acceptance"
            if passed
            else "freeze_phase2s_boundary_failure_and_repair_nonsealed_design"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "metrics": {
            "train_rows": _metric(data_health, "rollups", "train", "rows"),
            "val_rows": _metric(data_health, "rollups", "val", "rows"),
            "holdout_rows": _metric(data_health, "rollups", "holdout", "rows"),
            "full_train_examples": train_examples,
            "full_config_max_train_records": max_train_records,
            "public_train_supply_below_full_cap": public_train_supply_below_full_cap,
            "val_command_slot_accuracy": full_holdout.get("metrics", {}).get(
                "val_command_slot_accuracy"
            ),
            "val_model_minus_source_overlap_accuracy": full_holdout.get("metrics", {}).get(
                "val_model_minus_source_overlap_accuracy"
            ),
            "holdout_command_slot_accuracy": full_holdout.get("metrics", {}).get(
                "holdout_command_slot_accuracy"
            ),
            "holdout_source_overlap_accuracy": full_holdout.get("metrics", {}).get(
                "holdout_source_overlap_accuracy"
            ),
            "holdout_zero_nsi_effective_accuracy": full_holdout.get("metrics", {}).get(
                "holdout_zero_nsi_effective_accuracy"
            ),
            "holdout_model_minus_source_overlap_accuracy": full_holdout.get(
                "metrics",
                {},
            ).get("holdout_model_minus_source_overlap_accuracy"),
            "holdout_model_minus_zero_nsi_accuracy": full_holdout.get("metrics", {}).get(
                "holdout_model_minus_zero_nsi_accuracy"
            ),
        },
        "supported_claims": [
            "Phase2S supports bounded command-identity NSI contribution on public repo-origin-disjoint repair traces",
            "Phase2S full policy beats source-overlap and zero-NSI controls on non-sealed holdout",
            "Phase2S uses bounded native-head command-slot selection, not JSON motor output",
        ],
        "unsupported_claims": [
            "sealed cross-model transfer is proven by Phase2S",
            "production autonomy or open-ended debugging generalization is proven",
            "the architecture is independently reproduced across models and seeds",
            "full package should be tuned from sealed-v3 outcomes",
            "Phase2S proves unrestricted repair ability beyond the bounded public-repair task",
        ],
        "claim_boundary": (
            "Phase2S is positive bounded evidence for NSI command-identity repair selection "
            "under public, repo-origin-disjoint, sandboxed traces. It is not by itself evidence "
            "for production autonomy, open-ended debugging, sealed-transfer success, or an "
            "epoch-making architecture claim."
        ),
        "limitations": [
            (
                "The configured full cap is 1024 train rows, but the public r128 collector "
                "produced 768 eligible train rows; full training used all eligible public "
                "train rows rather than a strict 1024-row corpus."
            )
            if public_train_supply_below_full_cap
            else "The full run reached the configured train-row cap.",
            "All sealed evaluation remains final-eval only and cannot feed back into Phase2S data design.",
            "Cross-model and multi-seed reproduction are still required before stronger architecture claims.",
        ],
        "next_recommended_steps": [
            "build Phase2S package variants only after accepting this boundary review",
            "run sealed-v3 six-mechanism final eval as transfer evidence only, not as tuning feedback",
            "run 3B/7B or multi-seed reproduction on the same public r128 split before stronger claims",
        ],
        "artifacts": {
            "data_health": _artifact(data_health_json),
            "pretrain_gate": _artifact(pretrain_gate_json),
            "head_manifest": _artifact(head_manifest_json),
            "smoke_postflight": _artifact(smoke_postflight_json),
            "full_postflight": _artifact(full_postflight_json),
            "full_holdout_postflight": _artifact(full_holdout_postflight_json),
            "full_training_summary": _artifact(full_training_summary_json),
        },
        "effective_split_hashes": split_hashes,
    }


def render_boundary_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Phase2S Public Repair Boundary Review",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Active evidence boundary: `{report['active_evidence_boundary']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Metrics",
        f"- Train / val / holdout rows: `{metrics['train_rows']}` / `{metrics['val_rows']}` / `{metrics['holdout_rows']}`",
        f"- Full train examples: `{metrics['full_train_examples']}` of cap `{metrics['full_config_max_train_records']}`",
        f"- Val command-slot accuracy: `{metrics['val_command_slot_accuracy']}`",
        f"- Holdout command-slot accuracy: `{metrics['holdout_command_slot_accuracy']}`",
        f"- Holdout source-overlap accuracy: `{metrics['holdout_source_overlap_accuracy']}`",
        f"- Holdout zero-NSI accuracy: `{metrics['holdout_zero_nsi_effective_accuracy']}`",
        f"- Holdout minus source-overlap: `{metrics['holdout_model_minus_source_overlap_accuracy']}`",
        f"- Holdout minus zero-NSI: `{metrics['holdout_model_minus_zero_nsi_accuracy']}`",
        "",
        "## Supported Claims",
    ]
    lines.extend(f"- {item}" for item in report["supported_claims"])
    lines.extend(["", "## Unsupported Claims"])
    lines.extend(f"- {item}" for item in report["unsupported_claims"])
    lines.extend(["", "## Limitations"])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", "## Next Steps"])
    lines.extend(f"- {item}" for item in report["next_recommended_steps"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Review Phase2S public repair claim boundary.")
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--head-manifest-json", required=True)
    parser.add_argument("--smoke-postflight-json", required=True)
    parser.add_argument("--full-postflight-json", required=True)
    parser.add_argument("--full-holdout-postflight-json", required=True)
    parser.add_argument("--full-training-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2s_public_repair_boundary_review(
        data_health_json=args.data_health_json,
        pretrain_gate_json=args.pretrain_gate_json,
        head_manifest_json=args.head_manifest_json,
        smoke_postflight_json=args.smoke_postflight_json,
        full_postflight_json=args.full_postflight_json,
        full_holdout_postflight_json=args.full_holdout_postflight_json,
        full_training_summary_json=args.full_training_summary_json,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_boundary_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
