from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _summarize(records: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record.get(key, "<missing>"))].append(record)
    summaries: list[dict[str, Any]] = []
    for name, items in buckets.items():
        total = len(items)
        correct = sum(1 for item in items if item.get("command_slot_correct") is True)
        source_correct = sum(1 for item in items if item.get("source_overlap_correct") is True)
        summaries.append(
            {
                "name": name,
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "source_overlap_correct": source_correct,
                "source_overlap_accuracy": source_correct / total if total else 0.0,
            }
        )
    return sorted(summaries, key=lambda item: (item["accuracy"], -item["total"], item["name"]))


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    source_manifest = _dict(row.get("source_task_manifest"))
    policy = _dict(row.get("learned_patch_policy_target"))
    return {
        "repo_origin": source_manifest.get("repo_origin") or "<unknown>",
        "operation": policy.get("patch_operation") or "<unknown>",
        "template": policy.get("patch_template") or "<unknown>",
        "candidate_count": len(_list(row.get("candidate_commands"))),
    }


def _prediction_records(
    *,
    prediction_json: dict[str, Any],
    head_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_example = {str(row.get("example_id")): row for row in head_rows}
    records: list[dict[str, Any]] = []
    for prediction in _list(prediction_json.get("prediction_records")):
        row = rows_by_example.get(str(prediction.get("example_id")), {})
        metadata = _row_metadata(row)
        record = {
            "example_id": prediction.get("example_id"),
            **metadata,
            "gold_slot": prediction.get("command_slot_label"),
            "predicted_slot": prediction.get("command_slot_prediction"),
            "command_slot_correct": _bool(prediction.get("command_slot_correct")),
            "source_overlap_correct": _bool(prediction.get("source_overlap_correct")),
            "command_identity_margin": _float(prediction.get("command_identity_margin")),
            "command_identity_confidence": _float(prediction.get("command_identity_confidence")),
        }
        records.append(record)
    return records


def build_phase2av_holdout_failure_audit(
    *,
    holdout_postflight_json: str | Path,
    prediction_json: str | Path,
    head_rows_jsonl: str | Path,
    data_health_jsons: list[str | Path] | None = None,
    min_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_descriptor_accuracy: float = 0.85,
) -> dict[str, Any]:
    postflight = _read_json(holdout_postflight_json)
    prediction_report = _read_json(prediction_json)
    head_rows = _read_jsonl(head_rows_jsonl)
    data_health_reports = [_read_json(path) for path in data_health_jsons or []]
    metrics = _dict(postflight.get("metrics"))
    records = _prediction_records(prediction_json=prediction_report, head_rows=head_rows)

    total = len(records)
    correct = sum(1 for record in records if record["command_slot_correct"])
    source_correct = sum(1 for record in records if record["source_overlap_correct"])
    by_operation = _summarize(records, "operation")
    by_repo = _summarize(records, "repo_origin")
    by_slot = _summarize(records, "gold_slot")
    by_candidate_count = _summarize(records, "candidate_count")
    worst_operation = by_operation[0] if by_operation else {}
    error_examples = [record for record in records if not record["command_slot_correct"]][:20]

    command_slot = _float(metrics.get("command_slot_accuracy"))
    delta = _float(metrics.get("model_minus_source_overlap_accuracy"))
    patch_operation = _float(metrics.get("patch_operation_accuracy"))
    patch_template = _float(metrics.get("patch_template_slot_accuracy"))
    patch_target = _float(metrics.get("patch_target_file_slot_accuracy"))
    identity_bias = _float(prediction_report.get("command_identity_logit_bias"))

    issue_classification: list[str] = []
    if postflight.get("passed") is False:
        issue_classification.append("nonsealed_holdout_postflight_failed")
    if command_slot < min_command_slot_accuracy:
        issue_classification.append("command_slot_identity_transfer_below_holdout_gate")
    if delta >= min_model_minus_source_overlap:
        issue_classification.append("model_still_beats_source_overlap_on_holdout")
    if (
        patch_operation >= min_descriptor_accuracy
        and patch_template >= min_descriptor_accuracy
        and patch_target >= min_descriptor_accuracy
    ):
        issue_classification.append("descriptor_heads_pass_while_command_slot_fails")
    if identity_bias <= 0:
        issue_classification.append("command_identity_prior_disabled")
    if worst_operation and worst_operation.get("accuracy", 1.0) < min_command_slot_accuracy:
        issue_classification.append(f"operation_specific_gap:{worst_operation['name']}")
    if data_health_reports and all(report.get("passed") is True for report in data_health_reports):
        issue_classification.append("data_health_passed_nonsealed_failure_not_data_gate")

    operation_counts = Counter(str(record["operation"]) for record in records)
    return {
        "artifact_family": "phase2av_holdout_failure_audit",
        "passed": False,
        "claim_boundary": (
            "This audit explains a failed non-sealed Phase2AV holdout evaluation. "
            "It does not authorize full training, packaging, sealed evaluation, "
            "freeform patch generation, production autonomy, or epoch-making claims."
        ),
        "preconditions": {
            "holdout_postflight_passed": postflight.get("passed") is True,
            "prediction_records_present": bool(records),
            "data_health_reports_passed": bool(data_health_reports)
            and all(report.get("passed") is True for report in data_health_reports),
            "sealed_feedback_used": False,
        },
        "metrics": {
            "total": total,
            "command_slot_correct": correct,
            "command_slot_accuracy": correct / total if total else 0.0,
            "source_overlap_correct": source_correct,
            "source_overlap_accuracy": source_correct / total if total else 0.0,
            "model_minus_source_overlap_accuracy": delta,
            "patch_operation_accuracy": patch_operation,
            "patch_template_slot_accuracy": patch_template,
            "patch_target_file_slot_accuracy": patch_target,
            "command_identity_logit_bias": identity_bias,
            "operation_counts": dict(sorted(operation_counts.items())),
            "worst_operation": worst_operation,
            "by_operation": by_operation,
            "by_repo": by_repo,
            "by_gold_slot": by_slot,
            "by_candidate_count": by_candidate_count,
            "error_examples": error_examples,
        },
        "issue_classification": issue_classification,
        "failure_modes": postflight.get("failure_modes", []),
        "blocked_actions": [
            "do_not_start_phase2av_full_training",
            "do_not_package_phase2av",
            "do_not_run_sealed_eval_for_phase2av",
        ],
        "recommended_next_actions": [
            "freeze_holdout_failure_as_nonsealed_transfer_gap",
            "run_hash_bound_nonsealed_capacity_identity_prior_ablation_before_full_training",
            "keep_sealed_v3_excluded_from_data_design_training_and_failure_feedback",
        ],
        "unsupported_claims": [
            "phase2av_full_training_ready",
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "thresholds": {
            "min_command_slot_accuracy": min_command_slot_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_descriptor_accuracy": min_descriptor_accuracy,
        },
        "inputs": {
            "holdout_postflight_json": str(Path(holdout_postflight_json)),
            "prediction_json": str(Path(prediction_json)),
            "head_rows_jsonl": str(Path(head_rows_jsonl)),
            "data_health_jsons": [str(Path(path)) for path in data_health_jsons or []],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AV holdout failure audit.")
    parser.add_argument("--holdout-postflight-json", required=True)
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--head-rows-jsonl", required=True)
    parser.add_argument("--data-health-json", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-descriptor-accuracy", type=float, default=0.85)
    args = parser.parse_args()
    report = build_phase2av_holdout_failure_audit(
        holdout_postflight_json=args.holdout_postflight_json,
        prediction_json=args.prediction_json,
        head_rows_jsonl=args.head_rows_jsonl,
        data_health_jsons=args.data_health_json,
        min_command_slot_accuracy=args.min_command_slot_accuracy,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_descriptor_accuracy=args.min_descriptor_accuracy,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
