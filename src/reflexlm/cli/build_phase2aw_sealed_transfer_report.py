from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.check_external_trace_gates import build_external_gate_report
from reflexlm.cli.check_phase2d_gates import _metric, _trace_audit


DATASET = "external_trace_v3_semantic_required"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _round(value: float | None) -> float | None:
    return round(value, 6) if isinstance(value, float) else value


def _completion_counts(payload: dict[str, Any]) -> str:
    metric = payload.get("metrics", {}).get("aggregate", {}).get("task_completion_rate")
    positives = metric.get("positives") if isinstance(metric, dict) else None
    episodes = payload.get("episode_count")
    if isinstance(positives, int) and isinstance(episodes, int):
        return f"{positives}/{episodes}"
    completion = _metric(payload, "task_completion_rate")
    if completion is not None and isinstance(episodes, int):
        return f"{round(completion * episodes)}/{episodes}"
    return ""


def _row(policy: str, path: str | Path, gate_status: str) -> dict[str, Any]:
    payload = _read_json(path)
    trace = _trace_audit(payload)
    return {
        "policy": policy,
        "dataset": DATASET,
        "completion": _round(_metric(payload, "task_completion_rate")),
        "positives/episodes": _completion_counts(payload),
        "model_calls": _round(_metric(payload, "model_calls")),
        "token_equivalent_cost": _round(_metric(payload, "token_equivalent_cost")),
        "reaction_latency_ms": _round(_metric(payload, "reaction_latency_ms")),
        "command_decision_accuracy": _round(_metric(payload, "command_decision_accuracy")),
        "read_file_decision_accuracy": _round(_metric(payload, "read_file_decision_accuracy")),
        "state_hallucination": _round(_metric(payload, "state_hallucination_rate")),
        "false_reflex": _round(_metric(payload, "false_reflex_rate")),
        "low_level_qwen_calls": trace["low_level_qwen_calls"],
        "debug_qwen_calls": trace["debug_qwen_calls"],
        "qwen_on_non_debug": trace["qwen_on_non_debug"],
        "cache_hits": trace["cache_hits"],
        "trace_rows": trace["trace_rows"],
        "gate_status": gate_status,
        "eval_json": str(Path(path)),
        "run_path": payload.get("run_path"),
    }


def _zero_classification(row: dict[str, Any]) -> dict[str, Any] | None:
    completion = row.get("completion")
    if completion != 0.0:
        return None
    policy = str(row["policy"])
    command_accuracy = row.get("command_decision_accuracy")
    read_accuracy = row.get("read_file_decision_accuracy")
    false_reflex = row.get("false_reflex")
    if "native-head-only" in policy:
        category = "expected_zero_due_to_missing_continuation_or_read_file_path"
        interpretation = (
            "Native heads are evaluable as a mechanism ablation, but this sealed "
            "profile requires package-level state coordination that the native-head-only "
            "control intentionally lacks."
        )
    elif "continuation-only" in policy:
        category = "expected_zero_due_to_missing_native_command_slot_selection"
        interpretation = (
            "Continuation memory is evaluable and reads the required state, but it "
            "cannot choose the semantic command slot without native heads."
        )
    elif "prompt-only" in policy or "ReAct" in policy:
        category = "valid_zero_failure"
        interpretation = (
            "The text-loop baseline executed the sealed task and failed without "
            "state hallucination; use as a text-baseline failure, not as proof of "
            "general production autonomy."
        )
    else:
        category = "suspicious_zero_requires_redesign"
        interpretation = "Zero completion is not explained by a preregistered control capability gap."
    return {
        "policy": policy,
        "category": category,
        "command_decision_accuracy": command_accuracy,
        "read_file_decision_accuracy": read_accuracy,
        "false_reflex": false_reflex,
        "interpretation": interpretation,
    }


def _table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "table_family": "phase2aw_external_trace_v3_semantic_required_exact_baseline_table",
        "columns": [
            "policy",
            "dataset",
            "completion",
            "positives/episodes",
            "model_calls",
            "token_equivalent_cost",
            "reaction_latency_ms",
            "command_decision_accuracy",
            "read_file_decision_accuracy",
            "state_hallucination",
            "false_reflex",
            "low_level_qwen_calls",
            "debug_qwen_calls",
            "qwen_on_non_debug",
            "cache_hits",
            "trace_rows",
            "gate_status",
        ],
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase2AW Sealed V3 Transfer Report",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Claim scope: `{report['claim_scope']}`",
        f"- Ready for strong architecture claim: `{report['ready_for_strong_architecture_claim']}`",
        "",
        "## Metrics",
    ]
    for key, value in report["metrics"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks"])
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    columns = report["baseline_table"]["columns"]
    lines.extend(["", "## Baseline Table"])
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in report["baseline_table"]["rows"]:
        values = ["" if row.get(column) is None else f"`{row.get(column)}`" for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Zero-Control Classifications"])
    for item in report["zero_control_classifications"]:
        lines.append(f"- `{item['policy']}`: `{item['category']}`")
    lines.extend(["", "## Supported Claims"])
    lines.extend(f"- {item}" for item in report["supported_claims"])
    lines.extend(["", "## Unsupported Claims"])
    lines.extend(f"- {item}" for item in report["unsupported_claims"])
    lines.extend(["", "## Notes"])
    lines.extend(f"- {item}" for item in report["notes"])
    return "\n".join(lines) + "\n"


def build_phase2aw_sealed_transfer_report(
    *,
    full_eval_json: str | Path,
    prompt_eval_json: str | Path,
    react_eval_json: str | Path,
    no_nsi_eval_json: str | Path,
    native_head_only_eval_json: str | Path,
    continuation_only_eval_json: str | Path,
    dataset_manifest_json: str | Path,
    sealed_authorization_gate_json: str | Path,
    package_loaded_evidence_json: str | Path,
) -> dict[str, Any]:
    sealed_authorization = _read_json(sealed_authorization_gate_json)
    package_loaded_evidence = _read_json(package_loaded_evidence_json)
    generic_gate = build_external_gate_report(
        full_eval_json=full_eval_json,
        prompt_eval_json=prompt_eval_json,
        react_eval_json=react_eval_json,
        no_nsi_eval_json=no_nsi_eval_json,
        native_head_only_eval_json=native_head_only_eval_json,
        continuation_only_eval_json=continuation_only_eval_json,
        dataset_manifest_json=dataset_manifest_json,
    )
    rows = [
        _row("Phase2AW full package", full_eval_json, "sealed_transfer_candidate"),
        _row(
            "no-NSI latent + no candidate_identity",
            no_nsi_eval_json,
            "mechanism_ablation_nonzero_control",
        ),
        _row("native-head-only/no-cache", native_head_only_eval_json, "mechanism_ablation_zero_control"),
        _row(
            "continuation-only/no-native-heads",
            continuation_only_eval_json,
            "mechanism_ablation_zero_control",
        ),
        _row("prompt-only 7B", prompt_eval_json, "text_baseline_zero_control"),
        _row("ReAct 7B", react_eval_json, "text_baseline_zero_control"),
    ]
    full_row = rows[0]
    zero_classifications = [
        classification
        for row in rows[1:]
        if (classification := _zero_classification(row)) is not None
    ]
    suspicious_zero = any(
        item["category"] == "suspicious_zero_requires_redesign"
        for item in zero_classifications
    )
    nonzero_controls = [
        row["policy"]
        for row in rows[1:]
        if isinstance(row.get("completion"), float) and row["completion"] > 0.0
    ]
    metrics = generic_gate["metrics"]
    checks = {
        **generic_gate["checks"],
        "package_loaded_nonsealed_evidence_passed": package_loaded_evidence.get("passed")
        is True,
        "sealed_authorization_passed": sealed_authorization.get("passed") is True,
        "sealed_authorization_allows_eval": sealed_authorization.get("ready_for_sealed_eval")
        is True,
        "sealed_authorization_does_not_allow_claim_upgrade": sealed_authorization.get(
            "ready_for_claim_upgrade"
        )
        is False,
        "full_package_loaded_label": str(full_row["policy"]).startswith("Phase2AW full"),
        "full_minus_no_nsi_ge_0_15": metrics.get("full_minus_no_nsi") is not None
        and metrics["full_minus_no_nsi"] >= 0.15,
        "full_minus_native_head_only_ge_0_10": metrics.get("full_minus_native_head_only")
        is not None
        and metrics["full_minus_native_head_only"] >= 0.10,
        "full_minus_continuation_only_ge_0_15": metrics.get("full_minus_continuation_only")
        is not None
        and metrics["full_minus_continuation_only"] >= 0.15,
        "at_least_one_nonzero_control": bool(nonzero_controls),
        "no_suspicious_unexplained_zero": not suspicious_zero,
        "full_low_level_qwen_calls_zero": full_row["low_level_qwen_calls"] == 0,
        "full_qwen_on_non_debug_zero": full_row["qwen_on_non_debug"] == 0,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2aw_sealed_transfer_report",
        "passed": passed,
        "claim_scope": (
            "phase2aw_bounded_sealed_v3_package_transfer_positive"
            if passed
            else "phase2aw_sealed_v3_transfer_incomplete_or_failed"
        ),
        "ready_for_strong_architecture_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "ready_for_production_autonomy_claim": False,
        "checks": checks,
        "metrics": {
            **metrics,
            "full_low_level_qwen_calls": full_row["low_level_qwen_calls"],
            "full_debug_qwen_calls": full_row["debug_qwen_calls"],
            "full_qwen_on_non_debug": full_row["qwen_on_non_debug"],
            "nonzero_control_count": len(nonzero_controls),
            "zero_control_count": len(zero_classifications),
        },
        "baseline_table": _table(rows),
        "zero_control_classifications": zero_classifications,
        "nonzero_controls": nonzero_controls,
        "generic_external_trace_gate": generic_gate,
        "supported_claims": [
            "Phase2AW full package transfers on sealed v3 under final-evaluation-only rules",
            "Phase2AW full package beats no-NSI, native-head-only, continuation-only, prompt-only, and ReAct on this sealed v3 evaluation",
            "The no-NSI control is nonzero, so this sealed table is not an all-zero-control field",
            "No sealed failure is authorized as training, sampling, tuning, or data-design feedback",
        ]
        if passed
        else [],
        "unsupported_claims": [
            "epoch-making architecture is not proven by a single sealed 7B run",
            "production autonomy is not proven",
            "open-ended debugging generalization is not proven",
            "freeform patch generation is not proven",
            "cross-model or multi-seed transfer is not proven",
            "zero native-head-only/continuation-only/text controls must not be used as standalone proof without the recorded classifications",
        ],
        "notes": [
            "This report is an evidence gate, not a training signal.",
            "The sealed v3 task family is test_failure_reflex only; broader task-family generalization remains unproven.",
            "The no-NSI ablation is the only nonzero non-full control here; stronger graded claims still require non-sealed graded controls and cross-model/seed reproduction.",
        ],
        "inputs": {
            "full_eval_json": str(Path(full_eval_json)),
            "prompt_eval_json": str(Path(prompt_eval_json)),
            "react_eval_json": str(Path(react_eval_json)),
            "no_nsi_eval_json": str(Path(no_nsi_eval_json)),
            "native_head_only_eval_json": str(Path(native_head_only_eval_json)),
            "continuation_only_eval_json": str(Path(continuation_only_eval_json)),
            "dataset_manifest_json": str(Path(dataset_manifest_json)),
            "sealed_authorization_gate_json": str(Path(sealed_authorization_gate_json)),
            "package_loaded_evidence_json": str(Path(package_loaded_evidence_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AW sealed-v3 transfer table and gate.")
    parser.add_argument("--full-eval-json", required=True)
    parser.add_argument("--prompt-eval-json", required=True)
    parser.add_argument("--react-eval-json", required=True)
    parser.add_argument("--no-nsi-eval-json", required=True)
    parser.add_argument("--native-head-only-eval-json", required=True)
    parser.add_argument("--continuation-only-eval-json", required=True)
    parser.add_argument("--dataset-manifest-json", required=True)
    parser.add_argument("--sealed-authorization-gate-json", required=True)
    parser.add_argument("--package-loaded-evidence-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2aw_sealed_transfer_report(
        full_eval_json=args.full_eval_json,
        prompt_eval_json=args.prompt_eval_json,
        react_eval_json=args.react_eval_json,
        no_nsi_eval_json=args.no_nsi_eval_json,
        native_head_only_eval_json=args.native_head_only_eval_json,
        continuation_only_eval_json=args.continuation_only_eval_json,
        dataset_manifest_json=args.dataset_manifest_json,
        sealed_authorization_gate_json=args.sealed_authorization_gate_json,
        package_loaded_evidence_json=args.package_loaded_evidence_json,
    )
    _write_json(args.output_json, report)
    if args.output_md:
        Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
