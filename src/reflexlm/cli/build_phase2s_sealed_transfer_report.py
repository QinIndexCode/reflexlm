from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.check_external_trace_gates import build_external_gate_report
from reflexlm.cli.check_phase2d_gates import _metric, _trace_audit


DATASET = "external_trace_v3_semantic_required"


def _load(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _round(value: float | None) -> float | None:
    return round(value, 6) if isinstance(value, float) else value


def _completion_counts(payload: dict[str, Any]) -> str:
    value = payload.get("metrics", {}).get("aggregate", {}).get("task_completion_rate")
    positives = value.get("positives") if isinstance(value, dict) else None
    episodes = payload.get("episode_count")
    if isinstance(positives, int) and isinstance(episodes, int):
        return f"{positives}/{episodes}"
    completion = _metric(payload, "task_completion_rate")
    if completion is not None and isinstance(episodes, int):
        return f"{round(completion * episodes)}/{episodes}"
    return ""


def _row(policy: str, path: str | Path, gate_status: str) -> dict[str, Any]:
    payload = _load(path)
    if payload is None:
        raise FileNotFoundError(path)
    trace = _trace_audit(payload)
    return {
        "policy": policy,
        "dataset": DATASET,
        "completion": _round(_metric(payload, "task_completion_rate")),
        "positives/episodes": _completion_counts(payload),
        "model_calls": _round(_metric(payload, "model_calls")),
        "token_equivalent_cost": _round(_metric(payload, "token_equivalent_cost")),
        "reaction_latency_ms": _round(_metric(payload, "reaction_latency_ms")),
        "state_hallucination": _round(_metric(payload, "state_hallucination_rate")),
        "false_reflex": _round(_metric(payload, "false_reflex_rate")),
        "allowlist_hallucination": _round(_metric(payload, "state_hallucination_rate")),
        "low_level_qwen_calls": trace["low_level_qwen_calls"],
        "debug_qwen_calls": trace["debug_qwen_calls"],
        "qwen_on_non_debug": trace["qwen_on_non_debug"],
        "cache_hits": trace["cache_hits"],
        "gate_status": gate_status,
        "eval_json": str(Path(path)),
        "run_path": payload.get("run_path"),
    }


def _table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "table_family": "phase2s_external_trace_v3_semantic_required_exact_baseline_table",
        "columns": [
            "policy",
            "dataset",
            "completion",
            "positives/episodes",
            "model_calls",
            "token_equivalent_cost",
            "reaction_latency_ms",
            "state_hallucination",
            "false_reflex",
            "allowlist_hallucination",
            "low_level_qwen_calls",
            "debug_qwen_calls",
            "qwen_on_non_debug",
            "cache_hits",
            "gate_status",
        ],
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    table = report["baseline_table"]
    columns = table["columns"]
    lines = [
        "# Phase2S Sealed V3 Transfer Report",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Claim scope: `{report['claim_scope']}`",
        f"- Boundary review passed: `{report['checks']['boundary_review_passed']}`",
        f"- Sealed evaluation only: `{report['checks']['sealed_eval_only_after_boundary']}`",
        "",
        "## Gate Metrics",
    ]
    for key, value in report["metrics"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks"])
    for key, value in report["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Baseline Table"])
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in table["rows"]:
        values = ["" if row.get(column) is None else f"`{row.get(column)}`" for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Supported Claims"])
    lines.extend(f"- {item}" for item in report["supported_claims"])
    lines.extend(["", "## Unsupported Claims"])
    lines.extend(f"- {item}" for item in report["unsupported_claims"])
    lines.extend(["", "## Notes"])
    lines.extend(f"- {item}" for item in report["notes"])
    return "\n".join(lines) + "\n"


def build_phase2s_sealed_transfer_report(
    *,
    full_eval_json: str | Path,
    prompt_eval_json: str | Path,
    react_eval_json: str | Path,
    no_nsi_eval_json: str | Path,
    native_head_only_eval_json: str | Path,
    continuation_only_eval_json: str | Path,
    dataset_manifest_json: str | Path,
    boundary_review_json: str | Path | None = None,
) -> dict[str, Any]:
    boundary = _load(boundary_review_json)
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
        _row("Phase2S full package", full_eval_json, "sealed_transfer_candidate"),
        _row("no-NSI latent", no_nsi_eval_json, "mechanism_ablation"),
        _row("native-head-only", native_head_only_eval_json, "mechanism_ablation"),
        _row("continuation-only", continuation_only_eval_json, "mechanism_ablation"),
        _row("prompt-only 7B", prompt_eval_json, "text_baseline"),
        _row("ReAct 7B", react_eval_json, "text_baseline"),
    ]
    metrics = generic_gate["metrics"]
    full_row = rows[0]
    boundary_passed = boundary is not None and boundary.get("passed") is True
    phase2s_checks = {
        "boundary_review_passed": boundary_passed,
        "sealed_eval_only_after_boundary": boundary_passed
        and (
            boundary.get("allowed_next_action")
            == "package_for_final_eval_only_after_manual_boundary_acceptance"
        ),
        "full_completion_ge_0_90": metrics.get("full_completion") is not None
        and metrics["full_completion"] >= 0.90,
        "full_minus_continuation_only_ge_0_15": metrics.get(
            "full_minus_continuation_only"
        )
        is not None
        and metrics["full_minus_continuation_only"] >= 0.15,
        "full_minus_no_nsi_ge_0_15": metrics.get("full_minus_no_nsi") is not None
        and metrics["full_minus_no_nsi"] >= 0.15,
        "full_minus_native_head_only_ge_0_10": metrics.get(
            "full_minus_native_head_only"
        )
        is not None
        and metrics["full_minus_native_head_only"] >= 0.10,
        "allowlist_hallucination_zero": metrics.get("full_state_hallucination") == 0.0,
        "low_level_qwen_calls_zero": full_row["low_level_qwen_calls"] == 0,
        "qwen_on_non_debug_zero": full_row["qwen_on_non_debug"] == 0,
    }
    passed = bool(generic_gate["passed"] and all(phase2s_checks.values()))
    supported_claims = [
        "Phase2S full package transfers on sealed v3 under final-eval-only rules"
        if passed
        else "No upgraded Phase2S sealed-transfer claim is supported unless this report passes",
        "Phase2S full package beats no-NSI, native-head-only, continuation-only, prompt-only, and ReAct baselines on this sealed v3 evaluation"
        if passed
        else "Delta claims remain bounded by the failed checks",
        "Sealed v3 results are recorded as final evaluation evidence only, not training or tuning feedback"
        if phase2s_checks["sealed_eval_only_after_boundary"]
        else "Sealed evaluation-only boundary is not established",
    ]
    unsupported_claims = [
        "cross-model or multi-seed transfer is not proven by this single 7B package",
        "production autonomy and open-ended debugging generalization are not proven",
        "generation-free execution is not proven because the full package still records debug Qwen calls",
    ]
    return {
        "artifact_family": "phase2s_sealed_transfer_report",
        "passed": passed,
        "claim_scope": (
            "phase2s_sealed_v3_transfer_positive_bounded"
            if passed
            else "phase2s_sealed_v3_transfer_not_proven"
        ),
        "checks": {**generic_gate["checks"], **phase2s_checks},
        "metrics": {
            **metrics,
            "full_low_level_qwen_calls": full_row["low_level_qwen_calls"],
            "full_debug_qwen_calls": full_row["debug_qwen_calls"],
            "full_qwen_on_non_debug": full_row["qwen_on_non_debug"],
        },
        "baseline_table": _table(rows),
        "generic_external_trace_gate": generic_gate,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "notes": [
            "debug_qwen_calls are reported separately from low_level_qwen_calls; low-level calls exclude test_failure_reflex debug-cortex calls.",
            "A passing sealed report does not authorize sealed-tuned data design.",
            "Stronger architecture claims still require cross-model and multi-seed reproduction.",
        ],
        "inputs": {
            "full_eval_json": str(Path(full_eval_json)),
            "prompt_eval_json": str(Path(prompt_eval_json)),
            "react_eval_json": str(Path(react_eval_json)),
            "no_nsi_eval_json": str(Path(no_nsi_eval_json)),
            "native_head_only_eval_json": str(Path(native_head_only_eval_json)),
            "continuation_only_eval_json": str(Path(continuation_only_eval_json)),
            "dataset_manifest_json": str(Path(dataset_manifest_json)),
            "boundary_review_json": str(Path(boundary_review_json))
            if boundary_review_json
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2S sealed-v3 transfer table and gate.")
    parser.add_argument("--full-eval-json", required=True)
    parser.add_argument("--prompt-eval-json", required=True)
    parser.add_argument("--react-eval-json", required=True)
    parser.add_argument("--no-nsi-eval-json", required=True)
    parser.add_argument("--native-head-only-eval-json", required=True)
    parser.add_argument("--continuation-only-eval-json", required=True)
    parser.add_argument("--dataset-manifest-json", required=True)
    parser.add_argument("--boundary-review-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2s_sealed_transfer_report(
        full_eval_json=args.full_eval_json,
        prompt_eval_json=args.prompt_eval_json,
        react_eval_json=args.react_eval_json,
        no_nsi_eval_json=args.no_nsi_eval_json,
        native_head_only_eval_json=args.native_head_only_eval_json,
        continuation_only_eval_json=args.continuation_only_eval_json,
        dataset_manifest_json=args.dataset_manifest_json,
        boundary_review_json=args.boundary_review_json,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
