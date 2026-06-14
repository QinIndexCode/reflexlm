from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2d_gates import _metric, _trace_audit


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


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
        "completion": _round(_metric(payload, "task_completion_rate")),
        "positives/episodes": _completion_counts(payload),
        "model_calls": _round(_metric(payload, "model_calls")),
        "state_hallucination": _round(_metric(payload, "state_hallucination_rate")),
        "low_level_qwen_calls": trace["low_level_qwen_calls"],
        "debug_qwen_calls": trace["debug_qwen_calls"],
        "qwen_on_non_debug": trace["qwen_on_non_debug"],
        "cache_hits": trace["cache_hits"],
        "trace_rows": trace["trace_rows"],
        "gate_status": gate_status,
        "eval_json": str(Path(path)),
        "run_path": payload.get("run_path"),
    }


def build_phase2u_sealed_transfer_report(
    *,
    external_gate_json: str | Path,
    zero_baseline_audit_json: str | Path,
    postpackage_gate_json: str | Path,
    full_eval_json: str | Path,
    prompt_eval_json: str | Path,
    react_eval_json: str | Path,
    no_nsi_eval_json: str | Path,
    native_head_only_eval_json: str | Path,
    continuation_only_eval_json: str | Path,
) -> dict[str, Any]:
    external_gate = _read_json(external_gate_json)
    zero_audit = _read_json(zero_baseline_audit_json)
    postpackage = _read_json(postpackage_gate_json)
    checks = {
        "postpackage_gate_passed": postpackage.get("passed") is True,
        "postpackage_no_claim_upgrade": postpackage.get("ready_for_claim_upgrade") is False,
        "external_gate_passed": external_gate.get("passed") is True,
        "zero_baseline_audit_passed": zero_audit.get("passed") is True,
        "zero_baseline_bounded_only": zero_audit.get("ready_for_bounded_sealed_claim")
        is True
        and zero_audit.get("ready_for_strong_architecture_claim") is False,
        "no_nsi_candidate_identity_disabled": postpackage.get("checks", {}).get(
            "no_nsi_control_disables_candidate_identity"
        )
        is True,
        "all_zero_controls_classified": zero_audit.get("checks", {}).get(
            "all_zero_controls_classified"
        )
        is True,
        "no_suspicious_unexplained_zero": zero_audit.get("checks", {}).get(
            "no_suspicious_unexplained_zero"
        )
        is True,
    }
    rows = [
        _row("Phase2U full package", full_eval_json, "sealed_transfer_candidate"),
        _row("no-NSI latent + no candidate_identity", no_nsi_eval_json, "mechanism_ablation"),
        _row("native-head-only/no-cache", native_head_only_eval_json, "mechanism_ablation"),
        _row("continuation-only/no-native-heads", continuation_only_eval_json, "mechanism_ablation"),
        _row("prompt-only 7B", prompt_eval_json, "text_baseline_reused_fixed"),
        _row("ReAct 7B", react_eval_json, "text_baseline_reused_fixed"),
    ]
    passed = all(checks.values())
    all_controls_zero = bool(zero_audit.get("all_controls_zero"))
    mechanism_sufficiency_passed = passed and not all_controls_zero
    return {
        "artifact_family": "phase2u_sealed_transfer_report",
        "passed": passed,
        "claim_scope": (
            "phase2u_extreme_sealed_transfer_positive_but_not_mechanism_sufficient"
            if passed and all_controls_zero
            else "phase2u_bounded_sealed_transfer_supported"
            if passed
            else "phase2u_sealed_transfer_not_supported"
        ),
        "mechanism_sufficiency_passed": mechanism_sufficiency_passed,
        "checks": checks,
        "metrics": external_gate.get("metrics", {}),
        "baseline_table": {
            "table_family": "phase2u_external_trace_v3_semantic_required_exact_baseline_table",
            "columns": [
                "policy",
                "completion",
                "positives/episodes",
                "model_calls",
                "state_hallucination",
                "low_level_qwen_calls",
                "debug_qwen_calls",
                "qwen_on_non_debug",
                "cache_hits",
                "trace_rows",
                "gate_status",
            ],
            "rows": rows,
        },
        "zero_baseline_interpretation": zero_audit.get("interpretation", {}),
        "supported_claims": (
            [
                "Phase2U full package passes an extreme sealed-v3 semantic-required stress evaluation"
            ]
            if passed and all_controls_zero
            else [
                "Phase2U supports a bounded sealed-v3 semantic-required package transfer result"
            ]
            if passed
            else []
        ),
        "unsupported_claims": [
            "sealed-v3 all-zero controls do not prove a graded mechanism curve",
            "all-zero sealed controls are not sufficient for a strong architecture claim",
            "production autonomy is not proven",
            "open-ended debugging generalization is not proven",
            "epoch-making architecture claim is not proven",
            "sealed failures must not feed back into training, sampling, or tuning",
        ],
        "notes": [
            "Prompt-only and ReAct rows reuse fixed sealed-v3 text baselines because they are package-independent controls.",
            "The no-NSI control removes both latent contribution and the candidate_identity side channel.",
            "The zero-baseline audit is mandatory because all sealed control task-completion rates are zero.",
            "Mechanism sufficiency must be argued from a separate graded non-sealed sanity benchmark where controls have nonzero scores.",
        ],
        "inputs": {
            "external_gate_json": str(Path(external_gate_json)),
            "zero_baseline_audit_json": str(Path(zero_baseline_audit_json)),
            "postpackage_gate_json": str(Path(postpackage_gate_json)),
            "full_eval_json": str(Path(full_eval_json)),
            "prompt_eval_json": str(Path(prompt_eval_json)),
            "react_eval_json": str(Path(react_eval_json)),
            "no_nsi_eval_json": str(Path(no_nsi_eval_json)),
            "native_head_only_eval_json": str(Path(native_head_only_eval_json)),
            "continuation_only_eval_json": str(Path(continuation_only_eval_json)),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase2U Sealed V3 Transfer Report",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Claim scope: `{report['claim_scope']}`",
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
        lines.append("| " + " | ".join(f"`{row.get(column)}`" for column in columns) + " |")
    lines.extend(["", "## Supported Claims"])
    lines.extend(f"- {claim}" for claim in report["supported_claims"])
    lines.extend(["", "## Unsupported Claims"])
    lines.extend(f"- {claim}" for claim in report["unsupported_claims"])
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in report["notes"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2U sealed-v3 transfer report.")
    parser.add_argument("--external-gate-json", required=True)
    parser.add_argument("--zero-baseline-audit-json", required=True)
    parser.add_argument("--postpackage-gate-json", required=True)
    parser.add_argument("--full-eval-json", required=True)
    parser.add_argument("--prompt-eval-json", required=True)
    parser.add_argument("--react-eval-json", required=True)
    parser.add_argument("--no-nsi-eval-json", required=True)
    parser.add_argument("--native-head-only-eval-json", required=True)
    parser.add_argument("--continuation-only-eval-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2u_sealed_transfer_report(
        external_gate_json=args.external_gate_json,
        zero_baseline_audit_json=args.zero_baseline_audit_json,
        postpackage_gate_json=args.postpackage_gate_json,
        full_eval_json=args.full_eval_json,
        prompt_eval_json=args.prompt_eval_json,
        react_eval_json=args.react_eval_json,
        no_nsi_eval_json=args.no_nsi_eval_json,
        native_head_only_eval_json=args.native_head_only_eval_json,
        continuation_only_eval_json=args.continuation_only_eval_json,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
