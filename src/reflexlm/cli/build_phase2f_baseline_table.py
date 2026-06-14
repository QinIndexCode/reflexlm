from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2d_gates import _metric, _trace_audit


DEFAULT_REPORT_DIR = Path("artifacts/reports/phase2f_rich_latent_fusion_canary")
DEFAULT_EXTERNAL_REPORT_DIR = Path("artifacts/reports/phase2g_external_trace_v1")
DEFAULT_EXTERNAL_V2_REPORT_DIR = Path("artifacts/reports/phase2g_external_trace_v2_semantic_required")
DEFAULT_EXTERNAL_V3_REPORT_DIR = Path("artifacts/reports/phase2i_external_trace_v3_semantic_required")
DEFAULT_ADAPTER = "phase2f_rich_latent_fusion_canary_r16_alpha32_lr1e-4_len256_cap2048"


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _completion_counts(payload: dict[str, Any]) -> str:
    value = payload.get("metrics", {}).get("aggregate", {}).get("task_completion_rate")
    positives = None
    if isinstance(value, dict):
        positives = value.get("positives")
    episodes = payload.get("episode_count")
    if isinstance(positives, int) and isinstance(episodes, int):
        return f"{positives}/{episodes}"
    completion = _metric(payload, "task_completion_rate")
    if completion is not None and isinstance(episodes, int):
        return f"{round(completion * episodes)}/{episodes}"
    return ""


def _round(value: float | None) -> float | None:
    return round(value, 6) if isinstance(value, float) else value


def _row(policy: str, dataset: str, path: str | Path, gate_status: str) -> dict[str, Any]:
    payload = _load(path)
    trace = _trace_audit(payload)
    return {
        "policy": policy,
        "dataset": dataset,
        "completion": _round(_metric(payload, "task_completion_rate")),
        "positives/episodes": _completion_counts(payload),
        "model_calls": _round(_metric(payload, "model_calls")),
        "token_equivalent_cost": _round(_metric(payload, "token_equivalent_cost")),
        "reaction_latency_ms": _round(_metric(payload, "reaction_latency_ms")),
        "state_hallucination": _round(_metric(payload, "state_hallucination_rate")),
        "false_reflex": _round(_metric(payload, "false_reflex_rate")),
        "allowlist_hallucination": _round(_metric(payload, "state_hallucination_rate")),
        "low_level_qwen_calls": trace["low_level_qwen_calls"],
        "cache_hits": trace["cache_hits"],
        "gate_status": gate_status,
        "eval_json": str(Path(path)),
        "run_path": payload.get("run_path"),
    }


def build_table(
    rows: list[tuple[str, str, str | Path, str]],
    *,
    table_family: str = "phase2f_exact_baseline_table",
) -> dict[str, Any]:
    table_rows = [_row(policy, dataset, path, gate_status) for policy, dataset, path, gate_status in rows]
    return {
        "table_family": table_family,
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
            "cache_hits",
            "gate_status",
        ],
        "rows": table_rows,
    }


def markdown_table(table: dict[str, Any]) -> str:
    columns = table["columns"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table["rows"]:
        values = []
        for column in columns:
            value = row.get(column)
            values.append("" if value is None else f"`{value}`")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def default_phase2f_rows(
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    *,
    adapter_name: str = DEFAULT_ADAPTER,
    native_package_label: str = "Phase2F native package",
) -> list[tuple[str, str, Path, str]]:
    report = Path(report_dir)
    return [
        ("prompt-only 7B", "quasi_real_terminal_v1", report / "prompt_only_7b.quasi_real_eval.json", "text_baseline"),
        ("ReAct 7B", "quasi_real_terminal_v1", report / "react_7b.quasi_real_eval.json", "text_baseline"),
        (
            "no-NSI latent",
            "debug_ood_v2",
            report / f"{adapter_name}.no_nsi_latent_eval.json",
            "mechanism_ablation",
        ),
        (
            "native-head-only",
            "debug_ood_v2",
            report / f"{adapter_name}.native_head_only_debug_ood_v2_eval.json",
            "mechanism_ablation",
        ),
        (
            "continuation-only",
            "debug_ood_v2",
            report / f"{adapter_name}.continuation_only_debug_ood_v2_eval.json",
            "mechanism_ablation",
        ),
        (
            native_package_label,
            "debug_ood_v2",
            report / f"{adapter_name}.debug_ood_v2_eval.json",
            "strong_pass",
        ),
        (
            native_package_label,
            "quasi_real_terminal_v1",
            report / f"{adapter_name}.quasi_real_eval.json",
            "strong_pass",
        ),
    ]


def default_external_trace_rows(
    report_dir: str | Path = DEFAULT_EXTERNAL_REPORT_DIR,
    *,
    dataset: str = "external_trace_v1",
    gate_status: str = "external_transfer_gate",
    adapter_name: str = DEFAULT_ADAPTER,
    native_package_label: str = "Phase2F native package",
) -> list[tuple[str, str, Path, str]]:
    report = Path(report_dir)
    suffix = f"{dataset}_eval.json"
    return [
        ("prompt-only 7B", dataset, report / f"prompt_only_7b.{suffix}", "text_baseline"),
        ("ReAct 7B", dataset, report / f"react_7b.{suffix}", "text_baseline"),
        (
            "no-NSI latent",
            dataset,
            report / f"{adapter_name}.no_nsi_latent.{suffix}",
            "mechanism_ablation",
        ),
        (
            "native-head-only",
            dataset,
            report / f"{adapter_name}.native_head_only.{suffix}",
            "mechanism_ablation",
        ),
        (
            "continuation-only",
            dataset,
            report / f"{adapter_name}.continuation_only.{suffix}",
            "mechanism_ablation",
        ),
        (
            native_package_label,
            dataset,
            report / f"{adapter_name}.{suffix}",
            gate_status,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact Phase2F baseline table from eval JSON artifacts.")
    parser.add_argument(
        "--table",
        choices=[
            "phase2f",
            "external_trace_v1",
            "external_trace_v2_semantic_required",
            "external_trace_v3_semantic_required",
        ],
        default="phase2f",
    )
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument(
        "--adapter-name",
        default=DEFAULT_ADAPTER,
        help="Adapter/report filename stem for native package and ablation eval JSONs.",
    )
    parser.add_argument(
        "--native-package-label",
        default="Phase2F native package",
        help="Policy label used for the full native package row.",
    )
    parser.add_argument("--output-json", default=str(DEFAULT_REPORT_DIR / "phase2f_exact_baseline_table.json"))
    parser.add_argument("--output-md", default=str(DEFAULT_REPORT_DIR / "phase2f_exact_baseline_table.md"))
    args = parser.parse_args()
    if args.table == "external_trace_v1":
        rows = default_external_trace_rows(
            args.report_dir,
            dataset="external_trace_v1",
            adapter_name=args.adapter_name,
            native_package_label=args.native_package_label,
        )
    elif args.table == "external_trace_v2_semantic_required":
        rows = default_external_trace_rows(
            args.report_dir,
            dataset="external_trace_v2_semantic_required",
            gate_status="semantic_required_transfer_gate",
            adapter_name=args.adapter_name,
            native_package_label=args.native_package_label,
        )
    elif args.table == "external_trace_v3_semantic_required":
        rows = default_external_trace_rows(
            args.report_dir,
            dataset="external_trace_v3_semantic_required",
            gate_status="semantic_required_transfer_gate",
            adapter_name=args.adapter_name,
            native_package_label=args.native_package_label,
        )
    else:
        rows = default_phase2f_rows(
            args.report_dir,
            adapter_name=args.adapter_name,
            native_package_label=args.native_package_label,
        )
    missing = [path for _, _, path, _ in rows if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing baseline eval artifacts: " + ", ".join(str(path) for path in missing)
        )
    table = build_table(
        rows,
        table_family=(
            "external_trace_v1_exact_baseline_table"
            if args.table == "external_trace_v1"
            else "external_trace_v2_semantic_required_exact_baseline_table"
            if args.table == "external_trace_v2_semantic_required"
            else "external_trace_v3_semantic_required_exact_baseline_table"
            if args.table == "external_trace_v3_semantic_required"
            else "phase2f_exact_baseline_table"
        ),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md = Path(args.output_md)
    output_md.write_text(markdown_table(table) + "\n", encoding="utf-8")
    print(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
