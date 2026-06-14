from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.experiment import create_experiment_run
from reflexlm.reporting import (
    METRIC_ORDER,
    bootstrap_paired_difference,
    load_episode_results,
)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _ordered_labels(
    summaries: list[dict[str, Any]],
    explicit_order: list[str] | None,
) -> list[str]:
    inferred = [summary["policy"]["policy_label"] for summary in summaries]
    if not explicit_order:
        return inferred
    ordered = [label for label in explicit_order if label in inferred]
    for label in inferred:
        if label not in ordered:
            ordered.append(label)
    return ordered


def _format_metric_cell(metric_summary: dict[str, Any] | None) -> str:
    if metric_summary is None:
        return "NA"
    mean = metric_summary.get("mean")
    ci95 = metric_summary.get("ci95")
    if mean is None or ci95 is None:
        return "NA"
    return f"`{mean:.3f}` `[{ci95[0]:.3f}, {ci95[1]:.3f}]`"


def _format_scalar_cell(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"`{value:.3f}`"


def _render_main_results_markdown(
    *,
    labels: list[str],
    main_table: dict[str, Any],
) -> str:
    header = "| Metric | " + " | ".join(labels) + " |"
    divider = "|---|" + "|".join("---" for _ in labels) + "|"
    rows = [header, divider]
    metric_labels = {
        "reaction_latency_ms": "Reaction latency (ms)",
        "token_equivalent_cost": "Token-equivalent cost",
        "model_calls": "Model calls",
        "recovery_success_rate": "Recovery success rate",
        "false_reflex_rate": "False reflex rate",
        "dangerous_action_block_rate": "Dangerous action block rate",
        "long_run_stability": "Long-run stability",
        "state_hallucination_rate": "State hallucination rate",
        "stale_state_action_rate": "Stale-state action rate",
        "task_completion_rate": "Task completion rate",
    }
    for metric in METRIC_ORDER:
        cells = [_format_metric_cell(main_table[label].get(metric)) for label in labels]
        rows.append(f"| {metric_labels.get(metric, metric)} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def _render_task_completion_markdown(
    *,
    labels: list[str],
    per_task_table: dict[str, Any],
) -> str:
    first_label = labels[0]
    task_names = list(per_task_table[first_label].keys())
    header = "| Task family | " + " | ".join(labels) + " |"
    divider = "|---|" + "|".join("---" for _ in labels) + "|"
    rows = [header, divider]
    for task_name in task_names:
        cells = []
        for label in labels:
            metric_summary = (
                per_task_table[label]
                .get(task_name, {})
                .get("metrics", {})
                .get("task_completion_rate")
            )
            cells.append(_format_scalar_cell(None if metric_summary is None else metric_summary.get("mean")))
        rows.append(f"| {task_name} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple evaluation run directories.")
    parser.add_argument("--run-dir", action="append", required=True, help="Evaluation run directory")
    parser.add_argument("--reference-label", required=True, help="Policy label used as the reference")
    parser.add_argument("--label-order", action="append", help="Optional column order for markdown tables")
    parser.add_argument("--run-name", default="phase1_comparison")
    parser.add_argument("--run-root", help="Optional run root directory")
    parser.add_argument("--output-json", help="Optional path to write the comparison JSON")
    parser.add_argument("--output-markdown-dir", help="Optional directory to mirror markdown table outputs")
    args = parser.parse_args()

    run_dirs = [Path(item) for item in args.run_dir]
    summaries = [_load_summary(run_dir) for run_dir in run_dirs]
    labels = _ordered_labels(summaries, args.label_order)
    run = create_experiment_run(
        kind="comparison",
        name=args.run_name,
        config={
            "run_dirs": [str(path.resolve()) for path in run_dirs],
            "reference_label": args.reference_label,
            "label_order": labels,
        },
        run_root=args.run_root,
    )
    runs_by_label = {summary["policy"]["policy_label"]: summary for summary in summaries}
    if args.reference_label not in runs_by_label:
        raise ValueError(f"Reference label {args.reference_label!r} not found in provided runs")
    reference_run_dir = run_dirs[summaries.index(runs_by_label[args.reference_label])]
    reference_rows = load_episode_results(reference_run_dir / "episode_results.jsonl")
    main_table: dict[str, Any] = {}
    per_task_table: dict[str, Any] = {}
    pairwise: dict[str, Any] = {}
    for summary, run_dir in zip(summaries, run_dirs, strict=True):
        label = summary["policy"]["policy_label"]
        main_table[label] = summary["metrics"]["aggregate"]
        per_task_table[label] = summary["metrics"]["per_task"]
        rows = load_episode_results(run_dir / "episode_results.jsonl")
        pairwise[label] = {
            metric: bootstrap_paired_difference(reference_rows, rows, metric)
            for metric in METRIC_ORDER
        }
    markdown_tables = {
        "main_results": _render_main_results_markdown(labels=labels, main_table=main_table),
        "task_completion_by_task": _render_task_completion_markdown(
            labels=labels,
            per_task_table=per_task_table,
        ),
    }
    payload = {
        "reference_label": args.reference_label,
        "label_order": labels,
        "runs": {
            summary["policy"]["policy_label"]: {
                "run_dir": str(run_dir.resolve()),
                "summary_path": str((run_dir / "summary.json").resolve()),
            }
            for summary, run_dir in zip(summaries, run_dirs, strict=True)
        },
        "main_table": main_table,
        "per_task_table": per_task_table,
        "pairwise_vs_reference": pairwise,
        "markdown_tables": markdown_tables,
    }
    payload["run_manifest"] = run.finalize({"reference_label": args.reference_label})
    run.write_json("comparison.json", payload)
    run.write_text("main_results.md", markdown_tables["main_results"])
    run.write_text("task_completion_by_task.md", markdown_tables["task_completion_by_task"])
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_markdown_dir:
        output_dir = Path(args.output_markdown_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "main_results.md").write_text(markdown_tables["main_results"], encoding="utf-8")
        (output_dir / "task_completion_by_task.md").write_text(
            markdown_tables["task_completion_by_task"],
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
