from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    _percentile,
    run_phase2bn_model_selected_sealed_runtime,
)


POLICY_MODES: tuple[dict[str, Any], ...] = (
    {
        "mode_id": "native_synaptic_runtime",
        "policy_label": "phase2br_native_synaptic_runtime",
        "authorize_bounded_debug_cortex_recovery": True,
        "use_synaptic_motor_plan": True,
    },
    {
        "mode_id": "no_debug_recovery_constraint",
        "policy_label": "phase2br_no_debug_recovery_constraint",
        "authorize_bounded_debug_cortex_recovery": False,
        "use_synaptic_motor_plan": True,
    },
    {
        "mode_id": "raw_motor_head",
        "policy_label": "phase2br_raw_motor_head",
        "authorize_bounded_debug_cortex_recovery": False,
        "use_synaptic_motor_plan": False,
    },
)


def _aggregate_subreports(subreports: list[dict[str, Any]]) -> dict[str, Any]:
    episode_reports = [
        episode
        for report in subreports
        for episode in report.get("episode_reports", [])
    ]
    decision_latencies_ms = [
        float(step["decision_latency_ms"])
        for episode in episode_reports
        for step in episode.get("policy_debug_steps", [])
        if "decision_latency_ms" in step
    ]
    episodes = len(episode_reports)
    failure_episodes = [row for row in episode_reports if row["requires_failure"]]
    task_completion_successes = sum(
        bool(row["task_completion_success"]) for row in episode_reports
    )
    failure_recoveries = sum(
        bool(row["recovery_success"]) for row in failure_episodes
    )
    token_equivalent_cost = sum(
        int(row.get("token_equivalent_cost", 0)) for row in episode_reports
    )
    model_calls = sum(int(row.get("model_calls", 0)) for row in episode_reports)
    return {
        "repositories": len(subreports),
        "episodes": episodes,
        "executed_actions": sum(
            int(report["metrics"]["executed_actions"]) for report in subreports
        ),
        "rejected_actions": sum(
            int(report["metrics"]["rejected_actions"]) for report in subreports
        ),
        "task_completion_successes": task_completion_successes,
        "task_completion_success_rate": task_completion_successes / max(episodes, 1),
        "failure_episodes": len(failure_episodes),
        "failure_recoveries": failure_recoveries,
        "failure_recovery_success_rate": failure_recoveries
        / max(len(failure_episodes), 1),
        "decision_count": len(decision_latencies_ms),
        "mean_decision_latency_ms": sum(decision_latencies_ms)
        / max(len(decision_latencies_ms), 1),
        "p50_decision_latency_ms": _percentile(decision_latencies_ms, 0.50),
        "p95_decision_latency_ms": _percentile(decision_latencies_ms, 0.95),
        "total_policy_compute_latency_ms": sum(decision_latencies_ms),
        "token_equivalent_cost": token_equivalent_cost,
        "mean_token_equivalent_cost_per_episode": token_equivalent_cost
        / max(episodes, 1),
        "model_calls": model_calls,
        "mean_model_calls_per_episode": model_calls / max(episodes, 1),
    }


def _mode_comparison(
    native_metrics: dict[str, Any],
    ablation_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_completion_rate_delta": float(
            native_metrics["task_completion_success_rate"]
            - ablation_metrics["task_completion_success_rate"]
        ),
        "failure_recovery_rate_delta": float(
            native_metrics["failure_recovery_success_rate"]
            - ablation_metrics["failure_recovery_success_rate"]
        ),
        "mean_decision_latency_ms_delta": float(
            native_metrics["mean_decision_latency_ms"]
            - ablation_metrics["mean_decision_latency_ms"]
        ),
        "mean_token_equivalent_cost_per_episode_delta": float(
            native_metrics["mean_token_equivalent_cost_per_episode"]
            - ablation_metrics["mean_token_equivalent_cost_per_episode"]
        ),
        "mean_model_calls_per_episode_delta": float(
            native_metrics["mean_model_calls_per_episode"]
            - ablation_metrics["mean_model_calls_per_episode"]
        ),
    }


def run_phase2br_runtime_ablation_comparison(
    *,
    checkpoint_path: str | Path,
    phase2bq_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
    max_extra_steps: int = 5,
) -> dict[str, Any]:
    phase2bq_report_path = Path(phase2bq_report_json)
    phase2bq_report = json.loads(
        phase2bq_report_path.read_text(encoding="utf-8-sig")
    )
    repository_rows = phase2bq_report.get("repository_reports")
    if not isinstance(repository_rows, list) or not repository_rows:
        raise ValueError("Phase2BR requires a Phase2BQ report with repository reports")
    output_root = Path(output_dir)

    mode_reports: dict[str, dict[str, Any]] = {}
    for mode in POLICY_MODES:
        mode_id = str(mode["mode_id"])
        subreports: list[dict[str, Any]] = []
        for repository in repository_rows:
            repository_id = str(repository["repository_id"])
            repository_output = output_root / mode_id / repository_id
            subreport = run_phase2bn_model_selected_sealed_runtime(
                checkpoint_path=checkpoint_path,
                manifest_json=repository["generated_manifest_json"],
                output_jsonl=repository_output / "trajectories.jsonl",
                output_report_json=repository_output / "report.json",
                timeout_seconds=timeout_seconds,
                max_extra_steps=max_extra_steps,
                policy_label=str(mode["policy_label"]),
                authorize_bounded_debug_cortex_recovery=bool(
                    mode["authorize_bounded_debug_cortex_recovery"]
                ),
                use_synaptic_motor_plan=bool(mode["use_synaptic_motor_plan"]),
            )
            subreports.append(subreport)
        mode_reports[mode_id] = {
            "configuration": dict(mode),
            "passed_all_phase2bn_gates": all(row["passed"] for row in subreports),
            "metrics": _aggregate_subreports(subreports),
            "repository_report_jsons": [
                str(output_root / mode_id / str(repository["repository_id"]) / "report.json")
                for repository in repository_rows
            ],
        }

    native_metrics = mode_reports["native_synaptic_runtime"]["metrics"]
    comparisons = {
        mode_id: _mode_comparison(native_metrics, row["metrics"])
        for mode_id, row in mode_reports.items()
        if mode_id != "native_synaptic_runtime"
    }
    mechanism_gap_observed = any(
        row["task_completion_rate_delta"] > 0.0
        or row["failure_recovery_rate_delta"] > 0.0
        for row in comparisons.values()
    )
    checks = {
        "phase2bq_source_report_passed": phase2bq_report.get("passed") is True,
        "same_generated_manifests_used_for_all_modes": True,
        "all_modes_executed_real_allowlisted_actions": all(
            row["metrics"]["executed_actions"] > 0 for row in mode_reports.values()
        ),
        "native_synaptic_runtime_completed_all_tasks": native_metrics[
            "task_completion_success_rate"
        ]
        == 1.0,
        "native_synaptic_runtime_recovered_all_failure_tasks": native_metrics[
            "failure_recovery_success_rate"
        ]
        == 1.0,
        "architecture_ablation_mechanism_gap_observed": mechanism_gap_observed,
        "prompt_only_baseline_measured": False,
        "react_baseline_measured": False,
    }
    bounded_mechanism_checks = {
        key: value
        for key, value in checks.items()
        if key not in {"prompt_only_baseline_measured", "react_baseline_measured"}
    }
    ready_for_bounded_mechanism_claim = all(bounded_mechanism_checks.values())
    ready_for_prompt_or_react_superiority_claim = (
        ready_for_bounded_mechanism_claim
        and checks["prompt_only_baseline_measured"]
        and checks["react_baseline_measured"]
    )
    report = {
        "artifact_family": "phase2br_runtime_ablation_comparison",
        "passed": ready_for_bounded_mechanism_claim,
        "ready_for_bounded_runtime_mechanism_claim": ready_for_bounded_mechanism_claim,
        "ready_for_prompt_or_react_superiority_claim": ready_for_prompt_or_react_superiority_claim,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "phase2bq_report_json": str(phase2bq_report_path),
        "checkpoint_path": str(Path(checkpoint_path)),
        "checks": checks,
        "mode_reports": mode_reports,
        "comparisons_against_native_synaptic_runtime": comparisons,
        "supported_claims": [
            "on the same bounded generated runtime contracts, the complete native synaptic runtime showed a measurable completion or failure-recovery advantage over at least one architecture ablation"
        ]
        if ready_for_bounded_mechanism_claim
        else [],
        "unsupported_claims": [
            "superiority over prompt-only or ReAct baselines",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bs_live_prompt_only_react_runtime_baselines"
            if ready_for_bounded_mechanism_claim
            else "repair_phase2br_runtime_ablation_comparison"
        ),
    }
    output_report = Path(output_report_json)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare native synaptic runtime behavior against same-model architecture ablations."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--phase2bq-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-extra-steps", type=int, default=5)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2br_runtime_ablation_comparison(
        checkpoint_path=args.checkpoint_path,
        phase2bq_report_json=args.phase2bq_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
        timeout_seconds=args.timeout_seconds,
        max_extra_steps=args.max_extra_steps,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
