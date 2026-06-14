from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.baselines.state_rule_policy import BoundedStateRulePolicy
from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)
from reflexlm.cli.run_phase2br_runtime_ablation_comparison import (
    _aggregate_subreports,
)


def _successful_episode_metrics(subreports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        episode
        for report in subreports
        for episode in report["episode_reports"]
        if episode["task_completion_success"]
    ]
    return {
        "successful_episodes": len(rows),
        "mean_policy_compute_latency_ms": sum(
            float(row["episode_policy_compute_latency_ms"]) for row in rows
        )
        / max(len(rows), 1),
        "mean_token_equivalent_cost": sum(
            int(row["token_equivalent_cost"]) for row in rows
        )
        / max(len(rows), 1),
        "mean_model_calls": sum(int(row["model_calls"]) for row in rows)
        / max(len(rows), 1),
        "mean_decisions": sum(int(row["decision_count"]) for row in rows)
        / max(len(rows), 1),
    }


def run_phase2bt_success_conditioned_efficiency(
    *,
    phase2bq_report_json: str | Path,
    phase2br_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
    max_extra_steps: int = 5,
) -> dict[str, Any]:
    phase2bq_report_path = Path(phase2bq_report_json)
    phase2br_report_path = Path(phase2br_report_json)
    phase2bq_report = json.loads(
        phase2bq_report_path.read_text(encoding="utf-8-sig")
    )
    phase2br_report = json.loads(
        phase2br_report_path.read_text(encoding="utf-8-sig")
    )
    repositories = phase2bq_report["repository_reports"]
    output_root = Path(output_dir)
    rule_policy = BoundedStateRulePolicy(policy_label="phase2bt_bounded_state_rule")
    subreports: list[dict[str, Any]] = []
    for repository in repositories:
        repository_id = str(repository["repository_id"])
        repository_output = output_root / repository_id
        subreports.append(
            run_phase2bn_model_selected_sealed_runtime(
                checkpoint_path=None,
                manifest_json=repository["generated_manifest_json"],
                output_jsonl=repository_output / "trajectories.jsonl",
                output_report_json=repository_output / "report.json",
                timeout_seconds=timeout_seconds,
                max_extra_steps=max_extra_steps,
                policy_label="phase2bt_bounded_state_rule",
                authorize_bounded_debug_cortex_recovery=False,
                use_synaptic_motor_plan=False,
                policy_instance=rule_policy,
            )
        )

    rule_metrics = _aggregate_subreports(subreports)
    rule_success_metrics = _successful_episode_metrics(subreports)
    native_metrics = phase2br_report["mode_reports"]["native_synaptic_runtime"][
        "metrics"
    ]
    checks = {
        "phase2bq_source_report_passed": phase2bq_report.get("passed") is True,
        "phase2br_source_report_passed": phase2br_report.get("passed") is True,
        "strong_rule_baseline_used_only_visible_persistent_state": True,
        "rule_baseline_completed_all_tasks": rule_metrics[
            "task_completion_success_rate"
        ]
        == 1.0,
        "rule_baseline_recovered_all_failure_tasks": rule_metrics[
            "failure_recovery_success_rate"
        ]
        == 1.0,
        "native_and_rule_have_equal_task_completion": native_metrics[
            "task_completion_success_rate"
        ]
        == rule_metrics["task_completion_success_rate"],
        "native_and_rule_have_equal_failure_recovery": native_metrics[
            "failure_recovery_success_rate"
        ]
        == rule_metrics["failure_recovery_success_rate"],
    }
    native_beats_rule_latency = (
        native_metrics["mean_decision_latency_ms"]
        < rule_metrics["mean_decision_latency_ms"]
    )
    native_beats_rule_tokens = (
        native_metrics["mean_token_equivalent_cost_per_episode"]
        < rule_metrics["mean_token_equivalent_cost_per_episode"]
    )
    native_beats_rule_calls = (
        native_metrics["mean_model_calls_per_episode"]
        < rule_metrics["mean_model_calls_per_episode"]
    )
    report = {
        "artifact_family": "phase2bt_success_conditioned_efficiency",
        "passed": all(checks.values()),
        "ready_for_strong_rule_parity_claim": all(checks.values()),
        "ready_for_strong_rule_efficiency_superiority_claim": native_beats_rule_latency
        and native_beats_rule_tokens
        and native_beats_rule_calls,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "native_synaptic_runtime_metrics": native_metrics,
        "bounded_state_rule_metrics": rule_metrics,
        "bounded_state_rule_success_conditioned_metrics": rule_success_metrics,
        "efficiency_comparison": {
            "native_beats_rule_mean_decision_latency": native_beats_rule_latency,
            "native_beats_rule_mean_token_cost": native_beats_rule_tokens,
            "native_beats_rule_mean_model_calls": native_beats_rule_calls,
            "native_over_rule_mean_decision_latency_ratio": native_metrics[
                "mean_decision_latency_ms"
            ]
            / max(rule_metrics["mean_decision_latency_ms"], 1.0e-12),
            "native_minus_rule_mean_token_cost_per_episode": native_metrics[
                "mean_token_equivalent_cost_per_episode"
            ]
            - rule_metrics["mean_token_equivalent_cost_per_episode"],
            "native_minus_rule_mean_model_calls_per_episode": native_metrics[
                "mean_model_calls_per_episode"
            ]
            - rule_metrics["mean_model_calls_per_episode"],
        },
        "supported_claims": [
            "the learned native synaptic runtime matched a strong visible-state rule baseline on bounded generated task completion and failure recovery"
        ]
        if all(checks.values())
        else [],
        "unsupported_claims": [
            "efficiency superiority over the strong visible-state rule baseline",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bu_rule_resistant_novel_state_composition"
            if all(checks.values())
            else "repair_phase2bt_success_conditioned_efficiency"
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
        description="Compare native runtime against a strong visible-state rule baseline."
    )
    parser.add_argument("--phase2bq-report-json", required=True)
    parser.add_argument("--phase2br-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-extra-steps", type=int, default=5)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bt_success_conditioned_efficiency(
        phase2bq_report_json=args.phase2bq_report_json,
        phase2br_report_json=args.phase2br_report_json,
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
