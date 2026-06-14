from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)
from reflexlm.cli.run_phase2br_runtime_ablation_comparison import _aggregate_subreports
from reflexlm.eval import EventGatedSequencePolicy
from reflexlm.train import load_model_checkpoint


def run_phase2bu_event_gated_native_efficiency(
    *,
    checkpoint_path: str | Path,
    phase2bq_report_json: str | Path,
    phase2br_report_json: str | Path,
    phase2bt_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
    max_extra_steps: int = 5,
) -> dict[str, Any]:
    phase2bq_report = json.loads(Path(phase2bq_report_json).read_text(encoding="utf-8-sig"))
    phase2br_report = json.loads(Path(phase2br_report_json).read_text(encoding="utf-8-sig"))
    phase2bt_report = json.loads(Path(phase2bt_report_json).read_text(encoding="utf-8-sig"))
    model, vectorizer, checkpoint_payload = load_model_checkpoint(checkpoint_path, device="cpu")
    policy = EventGatedSequencePolicy(
        model,
        vectorizer,
        policy_label="phase2bu_event_gated_native",
        training_summary=checkpoint_payload.get("training_summary", {}),
        authorize_bounded_debug_cortex_recovery=True,
        use_synaptic_motor_plan=True,
    )
    output_root = Path(output_dir)
    subreports: list[dict[str, Any]] = []
    for repository in phase2bq_report["repository_reports"]:
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
                policy_label="phase2bu_event_gated_native",
                policy_instance=policy,
            )
        )
    event_metrics = _aggregate_subreports(subreports)
    native_metrics = phase2br_report["mode_reports"]["native_synaptic_runtime"]["metrics"]
    rule_metrics = phase2bt_report["bounded_state_rule_metrics"]
    checks = {
        "phase2bq_source_report_passed": phase2bq_report.get("passed") is True,
        "phase2br_source_report_passed": phase2br_report.get("passed") is True,
        "phase2bt_source_report_passed": phase2bt_report.get("passed") is True,
        "event_gated_native_completed_all_tasks": event_metrics["task_completion_success_rate"] == 1.0,
        "event_gated_native_recovered_all_failure_tasks": event_metrics["failure_recovery_success_rate"] == 1.0,
        "event_gated_native_reduced_model_calls_vs_full_native": event_metrics["mean_model_calls_per_episode"]
        < native_metrics["mean_model_calls_per_episode"],
        "event_gated_native_reduced_token_cost_vs_full_native": event_metrics[
            "mean_token_equivalent_cost_per_episode"
        ]
        < native_metrics["mean_token_equivalent_cost_per_episode"],
        "event_gated_native_reduced_latency_vs_full_native": event_metrics["mean_decision_latency_ms"]
        < native_metrics["mean_decision_latency_ms"],
    }
    report = {
        "artifact_family": "phase2bu_event_gated_native_efficiency",
        "passed": all(checks.values()),
        "ready_for_event_gated_efficiency_claim": all(checks.values()),
        "ready_for_strong_rule_efficiency_superiority_claim": (
            event_metrics["mean_decision_latency_ms"] < rule_metrics["mean_decision_latency_ms"]
            and event_metrics["mean_token_equivalent_cost_per_episode"]
            < rule_metrics["mean_token_equivalent_cost_per_episode"]
            and event_metrics["mean_model_calls_per_episode"]
            < rule_metrics["mean_model_calls_per_episode"]
        ),
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "event_gated_native_metrics": event_metrics,
        "full_native_metrics": native_metrics,
        "strong_rule_metrics": rule_metrics,
        "efficiency_deltas_vs_full_native": {
            "mean_model_calls_per_episode_delta": event_metrics["mean_model_calls_per_episode"]
            - native_metrics["mean_model_calls_per_episode"],
            "mean_token_cost_per_episode_delta": event_metrics[
                "mean_token_equivalent_cost_per_episode"
            ]
            - native_metrics["mean_token_equivalent_cost_per_episode"],
            "mean_decision_latency_ms_delta": event_metrics["mean_decision_latency_ms"]
            - native_metrics["mean_decision_latency_ms"],
        },
        "supported_claims": [
            "event gating preserved bounded generated-task completion while reducing neural calls, token-equivalent cost, and mean decision latency versus the full neural runtime"
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
            "phase2bv_rule_resistant_novel_state_composition"
            if all(checks.values())
            else "repair_phase2bu_event_gated_native_efficiency"
        ),
    }
    output_path = Path(output_report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run event-gated native policy on Phase2BQ contracts.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--phase2bq-report-json", required=True)
    parser.add_argument("--phase2br-report-json", required=True)
    parser.add_argument("--phase2bt-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-extra-steps", type=int, default=5)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bu_event_gated_native_efficiency(
        checkpoint_path=args.checkpoint_path,
        phase2bq_report_json=args.phase2bq_report_json,
        phase2br_report_json=args.phase2br_report_json,
        phase2bt_report_json=args.phase2bt_report_json,
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
