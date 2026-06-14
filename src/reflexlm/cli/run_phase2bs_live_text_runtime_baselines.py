from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.baselines.text_policies import HuggingFaceJSONPolicy
from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)
from reflexlm.cli.run_phase2br_runtime_ablation_comparison import (
    _aggregate_subreports,
)


TEXT_BASELINE_MODES: tuple[dict[str, Any], ...] = (
    {
        "mode_id": "prompt_only_json",
        "react_style": False,
    },
    {
        "mode_id": "react_style_json",
        "react_style": True,
    },
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _release_model_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def run_phase2bs_live_text_runtime_baselines(
    *,
    model_name: str,
    phase2bq_report_json: str | Path,
    phase2br_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    quantization: str = "none",
    max_new_tokens: int = 64,
    max_time_s: float | None = 10.0,
    max_retries: int = 0,
    include_full_history_react: bool = True,
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
    repository_rows = phase2bq_report.get("repository_reports")
    if not isinstance(repository_rows, list) or not repository_rows:
        raise ValueError("Phase2BS requires a Phase2BQ report with repository reports")
    manifest_hashes = {
        str(row["repository_id"]): _sha256(row["generated_manifest_json"])
        for row in repository_rows
    }
    output_root = Path(output_dir)
    baseline_reports: dict[str, dict[str, Any]] = {}

    modes = list(TEXT_BASELINE_MODES)
    if include_full_history_react:
        modes.append(
            {
                "mode_id": "full_history_react_json",
                "react_style": True,
                "maintain_history": True,
            }
        )
    for mode in modes:
        mode_id = str(mode["mode_id"])
        policy = HuggingFaceJSONPolicy(
            model_name,
            react_style=bool(mode["react_style"]),
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            max_time_s=max_time_s,
            max_retries=max_retries,
            policy_label=f"phase2bs_{mode_id}",
            maintain_history=bool(mode.get("maintain_history", False)),
        )
        subreports: list[dict[str, Any]] = []
        for repository in repository_rows:
            repository_id = str(repository["repository_id"])
            repository_output = output_root / mode_id / repository_id
            subreport = run_phase2bn_model_selected_sealed_runtime(
                checkpoint_path=None,
                manifest_json=repository["generated_manifest_json"],
                output_jsonl=repository_output / "trajectories.jsonl",
                output_report_json=repository_output / "report.json",
                timeout_seconds=timeout_seconds,
                max_extra_steps=max_extra_steps,
                policy_label=f"phase2bs_{mode_id}",
                authorize_bounded_debug_cortex_recovery=False,
                use_synaptic_motor_plan=False,
                policy_instance=policy,
            )
            subreports.append(subreport)
        baseline_reports[mode_id] = {
            "configuration": {
                **dict(mode),
                "model_name": model_name,
                "quantization": quantization,
                "max_new_tokens": max_new_tokens,
                "max_time_s": max_time_s,
                "max_retries": max_retries,
            },
            "metrics": _aggregate_subreports(subreports),
            "repository_report_jsons": [
                str(output_root / mode_id / str(row["repository_id"]) / "report.json")
                for row in repository_rows
            ],
        }
        del policy
        _release_model_memory()

    native_metrics = phase2br_report["mode_reports"]["native_synaptic_runtime"][
        "metrics"
    ]
    comparisons = {
        mode_id: {
            "task_completion_rate_delta": native_metrics[
                "task_completion_success_rate"
            ]
            - row["metrics"]["task_completion_success_rate"],
            "failure_recovery_rate_delta": native_metrics[
                "failure_recovery_success_rate"
            ]
            - row["metrics"]["failure_recovery_success_rate"],
            "mean_decision_latency_ms_ratio_native_over_baseline": native_metrics[
                "mean_decision_latency_ms"
            ]
            / max(row["metrics"]["mean_decision_latency_ms"], 1.0e-12),
            "mean_token_equivalent_cost_per_episode_ratio_native_over_baseline": native_metrics[
                "mean_token_equivalent_cost_per_episode"
            ]
            / max(row["metrics"]["mean_token_equivalent_cost_per_episode"], 1.0e-12),
            "mean_model_calls_per_episode_ratio_native_over_baseline": native_metrics[
                "mean_model_calls_per_episode"
            ]
            / max(row["metrics"]["mean_model_calls_per_episode"], 1.0e-12),
        }
        for mode_id, row in baseline_reports.items()
    }
    checks = {
        "phase2bq_source_report_passed": phase2bq_report.get("passed") is True,
        "phase2br_source_report_passed": phase2br_report.get("passed") is True,
        "same_generated_manifest_hashes_recorded": len(manifest_hashes)
        == len(repository_rows),
        "all_text_baselines_made_live_model_calls": all(
            row["metrics"]["model_calls"] > 0 for row in baseline_reports.values()
        ),
        "all_text_baselines_produced_runtime_decisions": all(
            row["metrics"]["decision_count"] > 0 for row in baseline_reports.values()
        ),
        "native_completed_more_tasks_than_all_measured_text_baselines": all(
            native_metrics["task_completion_success_rate"]
            > row["metrics"]["task_completion_success_rate"]
            for row in baseline_reports.values()
        ),
        "native_recovered_more_failure_tasks_than_all_measured_text_baselines": all(
            native_metrics["failure_recovery_success_rate"]
            > row["metrics"]["failure_recovery_success_rate"]
            for row in baseline_reports.values()
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2bs_live_text_runtime_baselines",
        "passed": passed,
        "ready_for_bounded_measured_text_baseline_claim": passed,
        "ready_for_full_history_react_baseline_claim": passed
        and include_full_history_react,
        "ready_for_full_tool_agent_superiority_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "model_name": model_name,
        "phase2bq_report_json": str(phase2bq_report_path),
        "phase2br_report_json": str(phase2br_report_path),
        "generated_manifest_sha256": manifest_hashes,
        "checks": checks,
        "native_synaptic_runtime_metrics": native_metrics,
        "baseline_reports": baseline_reports,
        "comparisons_against_measured_text_baselines": comparisons,
        "supported_claims": [
            f"on the same bounded generated runtime contracts, the native synaptic runtime outperformed the measured {model_name} prompt-only, ReAct-style one-step, and bounded full-history ReAct JSON policies in task completion and failure recovery"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "superiority over a general-purpose unrestricted tool-using ReAct agent",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "measurement_caveats": [
            "latency, token, and call-count ratios are descriptive because failed baselines may terminate early",
            "full_history_react_json preserves bounded state and decision history but remains restricted to the fixed action space",
        ],
        "next_required_experiment": (
            "phase2bt_success_conditioned_efficiency_comparison"
            if passed
            else "repair_phase2bs_live_text_runtime_baselines"
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
        description="Run live prompt-only and ReAct-style text policies on Phase2BQ contracts."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--phase2bq-report-json", required=True)
    parser.add_argument("--phase2br-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-time-s", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--exclude-full-history-react", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-extra-steps", type=int, default=5)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bs_live_text_runtime_baselines(
        model_name=args.model_name,
        phase2bq_report_json=args.phase2bq_report_json,
        phase2br_report_json=args.phase2br_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
        max_time_s=args.max_time_s,
        max_retries=args.max_retries,
        include_full_history_react=not args.exclude_full_history_react,
        timeout_seconds=args.timeout_seconds,
        max_extra_steps=args.max_extra_steps,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
