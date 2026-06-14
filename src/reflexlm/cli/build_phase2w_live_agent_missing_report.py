from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2w_live_agent_missing_report(
    *,
    intended_results_jsonl: str | Path,
    intended_config_json: str | Path,
    reason: str = "no_live_tool_agent_provider_configured",
) -> dict[str, Any]:
    return {
        "artifact_family": "phase2w_live_agent_missing_report",
        "passed": False,
        "baseline_kind": "live_tool_agent",
        "reason": reason,
        "blocked_actions": [
            "do_not_treat_static_overlap_baseline_as_live_agent",
            "do_not_claim_epoch_making_architecture_without_live_agent_baseline",
        ],
        "required_next_artifacts": {
            "config_json": str(Path(intended_config_json)),
            "results_jsonl": str(Path(intended_results_jsonl)),
        },
        "required_config_fields": [
            "run_id",
            "baseline_kind",
            "model_or_provider",
            "tool_budget",
            "context_policy",
            "retry_policy",
            "edit_permissions",
            "stop_rule",
            "cost_or_command_budget",
            "task_source_manifest_sha256",
            "executor_commit",
            "sandbox_image_or_env",
            "trace_archive_uri",
        ],
        "required_row_fields": [
            "task_id",
            "task_success",
            "patch_correctness",
            "test_pass_rate",
            "stop_condition_correctness",
            "false_completion",
            "unauthorized_write_count",
            "commands_used",
            "elapsed_seconds",
            "transcript_sha256",
            "patch_sha256",
            "pre_test_log_sha256",
            "post_test_log_sha256",
        ],
        "claim_boundary": "missing_report_only_not_evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Phase2W live-agent missing report.")
    parser.add_argument("--intended-results-jsonl", required=True)
    parser.add_argument("--intended-config-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--reason", default="no_live_tool_agent_provider_configured")
    args = parser.parse_args()
    report = build_phase2w_live_agent_missing_report(
        intended_results_jsonl=args.intended_results_jsonl,
        intended_config_json=args.intended_config_json,
        reason=args.reason,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
