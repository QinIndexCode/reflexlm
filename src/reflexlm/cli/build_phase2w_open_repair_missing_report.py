from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2w_open_repair_missing_report(
    *,
    bounded_repair_boundary_json: str | Path,
    intended_results_jsonl: str | Path,
) -> dict[str, Any]:
    bounded = _read_json(bounded_repair_boundary_json)
    return {
        "artifact_family": "phase2w_open_ended_repair_missing_report",
        "passed": False,
        "task_family": None,
        "bounded_repair_boundary_passed": bounded.get("passed") is True,
        "bounded_repair_active_boundary": bounded.get("active_evidence_boundary"),
        "reason": "current_phase2s_public_repair_is_bounded_command_selection_not_open_ended_repair",
        "blocked_actions": [
            "do_not_relabel_phase2s_bounded_repair_as_open_ended",
            "do_not_claim_open_ended_repair_generalization_without_phase2w_results",
        ],
        "required_next_artifact": str(Path(intended_results_jsonl)),
        "required_row_fields": [
            "task_id",
            "task_family=open_ended_repair",
            "repo_origin",
            "repo_commit",
            "task_spec_sha256",
            "difficulty_axes",
            "full_task_success",
            "full_patch_correctness",
            "full_test_pass_rate",
            "best_live_agent_task_success",
            "best_live_agent_patch_correctness",
            "rollback_success",
            "unauthorized_write_count",
            "false_completion",
            "full_transcript_sha256",
            "full_patch_diff_sha256",
            "full_test_log_sha256",
            "live_agent_transcript_sha256",
            "live_agent_patch_diff_sha256",
            "live_agent_test_log_sha256",
        ],
        "required_difficulty_axes": [
            "multi_file_patch",
            "ambiguous_traceback",
            "dependency_or_environment_issue",
            "stale_state_refresh",
            "hidden_side_effect_guard",
        ],
        "claim_boundary": "missing_report_only_not_evidence",
        "inputs": {"bounded_repair_boundary_json": str(Path(bounded_repair_boundary_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2W open-ended repair missing report.")
    parser.add_argument("--bounded-repair-boundary-json", required=True)
    parser.add_argument("--intended-results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2w_open_repair_missing_report(
        bounded_repair_boundary_json=args.bounded_repair_boundary_json,
        intended_results_jsonl=args.intended_results_jsonl,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
