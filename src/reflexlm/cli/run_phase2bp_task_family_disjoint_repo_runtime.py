from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)
from reflexlm.cli.run_phase2bo_repo_disjoint_model_selected_sealed_runtime import (
    _git_repository_provenance,
    _repo_identity_checks,
)


def _action_signature(steps: list[dict[str, Any]]) -> str:
    return " -> ".join(str(step["action_type"]) for step in steps)


def _manifest_signatures(manifest: dict[str, Any]) -> set[str]:
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("manifest requires episodes for task-family signature audit")
    signatures: set[str] = set()
    for episode in episodes:
        steps = episode.get("steps") if isinstance(episode, dict) else None
        if not isinstance(steps, list) or not steps:
            raise ValueError("each episode requires non-empty steps")
        signatures.add(_action_signature(steps))
    return signatures


def run_phase2bp_task_family_disjoint_repo_runtime(
    *,
    checkpoint_path: str | Path,
    suite_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
    max_extra_steps: int = 5,
) -> dict[str, Any]:
    suite_path = Path(suite_json).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8-sig"))
    source = _git_repository_provenance(suite["source_repository_root"])
    training_manifest_path = Path(str(suite["training_manifest_json"]))
    if not training_manifest_path.is_absolute():
        training_manifest_path = suite_path.parent / training_manifest_path
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8-sig")
    )
    training_signatures = _manifest_signatures(training_manifest)
    repositories = suite.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("task-family-disjoint suite requires repositories")
    minimum_repository_count = int(suite.get("minimum_repository_count", 3))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    repo_reports: list[dict[str, Any]] = []
    provenances: list[dict[str, str]] = []
    suite_signatures: set[str] = set()
    overlapping_signatures: set[str] = set()
    for repository in repositories:
        repository_id = str(repository["repository_id"])
        manifest_path = Path(str(repository["manifest_json"]))
        if not manifest_path.is_absolute():
            manifest_path = suite_path.parent / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        signatures = _manifest_signatures(manifest)
        suite_signatures.update(signatures)
        overlapping_signatures.update(signatures & training_signatures)
        provenance = _git_repository_provenance(manifest["workspace_root"])
        provenances.append(provenance)
        repository_output = output_root / repository_id
        subreport = run_phase2bn_model_selected_sealed_runtime(
            checkpoint_path=checkpoint_path,
            manifest_json=manifest_path,
            output_jsonl=repository_output / "trajectories.jsonl",
            output_report_json=repository_output / "report.json",
            timeout_seconds=timeout_seconds,
            max_extra_steps=max_extra_steps,
        )
        repo_reports.append(
            {
                "repository_id": repository_id,
                "manifest_json": str(manifest_path),
                "provenance": provenance,
                "action_signatures": sorted(signatures),
                "passed": subreport["passed"],
                "checks": subreport["checks"],
                "metrics": subreport["metrics"],
                "report_json": str(repository_output / "report.json"),
            }
        )

    identity_checks = _repo_identity_checks(
        source=source,
        repositories=provenances,
        minimum_repository_count=minimum_repository_count,
    )
    checks = {
        **identity_checks,
        "all_action_signatures_disjoint_from_training_manifest": not overlapping_signatures,
        "all_repository_runtime_suites_passed": all(
            row["passed"] for row in repo_reports
        ),
        "all_repository_actions_were_allowlisted": all(
            row["checks"]["all_model_selected_actions_were_allowlisted"]
            for row in repo_reports
        ),
        "all_repository_task_completion_predicates_satisfied": all(
            row["checks"]["all_task_completion_predicates_satisfied"]
            for row in repo_reports
        ),
    }
    passed = all(checks.values())
    total_episodes = sum(row["metrics"]["episodes"] for row in repo_reports)
    total_executed_actions = sum(
        row["metrics"]["executed_actions"] for row in repo_reports
    )
    total_task_completions = sum(
        row["metrics"]["task_completion_successes"] for row in repo_reports
    )
    report = {
        "artifact_family": "phase2bp_task_family_disjoint_repo_runtime",
        "passed": passed,
        "ready_for_bounded_task_family_disjoint_repo_runtime_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "source_repository": source,
        "training_manifest_json": str(training_manifest_path),
        "training_action_signatures": sorted(training_signatures),
        "suite_action_signatures": sorted(suite_signatures),
        "overlapping_training_action_signatures": sorted(overlapping_signatures),
        "checks": checks,
        "metrics": {
            "repositories": len(repo_reports),
            "episodes": total_episodes,
            "executed_actions": total_executed_actions,
            "task_completion_successes": total_task_completions,
            "task_completion_success_rate": total_task_completions
            / max(total_episodes, 1),
        },
        "repository_reports": repo_reports,
        "supported_claims": [
            "the bounded structured-action architecture completed repo-disjoint real runtime suites whose action signatures were absent from the Phase2BL training manifest"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bq_open_task_family_repo_runtime"
            if passed
            else "repair_phase2bp_task_family_disjoint_repo_runtime"
        ),
    }
    report_path = Path(output_report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run task-family-disjoint bounded runtime suites across independent Git repositories."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--suite-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-extra-steps", type=int, default=5)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bp_task_family_disjoint_repo_runtime(
        checkpoint_path=args.checkpoint_path,
        suite_json=args.suite_json,
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
