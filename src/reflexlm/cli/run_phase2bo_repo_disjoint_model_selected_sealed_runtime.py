from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)


def _git_value(repository_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git provenance command failed for {repository_root}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _git_repository_provenance(repository_root: str | Path) -> dict[str, str]:
    requested_root = Path(repository_root).resolve()
    git_root = Path(
        _git_value(requested_root, "rev-parse", "--show-toplevel")
    ).resolve()
    if git_root != requested_root:
        raise ValueError(
            f"repo-disjoint workspace must be a git root: {requested_root} != {git_root}"
        )
    return {
        "git_root": str(git_root),
        "origin": _git_value(git_root, "remote", "get-url", "origin"),
        "head": _git_value(git_root, "rev-parse", "HEAD"),
    }


def _repo_identity_checks(
    *,
    source: dict[str, str],
    repositories: list[dict[str, str]],
    minimum_repository_count: int,
) -> dict[str, bool]:
    roots = [row["git_root"] for row in repositories]
    origins = [row["origin"] for row in repositories]
    return {
        "minimum_independent_repository_count_met": len(repositories)
        >= minimum_repository_count,
        "all_git_roots_are_distinct": len(set(roots)) == len(roots),
        "all_git_roots_differ_from_source_repository": all(
            root != source["git_root"] for root in roots
        ),
        "all_origins_are_distinct": len(set(origins)) == len(origins),
        "all_heads_are_recorded": all(bool(row["head"]) for row in repositories),
    }


def run_phase2bo_repo_disjoint_model_selected_sealed_runtime(
    *,
    checkpoint_path: str | Path,
    suite_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
    max_extra_steps: int = 3,
) -> dict[str, Any]:
    suite_path = Path(suite_json).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8-sig"))
    source = _git_repository_provenance(suite["source_repository_root"])
    repositories = suite.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("repo-disjoint suite requires non-empty repositories")
    minimum_repository_count = int(suite.get("minimum_repository_count", 3))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    repo_reports: list[dict[str, Any]] = []
    provenances: list[dict[str, str]] = []
    for repository in repositories:
        if not isinstance(repository, dict):
            raise ValueError("each repo-disjoint repository entry must be an object")
        repository_id = str(repository["repository_id"])
        manifest_path = Path(str(repository["manifest_json"]))
        if not manifest_path.is_absolute():
            manifest_path = suite_path.parent / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
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
        "artifact_family": "phase2bo_repo_disjoint_model_selected_sealed_runtime",
        "passed": passed,
        "ready_for_bounded_repo_disjoint_model_selected_runtime_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "source_repository": source,
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
            "one bounded structured-action checkpoint completed allowlisted real runtime suites across independent public Git repositories"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "task-family-disjoint runtime transfer",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bp_task_family_disjoint_repo_runtime"
            if passed
            else "repair_phase2bo_repo_disjoint_model_selected_sealed_runtime"
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
        description="Run one bounded model-selected runtime checkpoint across independent Git repositories."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--suite-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-extra-steps", type=int, default=3)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bo_repo_disjoint_model_selected_sealed_runtime(
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
