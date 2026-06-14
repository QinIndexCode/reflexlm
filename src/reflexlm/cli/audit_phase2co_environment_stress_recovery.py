from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cj_runtime_interpreter_invariance import (
    _all_repo_checks,
    _all_repo_package_internal_runtime,
)
from reflexlm.cli.audit_phase2cl_runtime_environment_shell_matrix import (
    _repository_reports,
)
from reflexlm.cli.run_phase2co_environment_stress_with_failure_recovery import (
    ENV_KEY,
    STRESS_IDS,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _run_steps(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in episode.get("completion_requirements", [])
        if step.get("action_type") == "RUN_COMMAND"
    ]


def _episodes_by_stress(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(episode.get("generator", {}).get("stress_id", "")): episode
        for episode in manifest.get("episodes", [])
    }


def _load_generated_manifests(report: dict[str, Any]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for repo in _repository_reports(report):
        manifest_path = repo.get("generated_manifest_json")
        if manifest_path:
            manifests.append(_read_json(manifest_path))
    return manifests


def _all_manifests_have_expected_stressors(manifests: list[dict[str, Any]]) -> bool:
    expected = set(STRESS_IDS)
    return bool(manifests) and all(
        set(_episodes_by_stress(manifest)) == expected for manifest in manifests
    )


def _all_manifests_are_contract_bounded_failure_recovery(
    manifests: list[dict[str, Any]],
) -> bool:
    if not manifests:
        return False
    for manifest in manifests:
        for episode in manifest.get("episodes", []):
            if "steps" in episode or "expected_sequence" in episode:
                return False
            if episode.get("requires_failure") is not True:
                return False
            permissions = episode.get("permissions")
            completion = episode.get("completion_requirements")
            if not isinstance(permissions, list) or not isinstance(completion, list):
                return False
            run_steps = _run_steps(episode)
            if len(run_steps) != 2:
                return False
            if int(run_steps[0].get("expected_exit_code", 0)) == 0:
                return False
            if int(run_steps[1].get("expected_exit_code", 1)) != 0:
                return False
    return True


def _all_env_recovery_steps_are_declared(manifests: list[dict[str, Any]]) -> bool:
    for manifest in manifests:
        episode = _episodes_by_stress(manifest).get(
            "missing_env_then_overlay_recover"
        )
        if episode is None:
            return False
        token = str(episode.get("generator", {}).get("payload_token", ""))
        fail_step, recover_step = _run_steps(episode)
        if ENV_KEY in fail_step.get("env", {}):
            return False
        if recover_step.get("env") != {ENV_KEY: token}:
            return False
    return bool(manifests)


def _all_cwd_recovery_steps_are_generated_subdirs(
    manifests: list[dict[str, Any]],
) -> bool:
    for manifest in manifests:
        repository_id = str(manifest.get("generated_by", {}).get("repository_id", ""))
        episode = _episodes_by_stress(manifest).get("wrong_cwd_then_subdir_recover")
        if episode is None:
            return False
        token = str(episode.get("generator", {}).get("payload_token", ""))
        fail_step, recover_step = _run_steps(episode)
        cwd = str(recover_step.get("cwd", ""))
        expected_prefix = f".reflexlm_runtime_probe/phase2co-{repository_id}-"
        if "cwd" in fail_step:
            return False
        if not cwd.startswith(expected_prefix) or not cwd.endswith(token):
            return False
        path = Path(cwd)
        if path.is_absolute() or ".." in path.parts:
            return False
    return bool(manifests)


def _all_cmd_wrapper_steps_are_explicit_argv(manifests: list[dict[str, Any]]) -> bool:
    for manifest in manifests:
        episode = _episodes_by_stress(manifest).get(
            "cmd_wrapper_failure_then_recover"
        )
        if episode is None:
            return False
        for step in _run_steps(episode):
            argv = step.get("argv")
            if not isinstance(argv, list) or argv[:3] != ["cmd.exe", "/d", "/c"]:
                return False
            if step.get("shell") is True:
                return False
    return bool(manifests)


def _load_subreports(report: dict[str, Any]) -> list[dict[str, Any]]:
    subreports: list[dict[str, Any]] = []
    for repo in _repository_reports(report):
        report_path = repo.get("report_json")
        if report_path:
            subreports.append(_read_json(report_path))
    return subreports


def _all_subreport_episodes_observed_failure_recovery(
    subreports: list[dict[str, Any]],
) -> bool:
    if not subreports:
        return False
    for subreport in subreports:
        episodes = subreport.get("episode_reports", [])
        if not episodes:
            return False
        for episode in episodes:
            if episode.get("requires_failure") is not True:
                return False
            if episode.get("observed_failure") is not True:
                return False
            if episode.get("observed_recovery_after_failure") is not True:
                return False
            if episode.get("recovery_success") is not True:
                return False
            if episode.get("task_completion_success") is not True:
                return False
            if episode.get("unexpected_outcomes") != 0:
                return False
            if episode.get("selected_done") is not True:
                return False
    return True


def _all_repo_failure_metrics_complete(report: dict[str, Any]) -> bool:
    repos = _repository_reports(report)
    return bool(repos) and all(
        repo.get("metrics", {}).get("failure_recovery_gate_applicable") is True
        and repo.get("metrics", {}).get("failure_recovery_success_rate") == 1.0
        and repo.get("failure_recovery_metrics", {}).get("failure_episodes")
        == repo.get("metrics", {}).get("episodes")
        and repo.get("failure_recovery_metrics", {}).get("observed_failures")
        == repo.get("metrics", {}).get("episodes")
        and repo.get("failure_recovery_metrics", {}).get(
            "observed_recoveries_after_failure"
        )
        == repo.get("metrics", {}).get("episodes")
        for repo in repos
    )


def audit_phase2co_environment_stress_recovery(
    *,
    runtime_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    runtime_report = _read_json(runtime_report_json)
    repos = _repository_reports(runtime_report)
    manifests = _load_generated_manifests(runtime_report)
    subreports = _load_subreports(runtime_report)
    repo_count = len(repos)
    stress_counts = runtime_report.get("metrics", {}).get("stress_counts", {})
    checks = {
        "artifact_family_matches_phase2co": (
            runtime_report.get("artifact_family")
            == "phase2co_environment_stress_with_failure_recovery"
        ),
        "top_level_runtime_report_passed": runtime_report.get("passed") is True,
        "minimum_three_repositories_met": repo_count >= 3,
        "all_stress_families_present": all(
            stress_counts.get(stress_id, 0) >= repo_count for stress_id in STRESS_IDS
        ),
        "bounded_claim_true_only": (
            runtime_report.get(
                "ready_for_bounded_environment_stress_failure_recovery_claim"
            )
            is True
            and runtime_report.get("ready_for_general_shell_autonomy_claim") is False
            and runtime_report.get("ready_for_general_runtime_invariance_claim") is False
            and runtime_report.get("ready_for_open_ended_native_perception_claim")
            is False
            and runtime_report.get("ready_for_production_autonomy_claim") is False
            and runtime_report.get("ready_for_epoch_making_architecture_claim")
            is False
        ),
        "package_structured_runtime_only_view": (
            runtime_report.get("package_metadata", {}).get(
                "structured_runtime_cortex_packaged"
            )
            is True
            and runtime_report.get("package_metadata", {}).get(
                "native_head_policy_loaded"
            )
            is False
            and runtime_report.get("package_metadata", {}).get(
                "verification_cortex_loaded"
            )
            is False
        ),
        "all_repositories_used_package_internal_runtime_cortex": (
            _all_repo_package_internal_runtime(runtime_report)
        ),
        "all_repository_actions_allowlisted": _all_repo_checks(
            runtime_report,
            "all_model_selected_actions_were_allowlisted",
        ),
        "all_repository_completion_predicates_satisfied": _all_repo_checks(
            runtime_report,
            "all_task_completion_predicates_satisfied",
        ),
        "all_generated_manifests_were_loaded": len(manifests) == repo_count
        and repo_count > 0,
        "all_generated_manifests_have_expected_stressors": (
            _all_manifests_have_expected_stressors(manifests)
        ),
        "all_generated_manifests_are_contract_bounded_failure_recovery": (
            _all_manifests_are_contract_bounded_failure_recovery(manifests)
        ),
        "all_env_recovery_steps_are_declared": (
            _all_env_recovery_steps_are_declared(manifests)
        ),
        "all_cwd_recovery_steps_are_generated_relative_subdirs": (
            _all_cwd_recovery_steps_are_generated_subdirs(manifests)
        ),
        "all_cmd_wrapper_steps_are_explicit_argv_not_shell_true": (
            _all_cmd_wrapper_steps_are_explicit_argv(manifests)
        ),
        "all_repo_failure_metrics_complete": _all_repo_failure_metrics_complete(
            runtime_report
        ),
        "all_subreports_loaded": len(subreports) == repo_count and repo_count > 0,
        "all_subreport_episodes_observed_failure_recovery": (
            _all_subreport_episodes_observed_failure_recovery(subreports)
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2co_environment_stress_recovery_audit",
        "passed": passed,
        "ready_for_bounded_environment_stress_failure_recovery_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "repositories": repo_count,
            "stress_ids": list(STRESS_IDS),
            "stress_counts": stress_counts,
            "episodes": runtime_report.get("metrics", {}).get("episodes"),
            "executed_actions": runtime_report.get("metrics", {}).get(
                "executed_actions"
            ),
            "task_completion_success_rate": runtime_report.get("metrics", {}).get(
                "task_completion_success_rate"
            ),
            "failure_episodes": runtime_report.get("metrics", {}).get(
                "failure_episodes"
            ),
            "observed_failures": runtime_report.get("metrics", {}).get(
                "observed_failures"
            ),
            "observed_recoveries_after_failure": runtime_report.get(
                "metrics", {}
            ).get("observed_recoveries_after_failure"),
            "failure_recovery_success_rate": runtime_report.get("metrics", {}).get(
                "failure_recovery_success_rate"
            ),
            "runtime_interpreter": runtime_report.get("runtime_interpreter"),
            "python_version": runtime_report.get("runtime_environment", {}).get(
                "version"
            ),
        },
        "supported_claims": [
            (
                "bounded package-internal structured-runtime cortex completed generated "
                "repository-disjoint environment and cmd-wrapper stress episodes with "
                "episode-level observed failure and subsequent bounded recovery evidence"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "free-form shell autonomy",
            "arbitrary shell/environment generalization",
            "general runtime invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cp_cross_runtime_environment_stress_recovery_matrix"
            if passed
            else "repair_phase2co_environment_stress_recovery"
        ),
        "evidence": {
            "runtime_report_json": str(runtime_report_json),
            "generated_manifest_jsons": [
                str(repo.get("generated_manifest_json")) for repo in repos
            ],
            "subreport_jsons": [str(repo.get("report_json")) for repo in repos],
        },
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Phase2CO environment stress recovery report."
    )
    parser.add_argument("--runtime-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2co_environment_stress_recovery(
        runtime_report_json=args.runtime_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
