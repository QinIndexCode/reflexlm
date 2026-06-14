from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cj_runtime_interpreter_invariance import (
    _all_repo_checks,
    _all_repo_package_internal_runtime,
    _top_level_success,
)


PERTURBATION_IDS = (
    "direct_env_overlay",
    "cwd_subdir_probe",
    "cmd_wrapper_env_overlay",
)
ENV_KEY = "PHASE2CL_ENV_TOKEN"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _repository_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("repository_reports", [])
    return rows if isinstance(rows, list) else []


def _load_generated_manifests(report: dict[str, Any]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for repo in _repository_reports(report):
        manifest_path = repo.get("generated_manifest_json")
        if not manifest_path:
            continue
        manifests.append(_read_json(manifest_path))
    return manifests


def _run_steps(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in episode.get("permissions", [])
        if step.get("action_type") == "RUN_COMMAND"
    ]


def _episodes_by_perturbation(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for episode in manifest.get("episodes", []):
        perturbation_id = str(episode.get("generator", {}).get("perturbation_id", ""))
        rows[perturbation_id] = episode
    return rows


def _all_manifests_contract_bounded(manifests: list[dict[str, Any]]) -> bool:
    if not manifests:
        return False
    for manifest in manifests:
        for episode in manifest.get("episodes", []):
            if "steps" in episode or "expected_sequence" in episode:
                return False
            permissions = episode.get("permissions")
            completion = episode.get("completion_requirements")
            if not isinstance(permissions, list) or not isinstance(completion, list):
                return False
            if not permissions or not completion:
                return False
            if len(_run_steps(episode)) != 1:
                return False
    return True


def _all_manifests_have_expected_perturbations(
    manifests: list[dict[str, Any]],
) -> bool:
    expected = set(PERTURBATION_IDS)
    return bool(manifests) and all(
        set(_episodes_by_perturbation(manifest)) == expected for manifest in manifests
    )


def _all_env_overlays_are_manifest_declared(manifests: list[dict[str, Any]]) -> bool:
    for manifest in manifests:
        by_id = _episodes_by_perturbation(manifest)
        for perturbation_id in ("direct_env_overlay", "cmd_wrapper_env_overlay"):
            episode = by_id.get(perturbation_id)
            if episode is None:
                return False
            token = str(episode.get("generator", {}).get("payload_token", ""))
            run_step = _run_steps(episode)[0]
            env = run_step.get("env")
            if env != {ENV_KEY: token}:
                return False
            if "shell" in run_step:
                return False
    return bool(manifests)


def _all_cmd_wrappers_are_explicit_argv(manifests: list[dict[str, Any]]) -> bool:
    for manifest in manifests:
        episode = _episodes_by_perturbation(manifest).get("cmd_wrapper_env_overlay")
        if episode is None:
            return False
        run_step = _run_steps(episode)[0]
        argv = run_step.get("argv")
        if not isinstance(argv, list) or argv[:3] != ["cmd.exe", "/d", "/c"]:
            return False
        if run_step.get("shell") is True:
            return False
    return bool(manifests)


def _all_cwd_probes_are_generated_subdirs(manifests: list[dict[str, Any]]) -> bool:
    for manifest in manifests:
        repository_id = str(manifest.get("generated_by", {}).get("repository_id", ""))
        episode = _episodes_by_perturbation(manifest).get("cwd_subdir_probe")
        if episode is None:
            return False
        token = str(episode.get("generator", {}).get("payload_token", ""))
        expected_prefix = f".reflexlm_runtime_probe/phase2cl-{repository_id}-"
        run_step = _run_steps(episode)[0]
        cwd = str(run_step.get("cwd", ""))
        if not cwd.startswith(expected_prefix) or not cwd.endswith(token):
            return False
        if Path(cwd).is_absolute() or ".." in Path(cwd).parts:
            return False
    return bool(manifests)


def _all_failure_recovery_gates_not_applicable(report: dict[str, Any]) -> bool:
    repos = _repository_reports(report)
    return bool(repos) and all(
        repo.get("metrics", {}).get("failure_episodes") == 0
        and repo.get("metrics", {}).get("failure_recovery_gate_applicable") is False
        and repo.get("checks", {}).get("failure_recovery_success_rate_meets_gate")
        is True
        for repo in repos
    )


def audit_phase2cl_runtime_environment_shell_matrix(
    *,
    runtime_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    runtime_report = _read_json(runtime_report_json)
    manifests = _load_generated_manifests(runtime_report)
    repo_count = len(_repository_reports(runtime_report))
    perturbation_counts = runtime_report.get("metrics", {}).get(
        "perturbation_counts",
        {},
    )
    checks = {
        "artifact_family_matches_phase2cl": (
            runtime_report.get("artifact_family")
            == "phase2cl_runtime_environment_and_shell_perturbation_matrix"
        ),
        "top_level_runtime_report_passed": _top_level_success(runtime_report),
        "minimum_three_repositories_met": repo_count >= 3,
        "all_perturbation_families_present": all(
            perturbation_counts.get(perturbation_id, 0) >= repo_count
            for perturbation_id in PERTURBATION_IDS
        ),
        "bounded_claim_true_only": (
            runtime_report.get(
                "ready_for_bounded_runtime_environment_shell_perturbation_claim"
            )
            is True
            and runtime_report.get("ready_for_general_shell_autonomy_claim") is False
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
        "all_generated_manifests_have_expected_perturbations": (
            _all_manifests_have_expected_perturbations(manifests)
        ),
        "all_generated_manifests_are_contract_bounded": (
            _all_manifests_contract_bounded(manifests)
        ),
        "all_env_overlays_are_manifest_declared": (
            _all_env_overlays_are_manifest_declared(manifests)
        ),
        "all_cmd_wrappers_are_explicit_argv_not_shell_true": (
            _all_cmd_wrappers_are_explicit_argv(manifests)
        ),
        "all_cwd_probes_are_generated_relative_subdirs": (
            _all_cwd_probes_are_generated_subdirs(manifests)
        ),
        "all_failure_recovery_gates_not_applicable_for_non_failure_tasks": (
            _all_failure_recovery_gates_not_applicable(runtime_report)
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cl_runtime_environment_shell_matrix_audit",
        "passed": passed,
        "ready_for_bounded_runtime_environment_shell_perturbation_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "repositories": repo_count,
            "perturbation_ids": list(PERTURBATION_IDS),
            "perturbation_counts": perturbation_counts,
            "episodes": runtime_report.get("metrics", {}).get("episodes"),
            "executed_actions": runtime_report.get("metrics", {}).get(
                "executed_actions"
            ),
            "task_completion_success_rate": runtime_report.get("metrics", {}).get(
                "task_completion_success_rate"
            ),
            "runtime_interpreter": runtime_report.get("runtime_interpreter"),
            "python_version": runtime_report.get("runtime_environment", {}).get(
                "version"
            ),
        },
        "supported_claims": [
            (
                "bounded package-internal structured-runtime cortex completed generated "
                "repository-disjoint environment overlay, cwd, and explicit cmd.exe argv "
                "wrapper perturbation tasks under a recorded CPython runtime without "
                "free-form shell execution or unrelated expert loading"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "free-form shell autonomy",
            "arbitrary environment generalization",
            "general shell invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cm_cross_runtime_shell_environment_matrix"
            if passed
            else "repair_phase2cl_runtime_environment_shell_matrix"
        ),
        "evidence": {
            "runtime_report_json": str(runtime_report_json),
            "generated_manifest_jsons": [
                str(repo.get("generated_manifest_json"))
                for repo in _repository_reports(runtime_report)
            ],
        },
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Phase2CL runtime environment and shell matrix."
    )
    parser.add_argument("--runtime-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cl_runtime_environment_shell_matrix(
        runtime_report_json=args.runtime_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
