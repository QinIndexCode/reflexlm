from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)
from reflexlm.cli.run_phase2bo_repo_disjoint_model_selected_sealed_runtime import (
    _git_repository_provenance,
    _repo_identity_checks,
)
from reflexlm.cli.run_phase2bq_open_task_family_repo_runtime import _contract_signature
from reflexlm.llm.native_nervous_package import NativeNervousPolicyPackage


PERTURBATION_IDS = (
    "direct_env_overlay",
    "cwd_subdir_probe",
    "cmd_wrapper_env_overlay",
)


def _payload_token(*, seed: int, repository_id: str, perturbation_id: str) -> str:
    digest = hashlib.sha256(
        f"phase2cl|{seed}|{repository_id}|{perturbation_id}".encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _direct_env_overlay_episode(*, repository_id: str, token: str) -> dict[str, Any]:
    env_key = "PHASE2CL_ENV_TOKEN"
    code = f"import os; print('direct-env-{token}-' + os.environ[{env_key!r}])"
    run_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", code],
        "env": {env_key: token},
        "expected_exit_code": 0,
        "expected_stdout_contains": f"direct-env-{token}-{token}",
    }
    return {
        "episode_id": f"phase2cl-{repository_id}-direct-env-{token}",
        "description": "Environment-overlay receptor task with direct Python argv.",
        "task_type": "common_error_recovery_routine",
        "permissions": [
            run_step,
            {"action_type": "READ_STDOUT"},
            {"action_type": "DONE"},
        ],
        "completion_requirements": [run_step, {"action_type": "READ_STDOUT"}],
        "requires_failure": False,
        "max_steps": 6,
        "generator": {
            "phase": "phase2cl",
            "perturbation_id": "direct_env_overlay",
            "payload_token": token,
        },
    }


def _cwd_subdir_probe_episode(*, repository_id: str, token: str) -> dict[str, Any]:
    cwd = f".reflexlm_runtime_probe/phase2cl-{repository_id}-{token}"
    code = "from pathlib import Path; print('cwd-' + Path.cwd().name)"
    run_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", code],
        "cwd": cwd,
        "expected_exit_code": 0,
        "expected_stdout_contains": f"cwd-phase2cl-{repository_id}-{token}",
    }
    return {
        "episode_id": f"phase2cl-{repository_id}-cwd-subdir-{token}",
        "description": "Working-directory receptor task in a generated subdirectory.",
        "task_type": "common_error_recovery_routine",
        "permissions": [
            run_step,
            {"action_type": "READ_STDOUT"},
            {"action_type": "DONE"},
        ],
        "completion_requirements": [run_step, {"action_type": "READ_STDOUT"}],
        "requires_failure": False,
        "max_steps": 6,
        "generator": {
            "phase": "phase2cl",
            "perturbation_id": "cwd_subdir_probe",
            "payload_token": token,
            "cwd": cwd,
        },
    }


def _cmd_wrapper_env_overlay_episode(
    *,
    repository_id: str,
    token: str,
) -> dict[str, Any]:
    env_key = "PHASE2CL_ENV_TOKEN"
    code = f"import os; print('cmd-env-{token}-' + os.environ[{env_key!r}])"
    run_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["cmd.exe", "/d", "/c", "<PYTHON>", "-c", code],
        "env": {env_key: token},
        "expected_exit_code": 0,
        "expected_stdout_contains": f"cmd-env-{token}-{token}",
    }
    return {
        "episode_id": f"phase2cl-{repository_id}-cmd-wrapper-env-{token}",
        "description": "Explicit cmd.exe wrapper task with environment overlay.",
        "task_type": "common_error_recovery_routine",
        "permissions": [
            run_step,
            {"action_type": "READ_STDOUT"},
            {"action_type": "DONE"},
        ],
        "completion_requirements": [run_step, {"action_type": "READ_STDOUT"}],
        "requires_failure": False,
        "max_steps": 6,
        "generator": {
            "phase": "phase2cl",
            "perturbation_id": "cmd_wrapper_env_overlay",
            "payload_token": token,
            "shell": "cmd.exe /d /c",
        },
    }


def _build_episode(
    *,
    perturbation_id: str,
    repository_id: str,
    token: str,
) -> dict[str, Any]:
    if perturbation_id == "direct_env_overlay":
        return _direct_env_overlay_episode(repository_id=repository_id, token=token)
    if perturbation_id == "cwd_subdir_probe":
        return _cwd_subdir_probe_episode(repository_id=repository_id, token=token)
    if perturbation_id == "cmd_wrapper_env_overlay":
        return _cmd_wrapper_env_overlay_episode(repository_id=repository_id, token=token)
    raise ValueError(f"unknown phase2cl perturbation_id: {perturbation_id}")


def _generate_manifest_for_repository(
    *,
    suite_seed: int,
    repository: dict[str, Any],
) -> dict[str, Any]:
    repository_id = str(repository["repository_id"])
    episodes = [
        _build_episode(
            perturbation_id=perturbation_id,
            repository_id=repository_id,
            token=_payload_token(
                seed=suite_seed,
                repository_id=repository_id,
                perturbation_id=perturbation_id,
            ),
        )
        for perturbation_id in PERTURBATION_IDS
    ]
    return {
        "workspace_root": str(repository["workspace_root"]),
        "repetitions_per_episode": 1,
        "generated_by": {
            "phase": "phase2cl",
            "seed": suite_seed,
            "repository_id": repository_id,
            "perturbation_ids": list(PERTURBATION_IDS),
        },
        "episodes": episodes,
    }


def _prepare_manifest_workspace(manifest: dict[str, Any]) -> None:
    workspace_root = Path(str(manifest["workspace_root"]))
    for episode in manifest["episodes"]:
        for step in episode.get("permissions", []):
            cwd = step.get("cwd")
            if cwd is not None:
                (workspace_root / str(cwd)).mkdir(parents=True, exist_ok=True)


def run_phase2cl_runtime_environment_and_shell_perturbation_matrix(
    *,
    package_path: str | Path,
    suite_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 5.0,
    max_extra_steps: int = 4,
    package_device: str | None = None,
    package_quantization: str | None = None,
    package_model_load_strategy: str | None = None,
    package_offload_state_dict: bool | None = None,
) -> dict[str, Any]:
    suite_path = Path(suite_json).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8-sig"))
    source = _git_repository_provenance(suite["source_repository_root"])
    seed = int(suite["seed"])
    repositories = suite.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("phase2cl suite requires non-empty repositories")
    minimum_repository_count = int(suite.get("minimum_repository_count", 3))
    output_root = Path(output_dir)
    generated_manifest_dir = output_root / "generated_manifests"
    generated_manifest_dir.mkdir(parents=True, exist_ok=True)
    package_policy = NativeNervousPolicyPackage(
        package_path,
        device=package_device,
        quantization=package_quantization,
        model_load_strategy=package_model_load_strategy,
        offload_state_dict=package_offload_state_dict,
        load_native_head_policy=False,
        load_verification_cortex=False,
    )
    runtime_policy = package_policy.create_structured_runtime_policy()
    package_metadata = package_policy.metadata()

    repo_reports: list[dict[str, Any]] = []
    provenances: list[dict[str, str]] = []
    all_contract_signatures: set[str] = set()
    perturbation_counts: dict[str, int] = {key: 0 for key in PERTURBATION_IDS}
    for repository in repositories:
        repository_id = str(repository["repository_id"])
        manifest = _generate_manifest_for_repository(
            suite_seed=seed,
            repository=repository,
        )
        _prepare_manifest_workspace(manifest)
        manifest_path = generated_manifest_dir / f"{repository_id}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        provenance = _git_repository_provenance(manifest["workspace_root"])
        provenances.append(provenance)
        contract_signatures = {
            _contract_signature(episode) for episode in manifest["episodes"]
        }
        all_contract_signatures.update(contract_signatures)
        for episode in manifest["episodes"]:
            perturbation_counts[str(episode["generator"]["perturbation_id"])] += 1
        repository_output = output_root / repository_id
        subreport = run_phase2bn_model_selected_sealed_runtime(
            checkpoint_path=None,
            manifest_json=manifest_path,
            output_jsonl=repository_output / "trajectories.jsonl",
            output_report_json=repository_output / "report.json",
            timeout_seconds=timeout_seconds,
            max_extra_steps=max_extra_steps,
            policy_instance=runtime_policy,
            policy_label="package_internal_structured_runtime_cortex",
        )
        repo_reports.append(
            {
                "repository_id": repository_id,
                "generated_manifest_json": str(manifest_path),
                "provenance": provenance,
                "perturbation_ids": manifest["generated_by"]["perturbation_ids"],
                "contract_signatures": sorted(contract_signatures),
                "passed": subreport["passed"],
                "checks": subreport["checks"],
                "metrics": subreport["metrics"],
                "policy_configuration": subreport.get("policy_configuration") or {},
                "report_json": str(repository_output / "report.json"),
            }
        )

    identity_checks = _repo_identity_checks(
        source=source,
        repositories=provenances,
        minimum_repository_count=minimum_repository_count,
    )
    total_episodes = sum(row["metrics"]["episodes"] for row in repo_reports)
    total_executed_actions = sum(
        row["metrics"]["executed_actions"] for row in repo_reports
    )
    total_task_completions = sum(
        row["metrics"]["task_completion_successes"] for row in repo_reports
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
        "package_structured_runtime_cortex_packaged": (
            package_metadata.get("structured_runtime_cortex_packaged") is True
        ),
        "package_native_head_not_loaded": (
            package_metadata.get("native_head_policy_loaded") is False
        ),
        "package_verification_cortex_not_loaded": (
            package_metadata.get("verification_cortex_loaded") is False
        ),
        "all_perturbation_families_present": all(
            perturbation_counts[key] >= len(repositories) for key in PERTURBATION_IDS
        ),
    }
    package_internal_passed = all(
        (
            row["policy_configuration"]
            .get("policy_metadata", {})
            .get("package_internal_expert")
            is True
            and row["policy_configuration"]
            .get("policy_metadata", {})
            .get("expert_name")
            == "structured_runtime_cortex"
        )
        for row in repo_reports
    )
    checks["all_repositories_used_package_internal_runtime_cortex"] = (
        package_internal_passed
    )
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cl_runtime_environment_and_shell_perturbation_matrix",
        "passed": passed,
        "ready_for_bounded_runtime_environment_shell_perturbation_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "source_repository": source,
        "seed": seed,
        "runtime_interpreter": sys.executable,
        "runtime_environment": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
            "executable": sys.executable,
        },
        "perturbation_ids": list(PERTURBATION_IDS),
        "generated_manifest_dir": str(generated_manifest_dir),
        "generated_contract_signatures": sorted(all_contract_signatures),
        "package_metadata": package_metadata,
        "checks": checks,
        "metrics": {
            "repositories": len(repo_reports),
            "perturbation_counts": perturbation_counts,
            "episodes": total_episodes,
            "executed_actions": total_executed_actions,
            "task_completion_successes": total_task_completions,
            "task_completion_success_rate": total_task_completions
            / max(total_episodes, 1),
        },
        "repository_reports": repo_reports,
        "supported_claims": [
            (
                "the package-internal structured runtime cortex completed bounded "
                "environment-overlay, cwd, and explicit cmd.exe-wrapper perturbation "
                "tasks without free-form shell generation or unrelated expert loading"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "free-form shell autonomy",
            "arbitrary environment generalization",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cm_cross_runtime_shell_environment_matrix"
            if passed
            else "repair_phase2cl_runtime_environment_and_shell_perturbation_matrix"
        ),
    }
    output_path = Path(output_report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase2CL runtime environment and shell perturbation matrix."
    )
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--package-device")
    parser.add_argument("--package-quantization")
    parser.add_argument("--package-model-load-strategy")
    parser.add_argument(
        "--package-offload-state-dict",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--suite-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-extra-steps", type=int, default=4)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2cl_runtime_environment_and_shell_perturbation_matrix(
        package_path=args.package_path,
        suite_json=args.suite_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
        timeout_seconds=args.timeout_seconds,
        max_extra_steps=args.max_extra_steps,
        package_device=args.package_device,
        package_quantization=args.package_quantization,
        package_model_load_strategy=args.package_model_load_strategy,
        package_offload_state_dict=args.package_offload_state_dict,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
