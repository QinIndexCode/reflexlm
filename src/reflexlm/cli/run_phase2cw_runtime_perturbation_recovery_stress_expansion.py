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
from reflexlm.cli.run_phase2co_environment_stress_with_failure_recovery import (
    ENV_KEY,
    STRESS_IDS as PHASE2CO_STRESS_IDS,
    _build_episode as _build_phase2co_episode,
)
from reflexlm.llm.native_nervous_package import NativeNervousPolicyPackage


PHASE2CW_EXTRA_STRESS_IDS = (
    "missing_file_then_create_recover",
    "bad_arg_then_correct_arg_recover",
)
STRESS_IDS = tuple(PHASE2CO_STRESS_IDS) + PHASE2CW_EXTRA_STRESS_IDS


def _payload_token(*, seed: int, repository_id: str, stress_id: str) -> str:
    digest = hashlib.sha256(
        f"phase2cw|{seed}|{repository_id}|{stress_id}".encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _missing_file_then_create_recover_episode(
    *,
    repository_id: str,
    token: str,
) -> dict[str, Any]:
    relative_dir = f".reflexlm_runtime_probe/phase2cw-{repository_id}-{token}"
    relative_file = f"{relative_dir}/probe.txt"
    fail_code = (
        "from pathlib import Path; import sys; "
        f"path = Path({relative_file!r}); "
        f"sys.stderr.write('file-missing-{token}\\n'); "
        "raise SystemExit(0 if path.exists() else 41)"
    )
    recover_code = (
        "from pathlib import Path; "
        f"path = Path({relative_file!r}); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        f"path.write_text({token!r}, encoding='utf-8'); "
        f"print('file-recovered-{token}-' + path.read_text(encoding='utf-8'))"
    )
    fail_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", fail_code],
        "expected_exit_code": 41,
        "expected_stderr_contains": f"file-missing-{token}",
    }
    recover_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", recover_code],
        "expected_exit_code": 0,
        "expected_stdout_contains": f"file-recovered-{token}-{token}",
    }
    return {
        "episode_id": f"phase2cw-{repository_id}-missing-file-recover-{token}",
        "description": "Missing file failure followed by bounded file creation recovery.",
        "task_type": "common_error_recovery_routine",
        "ambient_observation_actions": ["READ_STDERR", "READ_STDOUT"],
        "permissions": [
            fail_step,
            {"action_type": "READ_STDERR"},
            recover_step,
            {"action_type": "READ_STDOUT"},
            {"action_type": "DONE"},
        ],
        "completion_requirements": [
            fail_step,
            {"action_type": "READ_STDERR"},
            recover_step,
            {"action_type": "READ_STDOUT"},
        ],
        "requires_failure": True,
        "max_steps": 9,
        "generator": {
            "phase": "phase2cw",
            "stress_id": "missing_file_then_create_recover",
            "payload_token": token,
            "recovery_file": relative_file,
        },
    }


def _bad_arg_then_correct_arg_recover_episode(
    *,
    repository_id: str,
    token: str,
) -> dict[str, Any]:
    fail_code = (
        "import sys; "
        f"sys.stderr.write('arg-mismatch-{token}-' + sys.argv[1] + '\\n'); "
        f"raise SystemExit(0 if sys.argv[1] == {token!r} else 42)"
    )
    recover_code = (
        "import sys; "
        f"print('arg-recovered-{token}-' + sys.argv[1])"
    )
    fail_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", fail_code, f"wrong-{token}"],
        "expected_exit_code": 42,
        "expected_stderr_contains": f"arg-mismatch-{token}-wrong-{token}",
    }
    recover_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", recover_code, token],
        "expected_exit_code": 0,
        "expected_stdout_contains": f"arg-recovered-{token}-{token}",
    }
    return {
        "episode_id": f"phase2cw-{repository_id}-bad-arg-recover-{token}",
        "description": "Bad command argument failure followed by bounded corrected-argument recovery.",
        "task_type": "common_error_recovery_routine",
        "ambient_observation_actions": ["READ_STDERR", "READ_STDOUT"],
        "permissions": [
            fail_step,
            {"action_type": "READ_STDERR"},
            recover_step,
            {"action_type": "READ_STDOUT"},
            {"action_type": "DONE"},
        ],
        "completion_requirements": [
            fail_step,
            {"action_type": "READ_STDERR"},
            recover_step,
            {"action_type": "READ_STDOUT"},
        ],
        "requires_failure": True,
        "max_steps": 9,
        "generator": {
            "phase": "phase2cw",
            "stress_id": "bad_arg_then_correct_arg_recover",
            "payload_token": token,
        },
    }


def _build_episode(
    *,
    stress_id: str,
    repository_id: str,
    token: str,
) -> dict[str, Any]:
    if stress_id in PHASE2CO_STRESS_IDS:
        return _build_phase2co_episode(
            stress_id=stress_id,
            repository_id=repository_id,
            token=token,
        )
    if stress_id == "missing_file_then_create_recover":
        return _missing_file_then_create_recover_episode(
            repository_id=repository_id,
            token=token,
        )
    if stress_id == "bad_arg_then_correct_arg_recover":
        return _bad_arg_then_correct_arg_recover_episode(
            repository_id=repository_id,
            token=token,
        )
    raise ValueError(f"unknown phase2cw stress_id: {stress_id}")


def _generate_manifest_for_repository(
    *,
    suite_seed: int,
    repository: dict[str, Any],
) -> dict[str, Any]:
    repository_id = str(repository["repository_id"])
    episodes = [
        _build_episode(
            stress_id=stress_id,
            repository_id=repository_id,
            token=_payload_token(
                seed=suite_seed,
                repository_id=repository_id,
                stress_id=stress_id,
            ),
        )
        for stress_id in STRESS_IDS
    ]
    return {
        "workspace_root": str(repository["workspace_root"]),
        "repetitions_per_episode": 1,
        "generated_by": {
            "phase": "phase2cw",
            "seed": suite_seed,
            "repository_id": repository_id,
            "stress_ids": list(STRESS_IDS),
        },
        "episodes": episodes,
    }


def _prepare_manifest_workspace(manifest: dict[str, Any]) -> None:
    workspace_root = Path(str(manifest["workspace_root"]))
    for episode in manifest["episodes"]:
        recovery_file = episode.get("generator", {}).get("recovery_file")
        if recovery_file:
            path = workspace_root / str(recovery_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()
        for step in episode.get("permissions", []):
            cwd = step.get("cwd")
            if cwd is not None:
                (workspace_root / str(cwd)).mkdir(parents=True, exist_ok=True)


def _failure_recovery_metrics(subreport: dict[str, Any]) -> dict[str, int]:
    episode_reports = subreport.get("episode_reports", [])
    return {
        "failure_episodes": sum(row.get("requires_failure") is True for row in episode_reports),
        "observed_failures": sum(row.get("observed_failure") is True for row in episode_reports),
        "observed_recoveries_after_failure": sum(
            row.get("observed_recovery_after_failure") is True
            for row in episode_reports
        ),
        "recovery_successes": sum(row.get("recovery_success") is True for row in episode_reports),
    }


def run_phase2cw_runtime_perturbation_recovery_stress_expansion(
    *,
    package_path: str | Path,
    suite_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 5.0,
    max_extra_steps: int = 5,
) -> dict[str, Any]:
    suite = json.loads(Path(suite_json).read_text(encoding="utf-8-sig"))
    source = _git_repository_provenance(suite["source_repository_root"])
    seed = int(suite["seed"])
    repositories = suite.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("phase2cw suite requires non-empty repositories")
    output_root = Path(output_dir)
    generated_manifest_dir = output_root / "generated_manifests"
    generated_manifest_dir.mkdir(parents=True, exist_ok=True)
    package_policy = NativeNervousPolicyPackage(
        package_path,
        load_native_head_policy=False,
        load_verification_cortex=False,
    )
    runtime_policy = package_policy.create_structured_runtime_policy()
    package_metadata = package_policy.metadata()

    repo_reports: list[dict[str, Any]] = []
    provenances: list[dict[str, str]] = []
    stress_counts: dict[str, int] = {key: 0 for key in STRESS_IDS}
    all_contract_signatures: set[str] = set()
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
            stress_counts[str(episode["generator"]["stress_id"])] += 1
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
                "stress_ids": manifest["generated_by"]["stress_ids"],
                "contract_signatures": sorted(contract_signatures),
                "passed": subreport["passed"],
                "checks": subreport["checks"],
                "metrics": subreport["metrics"],
                "failure_recovery_metrics": _failure_recovery_metrics(subreport),
                "policy_configuration": subreport.get("policy_configuration") or {},
                "report_json": str(repository_output / "report.json"),
            }
        )

    identity_checks = _repo_identity_checks(
        source=source,
        repositories=provenances,
        minimum_repository_count=int(suite.get("minimum_repository_count", 3)),
    )
    total_episodes = sum(row["metrics"]["episodes"] for row in repo_reports)
    total_observed_recoveries = sum(
        row["failure_recovery_metrics"]["observed_recoveries_after_failure"]
        for row in repo_reports
    )
    total_observed_failures = sum(
        row["failure_recovery_metrics"]["observed_failures"] for row in repo_reports
    )
    total_failure_episodes = sum(
        row["failure_recovery_metrics"]["failure_episodes"] for row in repo_reports
    )
    total_task_completions = sum(
        row["metrics"]["task_completion_successes"] for row in repo_reports
    )
    total_executed_actions = sum(
        row["metrics"]["executed_actions"] for row in repo_reports
    )
    package_internal_passed = all(
        row["policy_configuration"]
        .get("policy_metadata", {})
        .get("package_internal_expert")
        is True
        and row["policy_configuration"]
        .get("policy_metadata", {})
        .get("expert_name")
        == "structured_runtime_cortex"
        for row in repo_reports
    )
    checks = {
        **identity_checks,
        "all_repository_runtime_suites_passed": all(row["passed"] for row in repo_reports),
        "all_repository_actions_were_allowlisted": all(
            row["checks"]["all_model_selected_actions_were_allowlisted"]
            for row in repo_reports
        ),
        "all_repository_task_completion_predicates_satisfied": all(
            row["checks"]["all_task_completion_predicates_satisfied"]
            for row in repo_reports
        ),
        "all_repository_failure_recovery_gates_passed": all(
            row["checks"]["failure_recovery_success_rate_meets_gate"]
            and row["metrics"]["failure_recovery_gate_applicable"] is True
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
        "all_stress_families_present": all(
            stress_counts[key] >= len(repositories) for key in STRESS_IDS
        ),
        "extra_stress_families_present": all(
            stress_counts[key] >= len(repositories)
            for key in PHASE2CW_EXTRA_STRESS_IDS
        ),
        "all_repositories_used_package_internal_runtime_cortex": package_internal_passed,
        "all_episodes_required_failure": total_failure_episodes == total_episodes,
        "all_failure_episodes_observed_failure": total_observed_failures == total_episodes,
        "all_failure_episodes_observed_recovery_after_failure": (
            total_observed_recoveries == total_episodes
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cw_runtime_perturbation_recovery_stress_expansion",
        "passed": passed,
        "ready_for_bounded_expanded_recovery_stress_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
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
        "stress_ids": list(STRESS_IDS),
        "extra_stress_ids": list(PHASE2CW_EXTRA_STRESS_IDS),
        "generated_manifest_dir": str(generated_manifest_dir),
        "generated_contract_signatures": sorted(all_contract_signatures),
        "package_metadata": package_metadata,
        "checks": checks,
        "metrics": {
            "repositories": len(repo_reports),
            "stress_counts": stress_counts,
            "episodes": total_episodes,
            "executed_actions": total_executed_actions,
            "task_completion_successes": total_task_completions,
            "task_completion_success_rate": total_task_completions / max(total_episodes, 1),
            "failure_episodes": total_failure_episodes,
            "observed_failures": total_observed_failures,
            "observed_recoveries_after_failure": total_observed_recoveries,
            "failure_recovery_success_rate": total_observed_recoveries
            / max(total_failure_episodes, 1),
        },
        "repository_reports": repo_reports,
        "supported_claims": [
            "bounded package-internal structured-runtime cortex recovered from expanded file and argument stress families without free-form shell generation"
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
            "phase2cx_expanded_recovery_stress_negative_controls"
            if passed
            else "repair_phase2cw_runtime_perturbation_recovery_stress_expansion"
        ),
    }
    _write_path = Path(output_report_json)
    _write_path.parent.mkdir(parents=True, exist_ok=True)
    _write_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase2CW expanded bounded recovery stress suite."
    )
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--suite-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-extra-steps", type=int, default=5)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2cw_runtime_perturbation_recovery_stress_expansion(
        package_path=args.package_path,
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
