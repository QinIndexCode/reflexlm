from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any

from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)
from reflexlm.cli.run_phase2bo_repo_disjoint_model_selected_sealed_runtime import (
    _git_repository_provenance,
    _repo_identity_checks,
)
from reflexlm.llm.native_nervous_package import NativeNervousPolicyPackage


RECIPE_IDS = (
    "file_terminal_fusion",
    "timeout_stderr_recovery",
    "failure_dual_observe_recover",
)


def _payload_token(*, seed: int, repository_id: str, recipe_id: str, index: int) -> str:
    digest = hashlib.sha256(
        f"phase2bq|{seed}|{repository_id}|{recipe_id}|{index}".encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _contract_signature(episode: dict[str, Any]) -> str:
    permissions = episode.get("permissions", [])
    completion = episode.get("completion_requirements", [])
    permission_types = ",".join(str(row["action_type"]) for row in permissions)
    completion_types = ",".join(str(row["action_type"]) for row in completion)
    return f"permissions=[{permission_types}] completion=[{completion_types}]"


def _homeostatic_authenticity_key(env_name: str | None) -> str | None:
    if env_name is None:
        return None
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(
            f"homeostatic authenticity key environment variable is not set: {env_name}"
        )
    return value


def _file_terminal_fusion_episode(
    *,
    repository_id: str,
    token: str,
) -> dict[str, Any]:
    probe = f".reflexlm_runtime_probe/{repository_id}-{token}-fusion.txt"
    write_code = (
        "from pathlib import Path; import sys; "
        f"p=Path({probe!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        f"p.write_text('fusion-file-{token}', encoding='utf-8'); "
        f"sys.stderr.write('fusion-stderr-{token}\\n'); "
        f"print('fusion-stdout-{token}')"
    )
    run_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", write_code],
        "watched_paths": [probe],
        "expected_exit_code": 0,
    }
    return {
        "episode_id": f"phase2bq-{repository_id}-file-terminal-fusion-{token}",
        "description": (
            "Procedurally generated file and terminal receptor fusion task."
        ),
        "task_type": "external_file_change_reflex",
        "watched_paths": [probe],
        "ambient_observation_actions": ["READ_STDOUT", "READ_STDERR"],
        "permissions": [
            run_step,
            {"action_type": "REFRESH_STATE", "watched_paths": [probe]},
            {"action_type": "READ_FILE", "file_target": probe, "watched_paths": [probe]},
            {"action_type": "READ_STDERR"},
            {"action_type": "READ_STDOUT"},
            {"action_type": "DONE"},
        ],
        "completion_requirements": [
            run_step,
            {"action_type": "READ_FILE", "file_target": probe},
            {"action_type": "READ_STDERR"},
            {"action_type": "READ_STDOUT"},
        ],
        "requires_failure": False,
        "max_steps": 9,
        "generator": {
            "phase": "phase2bq",
            "recipe_id": "file_terminal_fusion",
            "payload_token": token,
        },
    }


def _timeout_stderr_recovery_episode(
    *,
    repository_id: str,
    token: str,
    timeout_seconds: float = 0.5,
) -> dict[str, Any]:
    timeout_code = (
        "import sys, time; "
        f"sys.stderr.write('timeout-stderr-{token}\\n'); sys.stderr.flush(); "
        "time.sleep(2)"
    )
    recovery_code = f"print('timeout-recovery-stdout-{token}')"
    fail_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", timeout_code],
        "timeout_seconds": timeout_seconds,
        "expected_exit_code": 124,
        "expected_timed_out": True,
    }
    recover_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", recovery_code],
        "expected_exit_code": 0,
    }
    return {
        "episode_id": f"phase2bq-{repository_id}-timeout-stderr-recovery-{token}",
        "description": (
            "Procedurally generated timeout, diagnostic, refresh, and recovery task."
        ),
        "task_type": "process_hang_detection",
        "ambient_observation_actions": ["READ_STDOUT", "READ_STDERR"],
        "permissions": [
            fail_step,
            {"action_type": "READ_STDERR"},
            {"action_type": "WAIT", "wait_ms": 10},
            {"action_type": "REFRESH_STATE"},
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
        "max_steps": 10,
        "generator": {
            "phase": "phase2bq",
            "recipe_id": "timeout_stderr_recovery",
            "payload_token": token,
        },
    }


def _failure_dual_observe_recover_episode(
    *,
    repository_id: str,
    token: str,
) -> dict[str, Any]:
    fail_code = (
        "import sys; "
        f"sys.stderr.write('failure-stderr-{token}\\n'); "
        f"print('failure-stdout-{token}'); "
        "raise SystemExit(17)"
    )
    recovery_code = f"print('failure-recovery-stdout-{token}')"
    fail_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", fail_code],
        "expected_exit_code": 17,
    }
    recover_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", recovery_code],
        "expected_exit_code": 0,
    }
    return {
        "episode_id": f"phase2bq-{repository_id}-failure-dual-recover-{token}",
        "description": (
            "Procedurally generated failed command with dual-channel observation before recovery."
        ),
        "task_type": "test_failure_reflex",
        "ambient_observation_actions": ["READ_STDOUT", "READ_STDERR"],
        "permissions": [
            fail_step,
            {"action_type": "READ_STDERR"},
            {"action_type": "READ_STDOUT"},
            recover_step,
            {"action_type": "DONE"},
        ],
        "completion_requirements": [
            fail_step,
            {"action_type": "READ_STDERR"},
            {"action_type": "READ_STDOUT"},
            recover_step,
        ],
        "requires_failure": True,
        "max_steps": 9,
        "generator": {
            "phase": "phase2bq",
            "recipe_id": "failure_dual_observe_recover",
            "payload_token": token,
        },
    }


def _build_episode(
    *,
    recipe_id: str,
    repository_id: str,
    token: str,
    timeout_recovery_command_timeout_seconds: float = 0.5,
) -> dict[str, Any]:
    if recipe_id == "file_terminal_fusion":
        return _file_terminal_fusion_episode(repository_id=repository_id, token=token)
    if recipe_id == "timeout_stderr_recovery":
        return _timeout_stderr_recovery_episode(
            repository_id=repository_id,
            token=token,
            timeout_seconds=timeout_recovery_command_timeout_seconds,
        )
    if recipe_id == "failure_dual_observe_recover":
        return _failure_dual_observe_recover_episode(
            repository_id=repository_id,
            token=token,
        )
    raise ValueError(f"unknown phase2bq recipe_id: {recipe_id}")


def _generate_manifest_for_repository(
    *,
    suite_seed: int,
    repository: dict[str, Any],
    recipes_per_repository: int,
    repetitions_per_episode: int,
    timeout_recovery_command_timeout_seconds: float = 0.5,
) -> dict[str, Any]:
    repository_id = str(repository["repository_id"])
    rng = random.Random(f"phase2bq|{suite_seed}|{repository_id}")
    recipe_pool = list(RECIPE_IDS)
    rng.shuffle(recipe_pool)
    recipe_ids = recipe_pool[:recipes_per_repository]
    episodes = [
        _build_episode(
            recipe_id=recipe_id,
            repository_id=repository_id,
            token=_payload_token(
                seed=suite_seed,
                repository_id=repository_id,
                recipe_id=recipe_id,
                index=index,
            ),
            timeout_recovery_command_timeout_seconds=(
                timeout_recovery_command_timeout_seconds
            ),
        )
        for index, recipe_id in enumerate(recipe_ids)
    ]
    return {
        "workspace_root": str(repository["workspace_root"]),
        "repetitions_per_episode": repetitions_per_episode,
        "generated_by": {
            "phase": "phase2bq",
            "seed": suite_seed,
            "repository_id": repository_id,
            "recipe_ids": recipe_ids,
            "timeout_recovery_command_timeout_seconds": (
                timeout_recovery_command_timeout_seconds
            ),
        },
        "episodes": episodes,
    }


def _load_training_signatures(training_manifest_json: str | Path) -> set[str]:
    manifest = json.loads(Path(training_manifest_json).read_text(encoding="utf-8-sig"))
    signatures: set[str] = set()
    for episode in manifest.get("episodes", []):
        steps = episode.get("steps", [])
        signatures.add(" -> ".join(str(step["action_type"]) for step in steps))
    return signatures


def run_phase2bq_open_task_family_repo_runtime(
    *,
    checkpoint_path: str | Path | None,
    package_path: str | Path | None = None,
    suite_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
    max_extra_steps: int = 5,
    package_device: str | None = None,
    package_quantization: str | None = None,
    package_model_load_strategy: str | None = None,
    package_offload_state_dict: bool | None = None,
    package_online_homeostatic_adaptation: bool | None = None,
    package_cross_episode_homeostatic_memory: bool | None = None,
    package_homeostatic_state_input: str | Path | None = None,
    package_homeostatic_state_output: str | Path | None = None,
    package_homeostatic_auth_key_env: str | None = None,
) -> dict[str, Any]:
    suite_path = Path(suite_json).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8-sig"))
    source = _git_repository_provenance(suite["source_repository_root"])
    training_manifest_path = Path(str(suite["training_manifest_json"]))
    if not training_manifest_path.is_absolute():
        training_manifest_path = suite_path.parent / training_manifest_path
    training_signatures = _load_training_signatures(training_manifest_path)
    repositories = suite.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("phase2bq suite requires non-empty repositories")
    seed = int(suite["seed"])
    recipes_per_repository = int(suite.get("recipes_per_repository", len(RECIPE_IDS)))
    if not 1 <= recipes_per_repository <= len(RECIPE_IDS):
        raise ValueError("recipes_per_repository is outside the recipe pool")
    repetitions_per_episode = int(suite.get("repetitions_per_episode", 2))
    minimum_repository_count = int(suite.get("minimum_repository_count", 3))
    timeout_recovery_command_timeout_seconds = float(
        suite.get("timeout_recovery_command_timeout_seconds", 0.5)
    )
    if not 0.0 < timeout_recovery_command_timeout_seconds < 2.0:
        raise ValueError(
            "timeout_recovery_command_timeout_seconds must be > 0 and < 2.0"
        )
    output_root = Path(output_dir)
    generated_manifest_dir = output_root / "generated_manifests"
    generated_manifest_dir.mkdir(parents=True, exist_ok=True)
    package_policy = (
        NativeNervousPolicyPackage(
            package_path,
            device=package_device,
            quantization=package_quantization,
            model_load_strategy=package_model_load_strategy,
            offload_state_dict=package_offload_state_dict,
            load_native_head_policy=False,
            load_verification_cortex=False,
        )
        if package_path is not None
        else None
    )
    runtime_policy = None
    homeostatic_authenticity_key = _homeostatic_authenticity_key(
        package_homeostatic_auth_key_env
    )
    homeostatic_state_io: dict[str, Any] = {
        "input_path": (
            str(package_homeostatic_state_input)
            if package_homeostatic_state_input is not None
            else None
        ),
        "output_path": (
            str(package_homeostatic_state_output)
            if package_homeostatic_state_output is not None
            else None
        ),
        "loaded": False,
        "saved": False,
        "loaded_integrity_sha256": None,
        "saved_integrity_sha256": None,
        "auth_key_env": package_homeostatic_auth_key_env,
        "loaded_authenticator_algorithm": None,
        "saved_authenticator_algorithm": None,
        "loaded_key_fingerprint_sha256": None,
        "saved_key_fingerprint_sha256": None,
    }
    if package_policy is not None:
        homeostatic_overrides = {
            key: value
            for key, value in {
                "enable_online_homeostatic_adaptation": (
                    package_online_homeostatic_adaptation
                ),
                "enable_cross_episode_homeostatic_memory": (
                    package_cross_episode_homeostatic_memory
                ),
            }.items()
            if value is not None
        }
        if not homeostatic_overrides:
            runtime_policy = package_policy.create_structured_runtime_policy()
        else:
            runtime_policy = package_policy.create_structured_runtime_policy(
                **homeostatic_overrides
            )
        if package_homeostatic_state_input is not None:
            loaded_state = runtime_policy.load_homeostatic_state(
                package_homeostatic_state_input,
                authenticity_key=homeostatic_authenticity_key,
            )
            homeostatic_state_io["loaded"] = True
            homeostatic_state_io["loaded_integrity_sha256"] = loaded_state.get(
                "integrity_sha256"
            )
            loaded_authenticator = loaded_state.get("authenticator", {})
            homeostatic_state_io["loaded_authenticator_algorithm"] = (
                loaded_authenticator.get("algorithm")
                if isinstance(loaded_authenticator, dict)
                else None
            )
            homeostatic_state_io["loaded_key_fingerprint_sha256"] = (
                loaded_authenticator.get("key_fingerprint_sha256")
                if isinstance(loaded_authenticator, dict)
                else None
            )
    elif (
        package_homeostatic_state_input is not None
        or package_homeostatic_state_output is not None
    ):
        raise ValueError("homeostatic state input/output requires package runtime")
    package_metadata = package_policy.metadata() if package_policy is not None else {}

    repo_reports: list[dict[str, Any]] = []
    provenances: list[dict[str, str]] = []
    all_contract_signatures: set[str] = set()
    all_payload_tokens: list[str] = []
    for repository in repositories:
        repository_id = str(repository["repository_id"])
        manifest = _generate_manifest_for_repository(
            suite_seed=seed,
            repository=repository,
            recipes_per_repository=recipes_per_repository,
            repetitions_per_episode=repetitions_per_episode,
            timeout_recovery_command_timeout_seconds=(
                timeout_recovery_command_timeout_seconds
            ),
        )
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
        all_payload_tokens.extend(
            str(episode["generator"]["payload_token"]) for episode in manifest["episodes"]
        )
        repository_output = output_root / repository_id
        subreport = run_phase2bn_model_selected_sealed_runtime(
            checkpoint_path=None if runtime_policy is not None else checkpoint_path,
            manifest_json=manifest_path,
            output_jsonl=repository_output / "trajectories.jsonl",
            output_report_json=repository_output / "report.json",
            timeout_seconds=timeout_seconds,
            max_extra_steps=max_extra_steps,
            policy_instance=runtime_policy,
            policy_label=(
                "package_internal_structured_runtime_cortex"
                if runtime_policy is not None
                else "phase2bn_model_selected_sealed_runtime"
            ),
        )
        repo_reports.append(
            {
                "repository_id": repository_id,
                "generated_manifest_json": str(manifest_path),
                "provenance": provenance,
                "recipe_ids": manifest["generated_by"]["recipe_ids"],
                "contract_signatures": sorted(contract_signatures),
                "passed": subreport["passed"],
                "checks": subreport["checks"],
                "metrics": subreport["metrics"],
                "policy_configuration": subreport.get("policy_configuration") or {},
                "report_json": str(repository_output / "report.json"),
            }
        )

    if package_homeostatic_state_output is not None:
        saved_state = runtime_policy.save_homeostatic_state(
            package_homeostatic_state_output,
            authenticity_key=homeostatic_authenticity_key,
        )
        homeostatic_state_io["saved"] = True
        homeostatic_state_io["saved_integrity_sha256"] = saved_state.get(
            "integrity_sha256"
        )
        saved_authenticator = saved_state.get("authenticator", {})
        homeostatic_state_io["saved_authenticator_algorithm"] = (
            saved_authenticator.get("algorithm")
            if isinstance(saved_authenticator, dict)
            else None
        )
        homeostatic_state_io["saved_key_fingerprint_sha256"] = (
            saved_authenticator.get("key_fingerprint_sha256")
            if isinstance(saved_authenticator, dict)
            else None
        )

    identity_checks = _repo_identity_checks(
        source=source,
        repositories=provenances,
        minimum_repository_count=minimum_repository_count,
    )
    generated_manifests_have_no_steps = all(
        "steps" not in episode
        and "permissions" in episode
        and "completion_requirements" in episode
        and "expected_sequence" not in episode
        for path in generated_manifest_dir.glob("*.json")
        for episode in json.loads(path.read_text(encoding="utf-8"))["episodes"]
    )
    generated_action_sequence_overlap = sorted(
        signature
        for signature in all_contract_signatures
        if signature in training_signatures
    )
    checks = {
        **identity_checks,
        "generated_manifests_have_no_expected_sequence_or_steps": generated_manifests_have_no_steps,
        "generated_payload_tokens_are_unique": len(set(all_payload_tokens))
        == len(all_payload_tokens),
        "generated_contract_signatures_do_not_equal_training_action_sequences": not generated_action_sequence_overlap,
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
            package_policy is None
            or package_metadata.get("structured_runtime_cortex_packaged") is True
        ),
        "all_repositories_used_package_internal_runtime_cortex": (
            package_policy is None
            or all(
                row.get("policy_configuration", {})
                .get("policy_metadata", {})
                .get("package_internal_expert")
                is True
                and row.get("policy_configuration", {})
                .get("policy_metadata", {})
                .get("expert_name")
                == "structured_runtime_cortex"
                for row in repo_reports
            )
        ),
        "requested_homeostatic_state_loaded": (
            package_homeostatic_state_input is None
            or homeostatic_state_io["loaded"] is True
        ),
        "requested_homeostatic_state_saved": (
            package_homeostatic_state_output is None
            or homeostatic_state_io["saved"] is True
        ),
    }
    passed = all(checks.values())
    package_internal_passed = (
        passed
        and package_policy is not None
        and checks["package_structured_runtime_cortex_packaged"]
        and checks["all_repositories_used_package_internal_runtime_cortex"]
    )
    total_episodes = sum(row["metrics"]["episodes"] for row in repo_reports)
    total_executed_actions = sum(
        row["metrics"]["executed_actions"] for row in repo_reports
    )
    total_task_completions = sum(
        row["metrics"]["task_completion_successes"] for row in repo_reports
    )
    report = {
        "artifact_family": (
            "phase2ci_unified_package_open_task_family_repo_runtime"
            if package_policy is not None
            else "phase2bq_open_task_family_repo_runtime"
        ),
        "passed": passed,
        "ready_for_bounded_generated_task_family_runtime_claim": passed,
        "ready_for_unified_package_generated_task_family_runtime_claim": (
            package_internal_passed
        ),
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
        "recipe_pool": list(RECIPE_IDS),
        "training_manifest_json": str(training_manifest_path),
        "generated_manifest_dir": str(generated_manifest_dir),
        "timeout_recovery_command_timeout_seconds": (
            timeout_recovery_command_timeout_seconds
        ),
        "generated_contract_signatures": sorted(all_contract_signatures),
        "generated_action_sequence_overlap": generated_action_sequence_overlap,
        "package_metadata": package_metadata,
        "homeostatic_state_io": homeostatic_state_io,
        "checks": checks,
        "metrics": {
            "repositories": len(repo_reports),
            "generated_episode_templates": len(all_payload_tokens),
            "episodes": total_episodes,
            "executed_actions": total_executed_actions,
            "task_completion_successes": total_task_completions,
            "task_completion_success_rate": total_task_completions
            / max(total_episodes, 1),
        },
        "repository_reports": repo_reports,
        "supported_claims": [
            (
                "the unified multi-cortical package completed seed-generated "
                "repo-disjoint runtime task families with its packaged structured "
                "runtime cortex and without expected action sequences in generated "
                "manifests"
                if package_internal_passed
                else "the bounded architecture completed seed-generated "
                "repo-disjoint runtime task families without expected action "
                "sequences in the generated manifests"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "unbounded task generation",
            "runtime-interpreter-invariant structured control",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cj_runtime_interpreter_invariance"
            if passed
            else "repair_phase2bq_open_task_family_repo_runtime"
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
        description="Generate and run bounded open task-family runtime suites across independent Git repositories."
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--package-path")
    parser.add_argument("--package-device")
    parser.add_argument("--package-quantization")
    parser.add_argument("--package-model-load-strategy")
    parser.add_argument(
        "--package-offload-state-dict",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--package-online-homeostatic-adaptation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--package-cross-episode-homeostatic-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--package-homeostatic-state-input")
    parser.add_argument("--package-homeostatic-state-output")
    parser.add_argument("--package-homeostatic-auth-key-env")
    parser.add_argument("--suite-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-extra-steps", type=int, default=5)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bq_open_task_family_repo_runtime(
        checkpoint_path=args.checkpoint_path,
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
        package_online_homeostatic_adaptation=(
            args.package_online_homeostatic_adaptation
        ),
        package_cross_episode_homeostatic_memory=(
            args.package_cross_episode_homeostatic_memory
        ),
        package_homeostatic_state_input=args.package_homeostatic_state_input,
        package_homeostatic_state_output=args.package_homeostatic_state_output,
        package_homeostatic_auth_key_env=args.package_homeostatic_auth_key_env,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
