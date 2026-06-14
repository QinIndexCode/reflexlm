from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
from typing import Any

from reflexlm.runtime.homeostasis import (
    HOMEOSTATIC_CONTROLLER_SCHEMA,
    HOMEOSTATIC_PERSISTENT_STATE_SCHEMA,
    PERSISTENT_STATE_KEYS,
    SIDE_EFFECT_ACTIONS,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _auth_key_bytes(key: str | bytes | None) -> bytes | None:
    if key is None:
        return None
    if isinstance(key, bytes):
        return key
    if not isinstance(key, str) or key == "":
        return None
    return key.encode("utf-8")


def _unsigned_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in artifact.items()
        if key not in {"integrity_sha256", "authenticator"}
    }


def _hmac_digest(payload: dict[str, Any], key: bytes) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _artifact_integrity_valid(
    artifact: dict[str, Any],
    *,
    authenticity_key: str | bytes | None = None,
) -> bool:
    supplied = artifact.get("integrity_sha256")
    unsigned = _unsigned_artifact(artifact)
    if not isinstance(supplied, str) or supplied != _stable_hash(unsigned):
        return False
    authenticator = artifact.get("authenticator")
    if not isinstance(authenticator, dict):
        return False
    algorithm = authenticator.get("algorithm")
    digest = authenticator.get("digest")
    key_fingerprint = authenticator.get("key_fingerprint_sha256")
    if not isinstance(algorithm, str) or not isinstance(digest, str):
        return False
    key = _auth_key_bytes(authenticity_key)
    if algorithm == "sha256":
        return (
            key is None
            and key_fingerprint is None
            and hmac.compare_digest(digest, _stable_hash(unsigned))
        )
    if algorithm == "hmac-sha256":
        if key is None:
            return False
        return (
            key_fingerprint == hashlib.sha256(key).hexdigest()
            and hmac.compare_digest(digest, _hmac_digest(unsigned, key))
        )
    return False


def _authenticator(artifact: dict[str, Any]) -> dict[str, Any]:
    value = artifact.get("authenticator")
    return value if isinstance(value, dict) else {}


def _homeostatic_authenticity_key(env_name: str | None) -> str | None:
    if env_name is None:
        return None
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(
            f"homeostatic authenticity key environment variable is not set: {env_name}"
        )
    return value


def _final_runtime_state(report: dict[str, Any]) -> dict[str, Any]:
    repositories = report.get("repository_reports", [])
    if not repositories:
        return {}
    return (
        repositories[-1]
        .get("policy_configuration", {})
        .get("policy_metadata", {})
        .get("expert_policy", {})
        .get("homeostatic_control", {})
    )


def _persistent_subset(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state.get(key) for key in PERSISTENT_STATE_KEYS}


def _runtime_trace(report: dict[str, Any]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    runtime_python = str(
        report.get("runtime_interpreter")
        or report.get("runtime_environment", {}).get("executable")
        or ""
    )
    for repository in report.get("repository_reports", []):
        report_json = repository.get("report_json")
        if not report_json:
            continue
        subreport = _read_json(report_json)
        for episode in subreport.get("episode_reports", []):
            trace.append(
                {
                    "repository_id": repository.get("repository_id"),
                    "episode_id": episode.get("episode_id"),
                    "actions": [
                        {
                            "type": action.get("type"),
                            "command": _normalize_runtime_command(
                                action.get("command"),
                                runtime_python=runtime_python,
                            ),
                            "file_target": action.get("file_target"),
                            "reason": action.get("reason"),
                        }
                        for action in episode.get("selected_actions", [])
                    ],
                }
            )
    return trace


def _side_effect_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    side_effect_types = {action.value for action in SIDE_EFFECT_ACTIONS}
    return [
        {
            "repository_id": row["repository_id"],
            "episode_id": row["episode_id"],
            "actions": [
                {
                    "type": action["type"],
                    "command": action["command"],
                    "file_target": action["file_target"],
                }
                for action in row["actions"]
                if action["type"] in side_effect_types
            ],
        }
        for row in trace
    ]


def _changed_episode_count(
    generation1_trace: list[dict[str, Any]],
    generation2_trace: list[dict[str, Any]],
) -> int:
    if len(generation1_trace) != len(generation2_trace):
        return max(len(generation1_trace), len(generation2_trace))
    return sum(
        generation1 != generation2
        for generation1, generation2 in zip(
            generation1_trace,
            generation2_trace,
            strict=True,
        )
    )


def _normalize_runtime_command(command: Any, *, runtime_python: str) -> Any:
    if not isinstance(command, str) or not runtime_python:
        return command
    if command == runtime_python:
        return "<RUNTIME_PYTHON>"
    prefix = f"{runtime_python} "
    if command.startswith(prefix):
        return f"<RUNTIME_PYTHON>{command[len(runtime_python):]}"
    return command


def _all_repo_check(report: dict[str, Any], name: str) -> bool:
    repositories = report.get("repository_reports", [])
    return bool(repositories) and all(
        row.get("checks", {}).get(name) is True for row in repositories
    )


def audit_phase2homeostasis_persistent_state_chain(
    *,
    generation1_report_json: str | Path,
    generation2_report_json: str | Path,
    generation1_state_json: str | Path,
    generation2_state_json: str | Path,
    output_report_json: str | Path,
    authenticity_key: str | bytes | None = None,
) -> dict[str, Any]:
    generation1_report = _read_json(generation1_report_json)
    generation2_report = _read_json(generation2_report_json)
    generation1_artifact = _read_json(generation1_state_json)
    generation2_artifact = _read_json(generation2_state_json)
    generation1_state = generation1_artifact.get("state", {})
    generation2_state = generation2_artifact.get("state", {})
    generation1_runtime_state = _final_runtime_state(generation1_report)
    generation2_runtime_state = _final_runtime_state(generation2_report)
    generation1_io = generation1_report.get("homeostatic_state_io", {})
    generation2_io = generation2_report.get("homeostatic_state_io", {})
    generation1_authenticator = _authenticator(generation1_artifact)
    generation2_authenticator = _authenticator(generation2_artifact)
    generation1_trace = _runtime_trace(generation1_report)
    generation2_trace = _runtime_trace(generation2_report)
    selected_action_traces_match = bool(generation1_trace) and (
        generation1_trace == generation2_trace
    )
    side_effect_action_traces_match = bool(generation1_trace) and (
        _side_effect_trace(generation1_trace) == _side_effect_trace(generation2_trace)
    )
    changed_episode_count = _changed_episode_count(
        generation1_trace,
        generation2_trace,
    )
    count_keys = PERSISTENT_STATE_KEYS - {"active_surprise_wake_threshold"}
    config = generation2_artifact.get("config", {})
    baseline = float(config.get("surprise_wake_threshold", -1.0))
    minimum = min(
        baseline,
        float(config.get("minimum_surprise_wake_threshold", -1.0)),
    )
    generation2_threshold = generation2_state.get("active_surprise_wake_threshold")
    checks = {
        "generation1_runtime_passed": generation1_report.get("passed") is True,
        "generation2_runtime_passed": generation2_report.get("passed") is True,
        "same_runtime_matrix_and_metrics": (
            generation1_report.get("seed") == generation2_report.get("seed")
            and generation1_report.get("metrics") == generation2_report.get("metrics")
        ),
        "generation1_saved_state": generation1_io.get("saved") is True,
        "generation2_loaded_and_saved_state": generation2_io.get("loaded") is True
        and generation2_io.get("saved") is True,
        "state_chain_integrity_linked": (
            generation1_io.get("saved_integrity_sha256")
            == generation1_artifact.get("integrity_sha256")
            == generation2_io.get("loaded_integrity_sha256")
            and generation2_io.get("saved_integrity_sha256")
            == generation2_artifact.get("integrity_sha256")
        ),
        "state_chain_authenticator_linked": (
            generation1_io.get("saved_authenticator_algorithm")
            == generation1_authenticator.get("algorithm")
            == generation2_io.get("loaded_authenticator_algorithm")
            and generation2_io.get("saved_authenticator_algorithm")
            == generation2_authenticator.get("algorithm")
            and generation1_io.get("saved_key_fingerprint_sha256")
            == generation1_authenticator.get("key_fingerprint_sha256")
            == generation2_io.get("loaded_key_fingerprint_sha256")
            and generation2_io.get("saved_key_fingerprint_sha256")
            == generation2_authenticator.get("key_fingerprint_sha256")
        ),
        "both_artifact_integrities_valid": _artifact_integrity_valid(
            generation1_artifact,
            authenticity_key=authenticity_key,
        )
        and _artifact_integrity_valid(
            generation2_artifact,
            authenticity_key=authenticity_key,
        ),
        "artifact_schema_config_and_scope_match": (
            generation1_artifact.get("schema_version")
            == generation2_artifact.get("schema_version")
            == HOMEOSTATIC_PERSISTENT_STATE_SCHEMA
            and generation1_artifact.get("controller_schema_version")
            == generation2_artifact.get("controller_schema_version")
            == HOMEOSTATIC_CONTROLLER_SCHEMA
            and generation1_artifact.get("config_fingerprint")
            == generation2_artifact.get("config_fingerprint")
            == _stable_hash(generation2_artifact.get("config", {}))
            and generation1_artifact.get("scope_fingerprint")
            == generation2_artifact.get("scope_fingerprint")
        ),
        "artifact_payloads_contain_only_bounded_persistent_state": (
            isinstance(generation1_state, dict)
            and isinstance(generation2_state, dict)
            and set(generation1_state) == PERSISTENT_STATE_KEYS
            and set(generation2_state) == PERSISTENT_STATE_KEYS
        ),
        "generation1_artifact_matches_runtime_final_state": (
            generation1_state == _persistent_subset(generation1_runtime_state)
        ),
        "generation2_artifact_matches_runtime_final_state": (
            generation2_state == _persistent_subset(generation2_runtime_state)
        ),
        "generation2_persistent_counts_advance": bool(count_keys)
        and all(
            type(generation1_state.get(key)) is int
            and type(generation2_state.get(key)) is int
            and generation2_state[key] > generation1_state[key]
            for key in count_keys
        ),
        "generation2_threshold_within_configured_bounds": (
            isinstance(generation2_threshold, (int, float))
            and not isinstance(generation2_threshold, bool)
            and math.isfinite(float(generation2_threshold))
            and minimum <= float(generation2_threshold) <= baseline
        ),
        "side_effect_action_traces_match": side_effect_action_traces_match,
        "cross_generation_trace_modulation_is_observation_only": (
            selected_action_traces_match or side_effect_action_traces_match
        ),
        "both_runtime_actions_allowlisted": _all_repo_check(
            generation1_report,
            "all_model_selected_actions_were_allowlisted",
        )
        and _all_repo_check(
            generation2_report,
            "all_model_selected_actions_were_allowlisted",
        ),
        "both_runtime_completion_predicates_satisfied": _all_repo_check(
            generation1_report,
            "all_task_completion_predicates_satisfied",
        )
        and _all_repo_check(
            generation2_report,
            "all_task_completion_predicates_satisfied",
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_persistent_state_chain",
        "passed": passed,
        "ready_for_bounded_cross_process_homeostatic_memory_claim": passed,
        "ready_for_unbounded_long_term_memory_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "generation1_integrity_sha256": generation1_artifact.get(
                "integrity_sha256"
            ),
            "generation2_integrity_sha256": generation2_artifact.get(
                "integrity_sha256"
            ),
            "generation1_authenticator_algorithm": generation1_authenticator.get(
                "algorithm"
            ),
            "generation2_authenticator_algorithm": generation2_authenticator.get(
                "algorithm"
            ),
            "generation1_key_fingerprint_sha256": generation1_authenticator.get(
                "key_fingerprint_sha256"
            ),
            "generation2_key_fingerprint_sha256": generation2_authenticator.get(
                "key_fingerprint_sha256"
            ),
            "config_fingerprint": generation2_artifact.get("config_fingerprint"),
            "scope_fingerprint": generation2_artifact.get("scope_fingerprint"),
            "generation1_state": generation1_state,
            "generation2_state": generation2_state,
            "episodes_per_generation": generation1_report.get("metrics", {}).get(
                "episodes"
            ),
            "executed_actions_per_generation": generation1_report.get(
                "metrics", {}
            ).get("executed_actions"),
            "selected_action_traces_match": selected_action_traces_match,
            "changed_episode_count": changed_episode_count,
            "changed_episode_rate": (
                changed_episode_count / len(generation1_trace)
                if generation1_trace
                else 1.0
            ),
        },
        "supported_claims": [
            (
                "a verifier-gated, package-scope-bound, config-bound, "
                "controller-schema-bound, integrity-linked bounded "
                "homeostatic state artifact carried only approved persistent "
                "state across two independent package runtime processes while "
                "preserving the side-effect action trace and full task completion; "
                "any cross-generation action modulation was confined to "
                "observation-only actions"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "unbounded or semantic long-term memory",
            "general cross-machine persistence",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "cross_runtime_persistent_state_transfer"
            if passed
            else "repair_persistent_homeostatic_state_chain"
        ),
        "evidence": {
            "generation1_report_json": str(generation1_report_json),
            "generation2_report_json": str(generation2_report_json),
            "generation1_state_json": str(generation1_state_json),
            "generation2_state_json": str(generation2_state_json),
        },
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a verifier-gated persistent homeostatic state chain."
    )
    parser.add_argument("--generation1-report-json", required=True)
    parser.add_argument("--generation2-report-json", required=True)
    parser.add_argument("--generation1-state-json", required=True)
    parser.add_argument("--generation2-state-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--auth-key-env")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_persistent_state_chain(
        generation1_report_json=args.generation1_report_json,
        generation2_report_json=args.generation2_report_json,
        generation1_state_json=args.generation1_state_json,
        generation2_state_json=args.generation2_state_json,
        output_report_json=args.output_report_json,
        authenticity_key=_homeostatic_authenticity_key(args.auth_key_env),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
