from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SEALED_PATH_MARKERS = (
    "external_trace",
    "phase2g_external_trace",
    "phase2i_external_trace",
    "sealed",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _bool(payload: dict[str, Any], key: str, *, default: bool = False) -> bool:
    value = payload.get(key, default)
    return value if isinstance(value, bool) else default


def _path_mentions_sealed(value: Any) -> bool:
    text = str(value).replace("\\", "/").lower()
    if any(marker in text for marker in SEALED_PATH_MARKERS if marker != "sealed"):
        return True
    tokens = [token for token in re.split(r"[/_.\-\s]+", text) if token]
    return "sealed" in tokens


def _sealed_paths(policy: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in (
        "trace_roots",
        "training_roots",
        "validation_roots",
        "tuning_roots",
        "failure_feedback_roots",
    ):
        for value in _list(policy, key):
            if _path_mentions_sealed(value):
                paths.append(str(value))
    return paths


def build_phase2m_default_preregistration() -> dict[str, Any]:
    return {
        "experiment_id": "phase2m_external_generalization_trace_preregistration",
        "phase": "Phase2M",
        "objective": (
            "Evaluate external validity on read-only public or synthetic-safe repo traces "
            "without using sealed v3 failures for training, sampling, tuning, or feedback."
        ),
        "trace_policy": {
            "collection_mode": "read_only",
            "allowed_sources": ["public_repo", "synthetic_safe_repo"],
            "private_repo_allowed": False,
            "requires_network_or_secrets": False,
            "records_repo_url_or_origin": True,
            "records_commit_hash": True,
            "records_license_or_synthetic_origin": True,
            "records_collection_script_hash": True,
            "normalization_is_deterministic": True,
            "redacts_absolute_local_paths": True,
            "redacts_secrets_tokens_and_emails": True,
            "preserves_runtime_visible_evidence": True,
            "forbids_gold_target_or_hidden_hint": True,
            "forbids_sealed_failure_feedback": True,
        },
        "data_policy": {
            "trace_roots": ["artifacts/datasets/phase2m_external_repo_trace_raw"],
            "training_roots": ["artifacts/datasets/phase2m_external_repo_trace_train"],
            "validation_roots": ["artifacts/datasets/phase2m_external_repo_trace_val"],
            "eval_only_roots": ["artifacts/datasets/phase2m_external_repo_trace_holdout"],
            "uses_sealed_for_training": False,
            "uses_sealed_for_tuning": False,
            "uses_sealed_failures_for_analysis_feedback": False,
            "repo_disjoint_holdout_required": True,
            "deduplicate_by_repo_commit_trace_hash": True,
        },
        "benchmark_design": {
            "graded_difficulty_required": True,
            "evidence_density_levels": ["low", "medium", "high"],
            "candidate_counts": [2, 3, 4],
            "trace_types": [
                "test_failure_traceback_to_symbol",
                "changed_file_to_watched_test",
                "module_ownership_to_command",
                "stale_state_refresh",
            ],
            "baselines_required": [
                "source_overlap",
                "native_head_only",
                "continuation_only",
                "prompt_only",
                "react",
            ],
        },
        "gate_policy": {
            "data_health_required": True,
            "source_overlap_baseline_required": True,
            "native_head_only_baseline_required": True,
            "smoke_first_required": True,
            "full_train_requires_smoke_pass": True,
            "package_requires_full_postflight": True,
            "sealed_eval_requires_package": True,
            "full_minus_native_head_only_min": 0.10,
            "full_low_level_qwen_calls": 0,
        },
        "execution_plan": {
            "next_action": "collect_readonly_phase2m_trace_and_run_data_health_only",
            "starts_training": False,
            "starts_package": False,
            "starts_sealed_eval": False,
        },
        "paper_claim": {
            "bounded_until_gates_pass": True,
            "does_not_upgrade_from_phase2l_freeze": True,
            "does_not_claim_open_ended_debugging": True,
        },
    }


def build_phase2m_preregistration_check(*, proposal_json: str | Path) -> dict[str, Any]:
    proposal = _load_json(proposal_json)
    trace_policy = _section(proposal, "trace_policy")
    data_policy = _section(proposal, "data_policy")
    benchmark = _section(proposal, "benchmark_design")
    gates = _section(proposal, "gate_policy")
    execution = _section(proposal, "execution_plan")
    paper_claim = _section(proposal, "paper_claim")
    phase = str(proposal.get("phase") or "").lower().replace(" ", "")
    experiment_id = str(proposal.get("experiment_id") or "").lower()
    sealed_training_paths = _sealed_paths(data_policy)

    checks = {
        "proposal_exists": Path(proposal_json).exists(),
        "experiment_is_phase2m": phase == "phase2m" or experiment_id.startswith("phase2m"),
        "trace_collection_read_only": trace_policy.get("collection_mode") == "read_only",
        "trace_sources_are_public_or_synthetic_safe": (
            set(_list(trace_policy, "allowed_sources")).issubset(
                {"public_repo", "synthetic_safe_repo"}
            )
            and _bool(trace_policy, "private_repo_allowed") is False
            and _bool(trace_policy, "requires_network_or_secrets") is False
        ),
        "provenance_and_license_recorded": all(
            _bool(trace_policy, key)
            for key in (
                "records_repo_url_or_origin",
                "records_commit_hash",
                "records_license_or_synthetic_origin",
                "records_collection_script_hash",
            )
        ),
        "normalization_and_redaction_required": all(
            _bool(trace_policy, key)
            for key in (
                "normalization_is_deterministic",
                "redacts_absolute_local_paths",
                "redacts_secrets_tokens_and_emails",
                "preserves_runtime_visible_evidence",
            )
        ),
        "no_gold_hidden_or_sealed_feedback": (
            _bool(trace_policy, "forbids_gold_target_or_hidden_hint")
            and _bool(trace_policy, "forbids_sealed_failure_feedback")
            and not sealed_training_paths
            and not _bool(data_policy, "uses_sealed_for_training")
            and not _bool(data_policy, "uses_sealed_for_tuning")
            and not _bool(data_policy, "uses_sealed_failures_for_analysis_feedback")
        ),
        "holdout_and_dedup_preregistered": (
            _bool(data_policy, "repo_disjoint_holdout_required")
            and _bool(data_policy, "deduplicate_by_repo_commit_trace_hash")
        ),
        "graded_benchmark_and_baselines_preregistered": (
            _bool(benchmark, "graded_difficulty_required")
            and len(_list(benchmark, "evidence_density_levels")) >= 3
            and len(_list(benchmark, "candidate_counts")) >= 3
            and {"source_overlap", "native_head_only", "continuation_only"}.issubset(
                set(str(item) for item in _list(benchmark, "baselines_required"))
            )
        ),
        "gate_sequence_blocks_training_before_audit": all(
            _bool(gates, key)
            for key in (
                "data_health_required",
                "source_overlap_baseline_required",
                "native_head_only_baseline_required",
                "smoke_first_required",
                "full_train_requires_smoke_pass",
                "package_requires_full_postflight",
                "sealed_eval_requires_package",
            )
        )
        and not _bool(execution, "starts_training")
        and not _bool(execution, "starts_package")
        and not _bool(execution, "starts_sealed_eval"),
        "paper_claim_remains_bounded": (
            _bool(paper_claim, "bounded_until_gates_pass")
            and _bool(paper_claim, "does_not_upgrade_from_phase2l_freeze")
            and _bool(paper_claim, "does_not_claim_open_ended_debugging")
        ),
    }

    blocked_actions: list[str] = []
    if not checks["trace_collection_read_only"]:
        blocked_actions.append("do_not_collect_phase2m_trace_with_write_side_effects")
    if not checks["trace_sources_are_public_or_synthetic_safe"]:
        blocked_actions.append("do_not_use_private_or_secret_dependent_repos")
    if not checks["provenance_and_license_recorded"]:
        blocked_actions.append("do_not_generate_phase2m_data_without_provenance")
    if not checks["normalization_and_redaction_required"]:
        blocked_actions.append("do_not_generate_phase2m_data_without_redaction")
    if not checks["no_gold_hidden_or_sealed_feedback"]:
        blocked_actions.append("do_not_use_gold_hidden_or_sealed_feedback")
    if not checks["holdout_and_dedup_preregistered"]:
        blocked_actions.append("do_not_train_without_repo_disjoint_deduped_holdout")
    if not checks["graded_benchmark_and_baselines_preregistered"]:
        blocked_actions.append("do_not_train_without_graded_phase2m_baselines")
    if not checks["gate_sequence_blocks_training_before_audit"]:
        blocked_actions.append("do_not_start_phase2m_training_before_data_health")
    if not checks["paper_claim_remains_bounded"]:
        blocked_actions.append("do_not_upgrade_paper_claim_before_phase2m_gates")

    passed = all(checks.values())
    next_action = (
        "collect_readonly_phase2m_trace_and_run_data_health_only"
        if passed
        else "revise_phase2m_preregistration_before_trace_collection"
    )
    return {
        "audit_family": "phase2m_external_generalization_preregistration_check",
        "passed": passed,
        "next_action": next_action,
        "allowed_actions": [next_action] if passed else [],
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "observations": {
            "sealed_training_paths": sealed_training_paths,
            "starts_training": _bool(execution, "starts_training"),
            "starts_package": _bool(execution, "starts_package"),
            "starts_sealed_eval": _bool(execution, "starts_sealed_eval"),
            "allowed_sources": _list(trace_policy, "allowed_sources"),
            "baselines_required": _list(benchmark, "baselines_required"),
        },
        "inputs": {"proposal_json": str(Path(proposal_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase2M external generalization preregistration.")
    sub = parser.add_subparsers(dest="command", required=True)
    default = sub.add_parser("default-proposal")
    default.add_argument("--output-json", required=True)
    check = sub.add_parser("check")
    check.add_argument("--proposal-json", required=True)
    check.add_argument("--output-json")
    check.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    if args.command == "default-proposal":
        proposal = build_phase2m_default_preregistration()
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(proposal, indent=2, ensure_ascii=False))
        return

    report = build_phase2m_preregistration_check(proposal_json=args.proposal_json)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
