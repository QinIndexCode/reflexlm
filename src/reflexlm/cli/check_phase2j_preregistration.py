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
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _bool(payload: dict[str, Any], key: str, *, default: bool = False) -> bool:
    value = payload.get(key, default)
    return value if isinstance(value, bool) else default


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _path_mentions_sealed(value: Any) -> bool:
    text = str(value).replace("\\", "/").lower()
    if any(marker in text for marker in SEALED_PATH_MARKERS):
        return True
    tokens = [token for token in re.split(r"[/_.\-\s]+", text) if token]
    return "sealed" in tokens


def _sealed_training_paths(data_policy: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("training_roots", "train_roots", "tuning_roots", "validation_roots"):
        for value in _list(data_policy, key):
            if _path_mentions_sealed(value):
                paths.append(str(value))
    return paths


def _decision_blocks_same_architecture(decision: dict[str, Any]) -> bool:
    blocked = set(decision.get("blocked_actions") or [])
    return (
        decision.get("current_architecture_training_allowed") is False
        or "do_not_retrain_current_phase2i_architecture_for_nsi_latent_claim" in blocked
    )


def _phase2j_named(proposal: dict[str, Any]) -> bool:
    experiment_id = str(proposal.get("experiment_id") or "").lower()
    phase = str(proposal.get("phase") or "").lower()
    return phase.replace(" ", "") == "phase2j" or experiment_id.startswith("phase2j")


def _command_identity_provenance_allowed(mechanism: dict[str, Any]) -> bool:
    provenance = str(mechanism.get("command_identity_provenance") or "").lower()
    forbidden_terms = ("gold", "label", "target", "answer", "sealed", "test_name")
    allowed_source = any(
        term in provenance for term in ("runtime", "receptor", "static", "observed", "evidence")
    )
    return (
        allowed_source
        and not any(term in provenance for term in forbidden_terms)
        and not _bool(mechanism, "derives_identity_from_gold_label")
        and _bool(mechanism, "runtime_available_without_target_label")
        and _bool(mechanism, "generalizes_beyond_named_tests")
    )


def build_phase2j_preregistration_check(
    *,
    phase2i_decision_json: str | Path,
    proposal_json: str | Path,
) -> dict[str, Any]:
    decision = _load_json(phase2i_decision_json)
    proposal = _load_json(proposal_json)
    mechanism = _section(proposal, "mechanism")
    data_policy = _section(proposal, "data_policy")
    gate_policy = _section(proposal, "gate_policy")
    execution_plan = _section(proposal, "execution_plan")
    paper_claim = _section(proposal, "paper_claim")

    phase2i_blocks_same_arch = _decision_blocks_same_architecture(decision)
    phase2j_named = _phase2j_named(proposal)
    uses_current_phase2i_architecture = _bool(mechanism, "uses_current_phase2i_architecture")
    uses_command_identity_latent = _bool(mechanism, "uses_command_or_slot_identity_latent")
    mechanism_scope_changed = _bool(mechanism, "mechanism_scope_changed")
    starts_full_training = _bool(execution_plan, "starts_full_training")
    starts_package = _bool(execution_plan, "starts_package")
    starts_sealed_eval = _bool(execution_plan, "starts_sealed_eval")
    sealed_training_paths = _sealed_training_paths(data_policy)

    checks = {
        "proposal_exists": Path(proposal_json).exists(),
        "phase2i_decision_blocks_same_architecture": phase2i_blocks_same_arch,
        "experiment_is_separate_phase2j": phase2j_named,
        "not_same_architecture_retrain": (
            True
            if not phase2i_blocks_same_arch
            else not uses_current_phase2i_architecture and phase2j_named
        ),
        "command_identity_latent_is_scope_changed": (
            True
            if not uses_command_identity_latent
            else mechanism_scope_changed and phase2j_named
        ),
        "command_identity_provenance_is_non_label": (
            True
            if not uses_command_identity_latent
            else _command_identity_provenance_allowed(mechanism)
        ),
        "no_sealed_training_or_tuning": (
            not sealed_training_paths
            and not _bool(data_policy, "uses_sealed_for_training")
            and not _bool(data_policy, "uses_sealed_for_tuning")
            and not _bool(data_policy, "uses_sealed_failures_for_analysis_feedback")
        ),
        "nonsealed_gate_policy_preregistered": all(
            _bool(gate_policy, key)
            for key in (
                "latent_necessity_audit_required",
                "effective_split_hash_required",
                "source_overlap_baseline_required",
                "same_intent_ambiguity_required",
                "train_val_intent_coverage_required",
            )
        ),
        "smoke_before_full_training": (
            not starts_full_training
            and _bool(gate_policy, "smoke_first_required")
            and _bool(gate_policy, "full_train_requires_smoke_pass")
        ),
        "no_package_or_sealed_eval_before_gates": (
            not starts_package
            and not starts_sealed_eval
            and _bool(gate_policy, "package_requires_prepackage_gate")
            and _bool(gate_policy, "sealed_eval_requires_package")
        ),
        "paper_claim_remains_bounded": (
            not _bool(paper_claim, "upgrades_phase2i_claim")
            and _bool(paper_claim, "bounded_until_gates_pass")
            and _bool(paper_claim, "separate_phase2j_claim_required")
        ),
    }

    blocked_actions: list[str] = []
    if not checks["not_same_architecture_retrain"]:
        blocked_actions.append("do_not_retrain_current_phase2i_architecture")
    if not checks["command_identity_latent_is_scope_changed"]:
        blocked_actions.append("do_not_add_command_identity_latent_under_phase2i_claim")
    if not checks["command_identity_provenance_is_non_label"]:
        blocked_actions.append("do_not_use_gold_label_test_name_or_sealed_failure_as_latent_identity")
    if not checks["no_sealed_training_or_tuning"]:
        blocked_actions.append("do_not_use_sealed_data_for_training_tuning_or_feedback")
    if not checks["smoke_before_full_training"]:
        blocked_actions.append("do_not_start_full_training_before_nonsealed_smoke")
    if not checks["no_package_or_sealed_eval_before_gates"]:
        blocked_actions.append("do_not_package_or_run_sealed_eval_before_gates")
    if not checks["paper_claim_remains_bounded"]:
        blocked_actions.append("do_not_upgrade_phase2i_claim")

    passed = all(checks.values())
    next_action = (
        "prepare_nonsealed_phase2j_data_and_latent_necessity_audit"
        if passed
        else "revise_phase2j_preregistration_before_execution"
    )
    return {
        "audit_family": "phase2j_preregistration_check",
        "passed": passed,
        "next_action": next_action,
        "allowed_actions": [next_action] if passed else [],
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "observations": {
            "phase2i_blocks_same_architecture": phase2i_blocks_same_arch,
            "phase2j_named": phase2j_named,
            "uses_current_phase2i_architecture": uses_current_phase2i_architecture,
            "uses_command_or_slot_identity_latent": uses_command_identity_latent,
            "mechanism_scope_changed": mechanism_scope_changed,
            "command_identity_provenance": mechanism.get("command_identity_provenance"),
            "sealed_training_paths": sealed_training_paths,
            "starts_full_training": starts_full_training,
            "starts_package": starts_package,
            "starts_sealed_eval": starts_sealed_eval,
        },
        "inputs": {
            "phase2i_decision_json": str(Path(phase2i_decision_json)),
            "proposal_json": str(Path(proposal_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase2J preregistration before any training.")
    parser.add_argument("--phase2i-decision-json", required=True)
    parser.add_argument("--proposal-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_phase2j_preregistration_check(
        phase2i_decision_json=args.phase2i_decision_json,
        proposal_json=args.proposal_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
