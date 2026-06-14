from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from reflexlm.data import tasks as task_definitions
from reflexlm.data.tasks import TaskType, scenario_templates_for
from reflexlm.llm.receptor_latent import COMMAND_IDENTITY_LATENT_FIELDS
from reflexlm.llm.native_head_training import NSI_LATENT_FIELDS


COMMAND_IDENTITY_FIELD_MARKERS = (
    "command_identity",
    "command_slot_identity",
    "target_command_identity",
    "candidate_command_identity",
    "command_slot:",
)
NONLABEL_PROVENANCE_FORBIDDEN_MARKERS = (
    "answer",
    "correct",
    "gold",
    "label",
    "oracle",
    "sealed",
    "slot target",
    "target slot",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _has_command_identity_latent_field(fields: list[str] | tuple[str, ...]) -> bool:
    lowered = [str(field).lower() for field in fields]
    return any(any(marker in field for marker in COMMAND_IDENTITY_FIELD_MARKERS) for field in lowered)


def _has_required_command_identity_fields(fields: list[str] | tuple[str, ...]) -> bool:
    return set(COMMAND_IDENTITY_LATENT_FIELDS).issubset({str(field) for field in fields})


def _phase2j_profiles_are_explicit_nonsealed(train_profile: str, val_profile: str) -> bool:
    profiles = {train_profile, val_profile}
    allowed_pairs = {
        ("phase2j_semantic_train", "phase2j_semantic_val"),
        ("phase2j_source_overlap_hard_train", "phase2j_source_overlap_hard_val"),
        (
            "phase2j_source_overlap_hard_actiongate_train",
            "phase2j_source_overlap_hard_actiongate_val",
        ),
    }
    return (train_profile, val_profile) in allowed_pairs and not any(
        "sealed" in profile or "external_trace" in profile for profile in profiles
    )


def _command_identity_provenance_is_nonlabel(mechanism: dict[str, Any]) -> bool:
    provenance = str(mechanism.get("command_identity_provenance") or "").lower()
    return (
        bool(provenance)
        and mechanism.get("uses_command_or_slot_identity_latent") is True
        and mechanism.get("derives_identity_from_gold_label") is False
        and mechanism.get("runtime_available_without_target_label") is True
        and not any(marker in provenance for marker in NONLABEL_PROVENANCE_FORBIDDEN_MARKERS)
    )


def _profile_exists(profile: str) -> bool:
    if not profile:
        return False
    const_prefix = re.sub(r"[^0-9A-Za-z]+", "_", profile).upper()
    explicitly_registered = any(
        hasattr(task_definitions, f"{const_prefix}_{suffix}")
        for suffix in ("SCENARIO_TEMPLATES", "SCENARIO_PROFILES")
    )
    return explicitly_registered and bool(scenario_templates_for(TaskType.TEST_FAILURE, profile))


def build_phase2j_implementation_readiness_audit(
    *,
    preregistration_check_json: str | Path,
    proposal_json: str | Path,
    nsi_latent_fields: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    preregistration = _load_json(preregistration_check_json)
    proposal = _load_json(proposal_json)
    mechanism = _section(proposal, "mechanism")
    data_policy = _section(proposal, "data_policy")
    fields = tuple(nsi_latent_fields if nsi_latent_fields is not None else NSI_LATENT_FIELDS)
    train_profile = str(data_policy.get("train_profile") or "phase2j_semantic_train")
    val_profile = str(data_policy.get("val_profile") or "phase2j_semantic_val")

    checks = {
        "phase2j_preregistration_passed": preregistration.get("passed") is True,
        "command_identity_latent_field_present": _has_command_identity_latent_field(fields),
        "command_identity_required_fields_present": _has_required_command_identity_fields(fields),
        "command_identity_provenance_nonlabel": _command_identity_provenance_is_nonlabel(mechanism),
        "phase2j_profiles_are_explicit_nonsealed": _phase2j_profiles_are_explicit_nonsealed(
            train_profile,
            val_profile,
        ),
        "phase2j_train_profile_exists": _profile_exists(train_profile),
        "phase2j_val_profile_exists": _profile_exists(val_profile),
        "preregistration_allows_only_data_audit": preregistration.get("next_action")
        == "prepare_nonsealed_phase2j_data_and_latent_necessity_audit",
    }

    blocked_actions: list[str] = []
    if not checks["phase2j_preregistration_passed"]:
        blocked_actions.append("do_not_execute_phase2j_until_preregistration_passes")
    if not checks["command_identity_latent_field_present"]:
        blocked_actions.append("do_not_generate_phase2j_head_split_until_latent_fields_exist")
    if not checks["command_identity_required_fields_present"]:
        blocked_actions.append("do_not_generate_phase2j_head_split_until_required_identity_fields_exist")
    if not checks["command_identity_provenance_nonlabel"]:
        blocked_actions.append("do_not_generate_phase2j_data_until_nonlabel_identity_provenance_is_recorded")
    if not checks["phase2j_profiles_are_explicit_nonsealed"]:
        blocked_actions.append("do_not_generate_phase2j_data_until_explicit_nonsealed_profiles_are_selected")
    if not checks["phase2j_train_profile_exists"] or not checks["phase2j_val_profile_exists"]:
        blocked_actions.append("do_not_generate_phase2j_data_until_nonsealed_profiles_exist")
    if not checks["preregistration_allows_only_data_audit"]:
        blocked_actions.append("do_not_start_training_from_preregistration")

    passed = all(checks.values())
    return {
        "audit_family": "phase2j_implementation_readiness",
        "passed": passed,
        "ready_for_data_generation": passed,
        "ready_for_training": False,
        "recommended_next_step": (
            "generate_nonsealed_phase2j_data_and_latent_necessity_audit"
            if passed
            else "implement_nonlabel_command_identity_latent_fields_and_phase2j_profiles"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "observations": {
            "train_profile": train_profile,
            "val_profile": val_profile,
            "nsi_latent_dim": len(fields),
            "nsi_latent_fields": list(fields),
        },
        "inputs": {
            "preregistration_check_json": str(Path(preregistration_check_json)),
            "proposal_json": str(Path(proposal_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether Phase2J preregistration is implemented enough for data generation."
    )
    parser.add_argument("--preregistration-check-json", required=True)
    parser.add_argument("--proposal-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_phase2j_implementation_readiness_audit(
        preregistration_check_json=args.preregistration_check_json,
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
