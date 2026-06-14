from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8-sig"))


def _nested_bool(payload: dict[str, Any] | None, *path: str) -> bool | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, bool) else None


def _nested_float(payload: dict[str, Any] | None, *path: str) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def build_phase2i_next_step_decision(
    *,
    latent_necessity_audit_json: str | Path,
    prepackage_gate_json: str | Path | None = None,
    root_cause_json: str | Path | None = None,
    min_full_minus_no_nsi: float = 0.15,
) -> dict[str, Any]:
    latent_audit = _load_json(latent_necessity_audit_json) or {}
    prepackage_gate = _load_json(prepackage_gate_json)
    root_cause = _load_json(root_cause_json)

    latent_audit_passed = latent_audit.get("passed") is True
    architecture_identifiable = _nested_bool(
        latent_audit,
        "checks",
        "nsi_latent_command_identity_available",
    )
    head_latent_coverage = _nested_bool(latent_audit, "checks", "head_latent_coverage")
    latent_source_overlap_not_sufficient = _nested_bool(
        latent_audit,
        "checks",
        "latent_challenge_source_overlap_not_sufficient",
    )
    same_intent_ambiguous = _nested_bool(
        latent_audit,
        "checks",
        "latent_challenge_same_intent_ambiguous",
    )
    prepackage_passed = prepackage_gate.get("passed") if isinstance(prepackage_gate, dict) else None
    full_minus_no_nsi = _nested_float(root_cause, "deltas", "full_minus_no_nsi_latent")
    sealed_delta_passed = (
        None
        if full_minus_no_nsi is None
        else full_minus_no_nsi >= min_full_minus_no_nsi
    )

    blocked_actions: list[str] = []
    allowed_actions: list[str] = []
    required_before_training: list[str] = []
    rationale: list[str] = []

    if architecture_identifiable is False:
        blocked_actions.extend(
            [
                "do_not_retrain_current_phase2i_architecture_for_nsi_latent_claim",
                "do_not_package_or_promote_as_semantic_required_nsi_latent_mechanism",
            ]
        )
        required_before_training.append(
            "pre-register a different mechanism only if NSI latent is allowed to carry command or slot identity"
        )
        rationale.append(
            "same-intent command-slot ambiguity is not identifiable from the current NSI latent fields"
        )
    elif architecture_identifiable is None:
        blocked_actions.append("do_not_train_until_latent_architecture_identifiability_is_audited")
        required_before_training.append("run phase2i_latent_necessity audit with architecture checks")
        rationale.append("latent architecture identifiability was not established")

    if not latent_audit_passed:
        blocked_actions.append("do_not_start_full_pairwise_training")
        required_before_training.append(
            "make non-sealed latent data pass coverage, compression, ambiguity, and source-overlap gates"
        )
        rationale.append("current non-sealed evidence does not isolate NSI latent necessity")

    if sealed_delta_passed is False:
        blocked_actions.append("do_not_upgrade_paper_claim_from_bounded")
        rationale.append(
            f"sealed full-minus-no-NSI delta {full_minus_no_nsi:.6f} is below {min_full_minus_no_nsi:.6f}"
        )

    if latent_audit_passed and architecture_identifiable is True:
        allowed_actions.append("run_small_nonsealed_smoke_before_any_full_training")
        required_before_training.append(
            "record latent necessity audit hash and require smoke val gate before full train"
        )

    if prepackage_passed is False:
        rationale.append("pre-package gate currently rejects this evidence bundle")

    if architecture_identifiable is False:
        recommended_direction = "freeze_phase2i_bounded_claim_and_do_not_retrain_same_architecture"
        next_step = (
            "Use current Phase2I as evidence for continuation cache plus native command head. "
            "Only open a separate Phase2J-style experiment if the paper explicitly changes scope "
            "to test command-identity-bearing NSI latent."
        )
    elif not latent_audit_passed:
        recommended_direction = "repair_nonsealed_latent_identifiability_before_training"
        next_step = (
            "Build or regenerate non-sealed latent train/val evidence until the latent necessity "
            "audit passes, then run smoke only."
        )
    elif sealed_delta_passed is False:
        recommended_direction = "do_not_promote_current_package_run_new_smoke_only_if_mechanism_changes"
        next_step = (
            "The current package remains bounded evidence. A new run is justified only after an "
            "audited mechanism/data change."
        )
    else:
        recommended_direction = "run_nonsealed_smoke_only"
        next_step = "Run a small smoke canary; do not package or sealed-evaluate until smoke passes."

    current_arch_training_allowed = (
        latent_audit_passed
        and architecture_identifiable is True
        and sealed_delta_passed is not False
    )
    claim_upgrade_allowed = (
        current_arch_training_allowed
        and prepackage_passed is True
        and sealed_delta_passed is True
    )

    return {
        "decision_family": "phase2i_next_step_decision",
        "passed": current_arch_training_allowed,
        "recommended_direction": recommended_direction,
        "next_step": next_step,
        "current_architecture_training_allowed": current_arch_training_allowed,
        "paper_claim_upgrade_allowed": claim_upgrade_allowed,
        "allowed_actions": allowed_actions,
        "blocked_actions": sorted(set(blocked_actions)),
        "required_before_training": sorted(set(required_before_training)),
        "rationale": rationale,
        "observations": {
            "latent_audit_passed": latent_audit_passed,
            "architecture_identifiable": architecture_identifiable,
            "head_latent_coverage": head_latent_coverage,
            "latent_source_overlap_not_sufficient": latent_source_overlap_not_sufficient,
            "latent_challenge_same_intent_ambiguous": same_intent_ambiguous,
            "prepackage_gate_passed": prepackage_passed,
            "sealed_full_minus_no_nsi": full_minus_no_nsi,
            "sealed_delta_passed": sealed_delta_passed,
        },
        "inputs": {
            "latent_necessity_audit_json": str(Path(latent_necessity_audit_json)),
            "prepackage_gate_json": str(Path(prepackage_gate_json)) if prepackage_gate_json else None,
            "root_cause_json": str(Path(root_cause_json)) if root_cause_json else None,
            "min_full_minus_no_nsi": min_full_minus_no_nsi,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide the next Phase2I action from audited evidence.")
    parser.add_argument("--latent-necessity-audit-json", required=True)
    parser.add_argument("--prepackage-gate-json")
    parser.add_argument("--root-cause-json")
    parser.add_argument("--min-full-minus-no-nsi", type=float, default=0.15)
    parser.add_argument("--output-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_phase2i_next_step_decision(
        latent_necessity_audit_json=args.latent_necessity_audit_json,
        prepackage_gate_json=args.prepackage_gate_json,
        root_cause_json=args.root_cause_json,
        min_full_minus_no_nsi=args.min_full_minus_no_nsi,
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
