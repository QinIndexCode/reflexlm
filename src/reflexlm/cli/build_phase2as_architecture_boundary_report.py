from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2as_architecture_boundary_report(
    *,
    symbolic_evidence_report_json: str | Path,
    learned_relabel_audit_json: str | Path,
) -> dict[str, Any]:
    symbolic = _read_json(symbolic_evidence_report_json)
    learned_relabel = _read_json(learned_relabel_audit_json)
    learned_relabel_was_rejected = (
        learned_relabel.get("claimed_capability") == "learned_patch_generation"
        and learned_relabel.get("passed") is False
        and "learned_patch_generation" in set(learned_relabel.get("unsupported_claims") or [])
    )
    checks = {
        "symbolic_evidence_passed": symbolic.get("passed") is True,
        "learned_generation_relabel_rejected": learned_relabel_was_rejected,
        "phase2ar_not_overclaimed_as_learned_generation": (
            "do_not_relabel_symbolic_phase2ar_as_learned_patch_generation"
            in set(learned_relabel.get("blocked_actions") or [])
            or "do_not_relabel_bounded_candidate_selection_as_patch_generation"
            in set(learned_relabel.get("blocked_actions") or [])
        ),
        "sealed_cross_model_transfer_not_claimed": "sealed_cross_model_transfer"
        in set(symbolic.get("unsupported_claims") or []),
        "epoch_making_architecture_not_claimed": "epoch_making_architecture"
        in set(symbolic.get("unsupported_claims") or []),
    }
    passed = all(checks.values())
    supported_claims = []
    if passed:
        supported_claims.extend(
            claim
            for claim in symbolic.get("supported_claims", [])
            if isinstance(claim, str)
            and (
                claim.startswith("bounded_runtime_symbolic")
                or claim.startswith("phase2ar_")
            )
        )
        supported_claims.append("phase2as_claim_boundary_prevents_learned_repair_overclaim")
    unsupported_claims = sorted(
        set(symbolic.get("unsupported_claims") or [])
        | set(learned_relabel.get("unsupported_claims") or [])
        | {
            "learned_patch_generation",
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        }
    )
    blocked_actions = sorted(
        set(symbolic.get("blocked_actions") or [])
        | set(learned_relabel.get("blocked_actions") or [])
        | {
            "do_not_claim_learned_patch_generation_until_native_package_authors_patch_candidates",
            "do_not_claim_epoch_making_architecture_until_learned_generation_and_live_repo_transfer_pass",
        }
    )
    next_required_evidence = [
        "phase2at_native_package_learned_bounded_patch_candidate_generation",
        "phase2at_nonsealed_repo_origin_disjoint_execution_with_nonzero_symbolic_controls",
        "phase2at_multiseed_and_cross_model_reproduction_after_learned_generation_gate",
    ]
    return {
        "artifact_family": "phase2as_architecture_boundary_report",
        "passed": passed,
        "claim_boundary": (
            "Current Phase2AR evidence can support bounded symbolic structural "
            "repair proposal and reproducibility claims. It cannot support learned "
            "patch generation, open-ended debugging generalization, production "
            "autonomy, sealed cross-model transfer, or epoch-making architecture."
        ),
        "checks": checks,
        "supported_claims": sorted(set(supported_claims)),
        "unsupported_claims": unsupported_claims,
        "blocked_actions": blocked_actions,
        "next_required_evidence": next_required_evidence,
        "metrics": {
            "symbolic": symbolic.get("metrics"),
            "learned_relabel": learned_relabel.get("metrics"),
        },
        "inputs": {
            "symbolic_evidence_report_json": str(Path(symbolic_evidence_report_json)),
            "learned_relabel_audit_json": str(Path(learned_relabel_audit_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AS architecture boundary report."
    )
    parser.add_argument("--symbolic-evidence-report-json", required=True)
    parser.add_argument("--learned-relabel-audit-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2as_architecture_boundary_report(
        symbolic_evidence_report_json=args.symbolic_evidence_report_json,
        learned_relabel_audit_json=args.learned_relabel_audit_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
