from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2aw_architecture_evidence_synthesis(
    *,
    phase2aw_sealed_report_json: str | Path,
    phase2aw_package_matrix_json: str | Path,
    phase2k_freeze_json: str | Path,
    phase2l_freeze_json: str | Path,
) -> dict[str, Any]:
    sealed = _read_json(phase2aw_sealed_report_json)
    matrix = _read_json(phase2aw_package_matrix_json)
    phase2k = _read_json(phase2k_freeze_json)
    phase2l = _read_json(phase2l_freeze_json)
    checks = {
        "phase2aw_sealed_transfer_passed": sealed.get("passed") is True,
        "phase2aw_sealed_no_claim_overreach": sealed.get("ready_for_epoch_making_architecture_claim")
        is False,
        "phase2aw_nonsealed_package_matrix_measured": matrix.get("artifact_family")
        == "phase2aw_package_loaded_mechanism_matrix",
        "phase2aw_nonsealed_full_does_not_beat_native_head_only": matrix.get("checks", {}).get(
            "full_beats_native_head_only"
        )
        is False,
        "phase2k_nonsealed_continuation_positive": phase2k.get("checks", {}).get(
            "full_nonsealed_beats_native_head_only"
        )
        is True,
        "phase2k_sealed_strict_failure_recorded": phase2k.get("checks", {}).get(
            "sealed_gate_failed"
        )
        is True,
        "phase2l_nonsealed_counterfactual_positive": phase2l.get("checks", {}).get(
            "package_postflight_passed"
        )
        is True,
        "phase2l_sealed_transfer_failure_recorded": phase2l.get("checks", {}).get(
            "sealed_gate_failed"
        )
        is True,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2aw_architecture_evidence_synthesis",
        "passed": passed,
        "architecture_claim_status": "bounded_mechanism_supported_not_epoch_making",
        "ready_for_epoch_making_architecture_claim": False,
        "ready_for_open_ended_debugging_claim": False,
        "ready_for_production_autonomy_claim": False,
        "checks": checks,
        "evidence_summary": {
            "phase2aw_sealed_v3": {
                "claim_scope": sealed.get("claim_scope"),
                "full_completion": sealed.get("metrics", {}).get("full_completion"),
                "no_nsi_completion": sealed.get("metrics", {}).get("no_nsi_completion"),
                "native_head_only_completion": sealed.get("metrics", {}).get(
                    "native_head_only_completion"
                ),
                "continuation_only_completion": sealed.get("metrics", {}).get(
                    "continuation_only_completion"
                ),
                "full_minus_no_nsi": sealed.get("metrics", {}).get("full_minus_no_nsi"),
                "full_minus_native_head_only": sealed.get("metrics", {}).get(
                    "full_minus_native_head_only"
                ),
            },
            "phase2aw_package_loaded_nonsealed": {
                "claim_scope": matrix.get("claim_scope"),
                "full_success_rate": matrix.get("metrics", {}).get("full_success_rate"),
                "native_head_only_success_rate": matrix.get("metrics", {}).get(
                    "native_head_only_success_rate"
                ),
                "full_minus_native_head_only": matrix.get("metrics", {}).get(
                    "full_minus_native_head_only"
                ),
                "full_minus_no_nsi": matrix.get("metrics", {}).get("full_minus_no_nsi"),
            },
            "phase2k_nonsealed_continuation_pressure": phase2k.get("metrics", {}).get(
                "nonsealed"
            ),
            "phase2l_nonsealed_counterfactual_continuation": phase2l.get("metrics", {}).get(
                "nonsealed_package"
            ),
        },
        "interpretation": (
            "The current evidence supports a bounded NativeNervousPolicyPackage "
            "mechanism for semantic command-slot and descriptor repair selection. "
            "It does not yet prove that the full package is necessary over native "
            "heads for open-repair execution, because Phase2AW package-loaded "
            "nonsealed execution ties native-head-only. Phase2K/2L show that "
            "continuation can be made diagnostic in preregistered nonsealed tasks, "
            "but those diagnostics have not yet been bridged into the package-loaded "
            "open-repair execution setting."
        ),
        "next_required_experiment": {
            "phase": "phase2ax_package_loaded_counterfactual_repair",
            "goal": (
                "construct a non-sealed package-loaded open-repair runtime split where "
                "current visible repair candidates are identical but prior runtime "
                "evidence changes the correct patch/test/rollback decision"
            ),
            "hard_gates": [
                "repo-origin-disjoint and no sealed overlap",
                "full package success >= 0.85",
                "full minus native-head-only >= 0.10",
                "full minus no-NSI >= 0.15",
                "full minus wrong-cache >= 0.25",
                "full minus cache-erased >= 0.25",
                "source-overlap and native-head-only baselines measured, not declared",
                "no freeform patch generation claim",
            ],
            "stop_if": [
                "native-head-only ties full",
                "controls are all zero",
                "data audit finds candidate marker or hidden/gold leakage",
                "sealed feedback is needed to design the split",
            ],
        },
        "supported_claims": [
            "bounded semantic command-slot / descriptor repair selection mechanism",
            "sealed-v3 Phase2AW positive transfer under final-eval-only rules",
            "NSI/candidate-identity ablation is meaningful on Phase2AW sealed and nonsealed evidence",
        ],
        "unsupported_claims": [
            "epoch-making architecture",
            "production autonomy",
            "open-ended debugging generalization",
            "full package necessity for open-repair execution",
            "continuation memory necessity in package-loaded open-repair execution",
        ],
        "inputs": {
            "phase2aw_sealed_report_json": str(Path(phase2aw_sealed_report_json)),
            "phase2aw_package_matrix_json": str(Path(phase2aw_package_matrix_json)),
            "phase2k_freeze_json": str(Path(phase2k_freeze_json)),
            "phase2l_freeze_json": str(Path(phase2l_freeze_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AW architecture evidence synthesis.")
    parser.add_argument("--phase2aw-sealed-report-json", required=True)
    parser.add_argument("--phase2aw-package-matrix-json", required=True)
    parser.add_argument("--phase2k-freeze-json", required=True)
    parser.add_argument("--phase2l-freeze-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2aw_architecture_evidence_synthesis(
        phase2aw_sealed_report_json=args.phase2aw_sealed_report_json,
        phase2aw_package_matrix_json=args.phase2aw_package_matrix_json,
        phase2k_freeze_json=args.phase2k_freeze_json,
        phase2l_freeze_json=args.phase2l_freeze_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
