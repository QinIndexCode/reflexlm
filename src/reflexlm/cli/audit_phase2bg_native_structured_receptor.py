from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def audit_phase2bg_native_structured_receptor(
    *,
    phase2bf_audit_json: str | Path,
    normal_summary_json: str | Path,
    erased_summary_json: str | Path,
    wrong_summary_json: str | Path,
    min_rows: int = 12,
    min_normal_success: float = 0.85,
    min_erasure_delta: float = 0.30,
) -> dict[str, Any]:
    phase2bf = _read(phase2bf_audit_json)
    normal = _read(normal_summary_json)
    erased = _read(erased_summary_json)
    wrong = _read(wrong_summary_json)
    rows = int(_number(normal, "rows"))
    normal_success = _number(normal, "success_rate")
    erased_success = _number(erased, "success_rate")
    wrong_success = _number(wrong, "success_rate")
    erasure_delta = normal_success - erased_success
    checks = {
        "phase2bf_format_shift_audit_passed": phase2bf.get("passed") is True,
        "row_counts_match_and_meet_minimum": rows >= min_rows
        and int(_number(erased, "rows")) == rows
        and int(_number(wrong, "rows")) == rows,
        "all_use_structured_receptor_channel": all(
            payload.get("package_runtime_evidence_channel") == "structured_receptor"
            for payload in (normal, erased, wrong)
        ),
        "controls_are_distinct": normal.get("package_runtime_evidence_control") == "normal"
        and erased.get("package_runtime_evidence_control") == "erased"
        and wrong.get("package_runtime_evidence_control") == "wrong",
        "runtime_evidence_absent_from_all_prompts": all(
            _number(payload, "package_runtime_evidence_prompt_present_rows") == 0.0
            for payload in (normal, erased, wrong)
        ),
        "normal_structural_receptor_present": _number(
            normal, "package_structural_probe_receptor_rows"
        )
        == rows,
        "erased_structural_receptor_absent": _number(
            erased, "package_structural_probe_receptor_rows"
        )
        == 0.0,
        "wrong_structural_receptor_present": _number(
            wrong, "package_structural_probe_receptor_rows"
        )
        == rows,
        "normal_success_gate": normal_success >= min_normal_success,
        "erasure_delta_gate": erasure_delta >= min_erasure_delta,
        "wrong_receptor_blocks_execution": wrong_success == 0.0
        and _number(wrong, "execution_attempts") == 0.0,
        "normal_selected_repairs_execute": _number(normal, "attempt_success_rate") == 1.0,
        "real_package_and_qwen_used": all(
            _number(payload, "package_policy_loaded_rows") == rows
            and _number(payload, "package_qwen_called_rows") == rows
            for payload in (normal, erased, wrong)
        ),
        "no_reference_override": all(
            _number(payload, "package_nsi_reference_override_rows") == 0.0
            for payload in (normal, erased, wrong)
        ),
        "no_freeform_or_sealed_feedback": all(
            _number(payload, "freeform_patch_generation_rows") == 0.0
            and _number(payload, "sealed_feedback_used_rows") == 0.0
            for payload in (normal, erased, wrong)
        ),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2bg_native_structured_receptor",
        "passed": passed,
        "ready_for_bounded_native_structured_receptor_claim": passed,
        "ready_for_repo_disjoint_structured_receptor_transfer_claim": False,
        "ready_for_learned_receptor_plasticity_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "rows": rows,
            "normal_success_rate": normal_success,
            "erased_success_rate": erased_success,
            "wrong_success_rate": wrong_success,
            "normal_minus_erased_success_delta": erasure_delta,
            "normal_minus_wrong_success_delta": normal_success - wrong_success,
        },
        "supported_claims": [
            "a native structured SystemStateFrame receptor causally controls bounded fixed-action 7B execution without textual runtime evidence"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "repo-disjoint structured receptor transfer",
            "learned receptor plasticity",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bh_repo_disjoint_multiseed_structured_receptor_transfer"
            if passed
            else "repair_phase2bg_native_structured_receptor_failure"
        ),
        "inputs": {
            "phase2bf_audit_json": str(Path(phase2bf_audit_json)),
            "normal_summary_json": str(Path(normal_summary_json)),
            "erased_summary_json": str(Path(erased_summary_json)),
            "wrong_summary_json": str(Path(wrong_summary_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2BG native structured runtime receptor causality."
    )
    parser.add_argument("--phase2bf-audit-json", required=True)
    parser.add_argument("--normal-summary-json", required=True)
    parser.add_argument("--erased-summary-json", required=True)
    parser.add_argument("--wrong-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=12)
    parser.add_argument("--min-normal-success", type=float, default=0.85)
    parser.add_argument("--min-erasure-delta", type=float, default=0.30)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2bg_native_structured_receptor(
        phase2bf_audit_json=args.phase2bf_audit_json,
        normal_summary_json=args.normal_summary_json,
        erased_summary_json=args.erased_summary_json,
        wrong_summary_json=args.wrong_summary_json,
        min_rows=args.min_rows,
        min_normal_success=args.min_normal_success,
        min_erasure_delta=args.min_erasure_delta,
    )
    _write(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
