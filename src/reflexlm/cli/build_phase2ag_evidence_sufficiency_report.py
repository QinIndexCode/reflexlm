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


def _metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get("metrics", {}).get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_phase2ag_evidence_sufficiency_report(
    *,
    postflight_json: str | Path,
    sidecar_control_postflight_json: str | Path,
) -> dict[str, Any]:
    postflight = _read_json(postflight_json)
    controls = _read_json(sidecar_control_postflight_json)
    val_delta = _metric(postflight, "val_model_minus_source_overlap_accuracy")
    holdout_delta = _metric(postflight, "holdout_model_minus_source_overlap_accuracy")
    full_minus_erased = _metric(controls, "full_minus_sidecar_erased")
    full_minus_wrong = _metric(controls, "full_minus_wrong_sidecar")
    require_val_delta = (
        postflight.get("thresholds", {}).get("require_val_model_minus_source_overlap")
        is True
    )
    checks = {
        "postflight_passed": postflight.get("passed") is True,
        "sidecar_controls_passed": controls.get("passed") is True,
        "holdout_delta_positive": isinstance(holdout_delta, float) and holdout_delta > 0.0,
        "sidecar_erased_degrades": isinstance(full_minus_erased, float)
        and full_minus_erased > 0.0,
        "wrong_sidecar_guarded_degrades": isinstance(full_minus_wrong, float)
        and full_minus_wrong > 0.0,
        "package_blocked": postflight.get("ready_for_package") is False
        and controls.get("ready_for_package") is False,
        "sealed_eval_blocked": postflight.get("ready_for_sealed_eval") is False
        and controls.get("ready_for_sealed_eval") is False,
        "claim_bearing_blocked": postflight.get("claim_bearing_mechanism_evidence")
        is False
        and controls.get("claim_bearing_mechanism_evidence") is False,
    }
    passed = all(checks.values())
    val_ceiling_caveat = (
        not require_val_delta
        and isinstance(val_delta, float)
        and isinstance(holdout_delta, float)
        and val_delta < holdout_delta
    )
    return {
        "artifact_family": "phase2ag_evidence_sufficiency_report",
        "passed": passed,
        "claim_scope": (
            "phase2ag_bounded_runtime_visible_sidecar_dependency_evidence"
            if passed
            else "phase2ag_evidence_incomplete"
        ),
        "checks": checks,
        "metrics": {
            "val_command_slot_accuracy": _metric(postflight, "val_command_slot_accuracy"),
            "holdout_command_slot_accuracy": _metric(
                postflight, "holdout_command_slot_accuracy"
            ),
            "val_source_overlap_accuracy": _metric(
                postflight, "val_source_overlap_accuracy"
            ),
            "holdout_source_overlap_accuracy": _metric(
                postflight, "holdout_source_overlap_accuracy"
            ),
            "val_model_minus_source_overlap_accuracy": val_delta,
            "holdout_model_minus_source_overlap_accuracy": holdout_delta,
            "full_minus_sidecar_erased": full_minus_erased,
            "full_minus_wrong_sidecar": full_minus_wrong,
            "sidecar_erased_accuracy": _metric(controls, "sidecar_erased_accuracy"),
            "wrong_sidecar_accuracy": _metric(controls, "wrong_sidecar_accuracy"),
            "row_count": _metric(controls, "row_count"),
        },
        "caveats": [
            "Val split is near source-overlap ceiling; mechanism delta is assigned to repo-disjoint holdout and sidecar controls."
        ]
        if val_ceiling_caveat
        else [],
        "supported_claims": [
            "A runtime-visible verifiable candidate sidecar can be learned on a non-sealed repo-disjoint holdout.",
            "Erasing or contradicting the sidecar degrades command-slot selection while guarded wrong-sidecar remains non-catastrophic.",
        ]
        if passed
        else [],
        "unsupported_claims": [
            "sealed_cross_model_transfer",
            "model_generated_patch_repair",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": [
            "do_not_package_phase2ag_from_this_report",
            "do_not_run_sealed_phase2ag_from_this_report",
            "do_not_claim_epoch_making_architecture_from_phase2ag",
            "do_not_claim_open_ended_debugging_generalization_from_phase2ag",
        ],
        "next_required_evidence": [
            "independent public repo repair execution with nonzero feasible controls",
            "model-generated patch proposal or bounded edit-head evidence rather than recorded patch replay",
            "multi-seed and cross-model reproduction after non-sealed preregistered gates",
        ],
        "inputs": {
            "postflight_json": str(Path(postflight_json)),
            "sidecar_control_postflight_json": str(Path(sidecar_control_postflight_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AG bounded evidence sufficiency report."
    )
    parser.add_argument("--postflight-json", required=True)
    parser.add_argument("--sidecar-control-postflight-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2ag_evidence_sufficiency_report(
        postflight_json=args.postflight_json,
        sidecar_control_postflight_json=args.sidecar_control_postflight_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
