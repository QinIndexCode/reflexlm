from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _accuracy(eval_report: dict[str, Any]) -> float | None:
    return _number(_dict(eval_report.get("eval_metrics")).get("command_slot_accuracy"))


def _count(eval_report: dict[str, Any]) -> float | None:
    return _number(_dict(eval_report.get("eval_metrics")).get("command_slot_count"))


def _source_accuracy(eval_report: dict[str, Any], split: str) -> float | None:
    baselines = _dict(eval_report.get("source_overlap_command_slot_baseline"))
    direct = _number(_dict(baselines.get(split)).get("accuracy"))
    if direct is not None:
        return direct
    if len(baselines) == 1:
        only = next(iter(baselines.values()))
        return _number(_dict(only).get("accuracy"))
    return None


def build_phase2ag_sidecar_control_postflight(
    *,
    smoke_postflight_json: str | Path,
    controls_manifest_json: str | Path,
    holdout_eval_json: str | Path,
    erased_eval_json: str | Path,
    wrong_eval_json: str | Path,
    min_full_accuracy: float = 0.85,
    min_full_minus_erased: float = 0.25,
    min_full_minus_wrong: float = 0.25,
) -> dict[str, Any]:
    smoke = _read_json(smoke_postflight_json)
    controls = _read_json(controls_manifest_json)
    full = _read_json(holdout_eval_json)
    erased = _read_json(erased_eval_json)
    wrong = _read_json(wrong_eval_json)

    full_acc = _accuracy(full)
    erased_acc = _accuracy(erased)
    wrong_acc = _accuracy(wrong)
    full_count = _count(full)
    erased_count = _count(erased)
    wrong_count = _count(wrong)
    source_acc = _source_accuracy(full, "holdout")
    erased_source = _source_accuracy(erased, "holdout_sidecar_erased")
    wrong_source = _source_accuracy(wrong, "holdout_wrong_sidecar")
    full_minus_erased = (
        full_acc - erased_acc
        if isinstance(full_acc, float) and isinstance(erased_acc, float)
        else None
    )
    full_minus_wrong = (
        full_acc - wrong_acc
        if isinstance(full_acc, float) and isinstance(wrong_acc, float)
        else None
    )

    checks = {
        "smoke_postflight_passed": smoke.get("passed") is True,
        "controls_manifest_passed": controls.get("passed") is True,
        "row_counts_match": isinstance(full_count, float)
        and full_count == erased_count == wrong_count,
        "full_accuracy_min": isinstance(full_acc, float) and full_acc >= min_full_accuracy,
        "erased_control_degrades": isinstance(full_minus_erased, float)
        and full_minus_erased >= min_full_minus_erased,
        "wrong_control_degrades": isinstance(full_minus_wrong, float)
        and full_minus_wrong >= min_full_minus_wrong,
        "source_overlap_stable_across_controls": isinstance(source_acc, float)
        and source_acc == erased_source == wrong_source,
        "wrong_control_below_source_overlap": isinstance(wrong_acc, float)
        and isinstance(source_acc, float)
        and wrong_acc < source_acc,
        "erased_control_not_claim_sufficient": isinstance(erased_acc, float)
        and isinstance(full_acc, float)
        and erased_acc < full_acc,
        "no_sealed_or_package_claim": smoke.get("ready_for_package") is False
        and smoke.get("ready_for_sealed_eval") is False,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ag_sidecar_control_postflight",
        "passed": passed,
        "sidecar_dependency_smoke_evidence": passed,
        "claim_bearing_mechanism_evidence": False,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "metrics": {
            "full_holdout_accuracy": full_acc,
            "sidecar_erased_accuracy": erased_acc,
            "wrong_sidecar_accuracy": wrong_acc,
            "source_overlap_accuracy": source_acc,
            "full_minus_sidecar_erased": full_minus_erased,
            "full_minus_wrong_sidecar": full_minus_wrong,
            "row_count": full_count,
        },
        "thresholds": {
            "min_full_accuracy": min_full_accuracy,
            "min_full_minus_erased": min_full_minus_erased,
            "min_full_minus_wrong": min_full_minus_wrong,
        },
        "blocked_actions": [
            "do_not_package_phase2ag_from_smoke_controls",
            "do_not_run_sealed_phase2ag_from_smoke_controls",
            "do_not_claim_epoch_making_architecture",
            "do_not_claim_open_ended_debugging_generalization",
        ],
        "allowed_next_action": (
            "scale_phase2ag_nonsealed_split_with_erased_and_wrong_sidecar_controls"
            if passed
            else "freeze_phase2ag_control_failure_and_fix_nonsealed_design"
        ),
        "interpretation": (
            "The full smoke model succeeds on the original repo-disjoint holdout and degrades under "
            "sidecar-erased and wrong-sidecar controls. This supports sidecar dependency in the smoke "
            "setting. If wrong-sidecar accuracy remains above zero, the runtime verification guard is "
            "mitigating catastrophic misdirection, but the smoke remains insufficient for package, sealed "
            "transfer, or broader architecture claims."
        ),
        "inputs": {
            "smoke_postflight_json": str(Path(smoke_postflight_json)),
            "controls_manifest_json": str(Path(controls_manifest_json)),
            "holdout_eval_json": str(Path(holdout_eval_json)),
            "erased_eval_json": str(Path(erased_eval_json)),
            "wrong_eval_json": str(Path(wrong_eval_json)),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AG sidecar control postflight.")
    parser.add_argument("--smoke-postflight-json", required=True)
    parser.add_argument("--controls-manifest-json", required=True)
    parser.add_argument("--holdout-eval-json", required=True)
    parser.add_argument("--erased-eval-json", required=True)
    parser.add_argument("--wrong-eval-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--min-full-accuracy", type=float, default=0.85)
    parser.add_argument("--min-full-minus-erased", type=float, default=0.25)
    parser.add_argument("--min-full-minus-wrong", type=float, default=0.25)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2ag_sidecar_control_postflight(
        smoke_postflight_json=args.smoke_postflight_json,
        controls_manifest_json=args.controls_manifest_json,
        holdout_eval_json=args.holdout_eval_json,
        erased_eval_json=args.erased_eval_json,
        wrong_eval_json=args.wrong_eval_json,
        min_full_accuracy=args.min_full_accuracy,
        min_full_minus_erased=args.min_full_minus_erased,
        min_full_minus_wrong=args.min_full_minus_wrong,
    )
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
