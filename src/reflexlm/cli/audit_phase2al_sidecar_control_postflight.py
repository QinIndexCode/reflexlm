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
        return _number(_dict(next(iter(baselines.values()))).get("accuracy"))
    return None


def build_phase2al_sidecar_control_postflight(
    *,
    phase2al_postflight_json: str | Path,
    controls_manifest_json: str | Path,
    holdout_eval_json: str | Path,
    erased_eval_json: str | Path,
    wrong_eval_json: str | Path,
    min_full_accuracy: float = 0.85,
    min_full_minus_source: float = 0.15,
    min_full_minus_erased: float = 0.25,
    min_full_minus_wrong: float = 0.25,
) -> dict[str, Any]:
    postflight = _read_json(phase2al_postflight_json)
    controls = _read_json(controls_manifest_json)
    full = _read_json(holdout_eval_json)
    erased = _read_json(erased_eval_json)
    wrong = _read_json(wrong_eval_json)

    split = str(postflight.get("split") or "holdout")
    full_acc = _accuracy(full)
    erased_acc = _accuracy(erased)
    wrong_acc = _accuracy(wrong)
    full_count = _count(full)
    erased_count = _count(erased)
    wrong_count = _count(wrong)
    source_acc = _source_accuracy(full, split)
    erased_source = _source_accuracy(erased, f"{split}_sidecar_erased")
    wrong_source = _source_accuracy(wrong, f"{split}_wrong_sidecar")
    full_minus_source = (
        full_acc - source_acc
        if isinstance(full_acc, float) and isinstance(source_acc, float)
        else None
    )
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

    # Phase2AL postflight intentionally stays bounded and does not need Phase2AG's
    # ready_for_package fields. Treat missing package/sealed readiness as false.
    no_package_or_sealed = not bool(postflight.get("ready_for_package")) and not bool(
        postflight.get("ready_for_sealed_eval")
    )
    checks = {
        "phase2al_postflight_passed": postflight.get("passed") is True,
        "phase2al_controlled_pressure_only": _dict(postflight.get("checks")).get(
            "controlled_pressure_only"
        )
        is True,
        "controls_manifest_passed": controls.get("passed") is True,
        "row_counts_match": isinstance(full_count, float)
        and full_count == erased_count == wrong_count,
        "full_accuracy_min": isinstance(full_acc, float) and full_acc >= min_full_accuracy,
        "full_beats_source_overlap": isinstance(full_minus_source, float)
        and full_minus_source >= min_full_minus_source,
        "erased_control_degrades": isinstance(full_minus_erased, float)
        and full_minus_erased >= min_full_minus_erased,
        "wrong_control_degrades": isinstance(full_minus_wrong, float)
        and full_minus_wrong >= min_full_minus_wrong,
        "source_overlap_stable_across_controls": isinstance(source_acc, float)
        and source_acc == erased_source == wrong_source,
        "controls_below_full": isinstance(erased_acc, float)
        and isinstance(wrong_acc, float)
        and isinstance(full_acc, float)
        and erased_acc < full_acc
        and wrong_acc < full_acc,
        "no_package_or_sealed_claim": no_package_or_sealed,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2al_sidecar_control_postflight",
        "passed": passed,
        "sidecar_dependency_evidence": passed,
        "claim_bearing_mechanism_evidence": False,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "metrics": {
            "full_holdout_accuracy": full_acc,
            "sidecar_erased_accuracy": erased_acc,
            "wrong_sidecar_accuracy": wrong_acc,
            "source_overlap_accuracy": source_acc,
            "full_minus_source_overlap": full_minus_source,
            "full_minus_sidecar_erased": full_minus_erased,
            "full_minus_wrong_sidecar": full_minus_wrong,
            "row_count": full_count,
        },
        "thresholds": {
            "min_full_accuracy": min_full_accuracy,
            "min_full_minus_source": min_full_minus_source,
            "min_full_minus_erased": min_full_minus_erased,
            "min_full_minus_wrong": min_full_minus_wrong,
        },
        "blocked_actions": [
            "do_not_package_phase2al_from_controlled_smoke",
            "do_not_run_sealed_phase2al_from_controlled_smoke",
            "do_not_claim_epoch_making_architecture",
            "do_not_claim_open_ended_debugging_generalization",
        ],
        "allowed_next_action": (
            "move_to_more_natural_phase2am_sidecar_controls"
            if passed
            else "freeze_phase2al_control_failure_and_fix_nonsealed_design"
        ),
        "interpretation": (
            "Phase2AL full evaluation beats a non-ceiling source-overlap baseline and degrades "
            "under erased or wrong command-identity sidecar controls. This supports dependence on "
            "the structural sidecar in a controlled non-sealed setting only."
        ),
        "claim_boundary": (
            "Phase2AL is controlled pressure evidence. It does not establish natural trace "
            "distribution, sealed transfer, production autonomy, open-ended debugging "
            "generalization, or an epoch-making architecture."
        ),
        "inputs": {
            "phase2al_postflight_json": str(Path(phase2al_postflight_json)),
            "controls_manifest_json": str(Path(controls_manifest_json)),
            "holdout_eval_json": str(Path(holdout_eval_json)),
            "erased_eval_json": str(Path(erased_eval_json)),
            "wrong_eval_json": str(Path(wrong_eval_json)),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AL sidecar control postflight.")
    parser.add_argument("--phase2al-postflight-json", required=True)
    parser.add_argument("--controls-manifest-json", required=True)
    parser.add_argument("--holdout-eval-json", required=True)
    parser.add_argument("--erased-eval-json", required=True)
    parser.add_argument("--wrong-eval-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--min-full-accuracy", type=float, default=0.85)
    parser.add_argument("--min-full-minus-source", type=float, default=0.15)
    parser.add_argument("--min-full-minus-erased", type=float, default=0.25)
    parser.add_argument("--min-full-minus-wrong", type=float, default=0.25)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_phase2al_sidecar_control_postflight(
        phase2al_postflight_json=args.phase2al_postflight_json,
        controls_manifest_json=args.controls_manifest_json,
        holdout_eval_json=args.holdout_eval_json,
        erased_eval_json=args.erased_eval_json,
        wrong_eval_json=args.wrong_eval_json,
        min_full_accuracy=args.min_full_accuracy,
        min_full_minus_source=args.min_full_minus_source,
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
