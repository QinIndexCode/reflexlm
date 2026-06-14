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


def _accuracy(report: dict[str, Any]) -> float | None:
    return _number(_dict(report.get("eval_metrics")).get("command_slot_accuracy"))


def _count(report: dict[str, Any]) -> float | None:
    return _number(_dict(report.get("eval_metrics")).get("command_slot_count"))


def _source_accuracy(report: dict[str, Any]) -> float | None:
    baselines = _dict(report.get("source_overlap_command_slot_baseline"))
    if len(baselines) == 1:
        return _number(_dict(next(iter(baselines.values()))).get("accuracy"))
    return None


def build_phase2am_natural_sidecar_postflight(
    *,
    controls_manifest_json: str | Path,
    original_eval_json: str | Path,
    erased_eval_json: str | Path,
    wrong_eval_json: str | Path,
    min_full_accuracy: float = 0.85,
    min_full_minus_source: float = 0.15,
    min_full_minus_erased: float = 0.25,
    min_full_minus_wrong: float = 0.25,
) -> dict[str, Any]:
    manifest = _read_json(controls_manifest_json)
    original = _read_json(original_eval_json)
    erased = _read_json(erased_eval_json)
    wrong = _read_json(wrong_eval_json)

    full_acc = _accuracy(original)
    erased_acc = _accuracy(erased)
    wrong_acc = _accuracy(wrong)
    full_count = _count(original)
    erased_count = _count(erased)
    wrong_count = _count(wrong)
    source_acc = _source_accuracy(original)
    erased_source = _source_accuracy(erased)
    wrong_source = _source_accuracy(wrong)
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

    checks = {
        "controls_manifest_passed": manifest.get("passed") is True,
        "source_data_unchanged": manifest.get("source_data_unchanged") is True,
        "sealed_v3_absent": manifest.get("sealed_v3_used") is False,
        "row_counts_match_manifest": isinstance(full_count, float)
        and full_count == erased_count == wrong_count == manifest.get("row_count_selected"),
        "full_accuracy_min": isinstance(full_acc, float) and full_acc >= min_full_accuracy,
        "source_overlap_nonzero_not_ceiling": _dict(manifest.get("checks")).get(
            "source_overlap_nonzero"
        )
        is True
        and _dict(manifest.get("checks")).get("source_overlap_not_ceiling") is True,
        "full_beats_source_overlap": isinstance(full_minus_source, float)
        and full_minus_source >= min_full_minus_source,
        "erased_control_degrades": isinstance(full_minus_erased, float)
        and full_minus_erased >= min_full_minus_erased,
        "wrong_control_degrades": isinstance(full_minus_wrong, float)
        and full_minus_wrong >= min_full_minus_wrong,
        "source_overlap_stable_across_controls": isinstance(source_acc, float)
        and source_acc == erased_source == wrong_source,
        "controls_below_source_overlap": isinstance(erased_acc, float)
        and isinstance(wrong_acc, float)
        and isinstance(source_acc, float)
        and erased_acc < source_acc
        and wrong_acc < source_acc,
        "evaluated_on_cuda": original.get("device") == erased.get("device") == wrong.get("device") == "cuda:0",
        "pairwise_disabled": original.get("use_pairwise_command_reranker") is False
        and erased.get("use_pairwise_command_reranker") is False
        and wrong.get("use_pairwise_command_reranker") is False,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2am_natural_sidecar_postflight",
        "passed": passed,
        "natural_repo_disjoint_sidecar_dependency_evidence": passed,
        "claim_bearing_mechanism_evidence": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "metrics": {
            "full_accuracy": full_acc,
            "source_overlap_accuracy": source_acc,
            "sidecar_erased_accuracy": erased_acc,
            "wrong_sidecar_accuracy": wrong_acc,
            "full_minus_source_overlap": full_minus_source,
            "full_minus_sidecar_erased": full_minus_erased,
            "full_minus_wrong_sidecar": full_minus_wrong,
            "row_count": full_count,
            "repo_count": manifest.get("repo_count"),
        },
        "thresholds": {
            "min_full_accuracy": min_full_accuracy,
            "min_full_minus_source": min_full_minus_source,
            "min_full_minus_erased": min_full_minus_erased,
            "min_full_minus_wrong": min_full_minus_wrong,
        },
        "blocked_actions": [
            "do_not_package_phase2am_from_single_adapter_control",
            "do_not_run_sealed_phase2am_from_single_adapter_control",
            "do_not_claim_open_ended_debugging_generalization",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
        "allowed_next_action": (
            "replicate_phase2am_across_seeds_and_models"
            if passed
            else "freeze_phase2am_control_failure_and_fix_nonsealed_design"
        ),
        "interpretation": (
            "On a natural, non-sealed, repo-disjoint sidecar-present holdout subset, the full "
            "adapter beats a non-ceiling source-overlap baseline and drops sharply under erased "
            "or wrong command-identity sidecar controls."
        ),
        "claim_boundary": (
            "This is natural held-out sidecar-dependency evidence for command-slot selection. "
            "It remains single-adapter evidence and does not establish sealed transfer, "
            "production autonomy, open-ended debugging, or an epoch-making architecture."
        ),
        "inputs": {
            "controls_manifest_json": str(Path(controls_manifest_json)),
            "original_eval_json": str(Path(original_eval_json)),
            "erased_eval_json": str(Path(erased_eval_json)),
            "wrong_eval_json": str(Path(wrong_eval_json)),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AM natural sidecar controls.")
    parser.add_argument("--controls-manifest-json", required=True)
    parser.add_argument("--original-eval-json", required=True)
    parser.add_argument("--erased-eval-json", required=True)
    parser.add_argument("--wrong-eval-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--min-full-accuracy", type=float, default=0.85)
    parser.add_argument("--min-full-minus-source", type=float, default=0.15)
    parser.add_argument("--min-full-minus-erased", type=float, default=0.25)
    parser.add_argument("--min-full-minus-wrong", type=float, default=0.25)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2am_natural_sidecar_postflight(
        controls_manifest_json=args.controls_manifest_json,
        original_eval_json=args.original_eval_json,
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
