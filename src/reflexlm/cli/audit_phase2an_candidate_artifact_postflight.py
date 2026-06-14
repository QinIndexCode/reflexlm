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


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _accuracy(report: dict[str, Any]) -> float | None:
    value = _dict(report.get("eval_metrics")).get("command_slot_accuracy")
    return float(value) if isinstance(value, int | float) else None


def _source_accuracy(report: dict[str, Any]) -> float | None:
    baselines = _dict(report.get("source_overlap_command_slot_baseline"))
    for payload in baselines.values():
        accuracy = _dict(payload).get("accuracy")
        if isinstance(accuracy, int | float):
            return float(accuracy)
    return None


def _device(report: dict[str, Any]) -> str | None:
    value = report.get("device")
    return str(value) if isinstance(value, str) else None


def audit_phase2an_candidate_artifact_postflight(
    *,
    controls_manifest_json: str | Path,
    neutral_original_eval_json: str | Path,
    neutral_erased_eval_json: str | Path,
    neutral_wrong_eval_json: str | Path,
    output_json: str | Path | None = None,
    min_original_accuracy: float = 0.85,
    min_original_minus_source: float = 0.50,
    min_original_minus_erased: float = 0.25,
    min_original_minus_wrong: float = 0.50,
) -> dict[str, Any]:
    manifest = _read_json(controls_manifest_json)
    original = _read_json(neutral_original_eval_json)
    erased = _read_json(neutral_erased_eval_json)
    wrong = _read_json(neutral_wrong_eval_json)
    original_acc = _accuracy(original)
    erased_acc = _accuracy(erased)
    wrong_acc = _accuracy(wrong)
    source_acc = _source_accuracy(original)
    original_minus_source = (
        original_acc - source_acc if isinstance(original_acc, float) and isinstance(source_acc, float) else None
    )
    original_minus_erased = (
        original_acc - erased_acc if isinstance(original_acc, float) and isinstance(erased_acc, float) else None
    )
    original_minus_wrong = (
        original_acc - wrong_acc if isinstance(original_acc, float) and isinstance(wrong_acc, float) else None
    )
    checks = {
        "controls_manifest_passed": manifest.get("passed") is True,
        "candidate_artifacts_removed": _dict(manifest.get("checks")).get("candidate_artifacts_removed")
        is True,
        "neutral_original_accuracy_min": isinstance(original_acc, float)
        and original_acc >= min_original_accuracy,
        "neutral_source_overlap_nonzero_not_ceiling": isinstance(source_acc, float)
        and 0.0 < source_acc < 0.75,
        "neutral_original_beats_source": isinstance(original_minus_source, float)
        and original_minus_source >= min_original_minus_source,
        "neutral_erased_degrades": isinstance(original_minus_erased, float)
        and original_minus_erased >= min_original_minus_erased,
        "neutral_wrong_degrades_strongly": isinstance(original_minus_wrong, float)
        and original_minus_wrong >= min_original_minus_wrong,
        "wrong_below_source_overlap": isinstance(wrong_acc, float)
        and isinstance(source_acc, float)
        and wrong_acc <= source_acc,
        "evaluated_on_cuda": _device(original) == _device(erased) == _device(wrong) == "cuda:0",
        "pairwise_disabled": original.get("use_pairwise_command_reranker") is False
        and erased.get("use_pairwise_command_reranker") is False
        and wrong.get("use_pairwise_command_reranker") is False,
    }
    erased_residual_above_source = (
        isinstance(erased_acc, float) and isinstance(source_acc, float) and erased_acc > source_acc
    )
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2an_candidate_artifact_postflight",
        "passed": passed,
        "diagnostic_control_passed": passed,
        "pure_sidecar_dependency_supported": passed and not erased_residual_above_source,
        "sidecar_control_effect_supported": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "metrics": {
            "neutral_original_accuracy": original_acc,
            "neutral_source_overlap_accuracy": source_acc,
            "neutral_sidecar_erased_accuracy": erased_acc,
            "neutral_wrong_sidecar_accuracy": wrong_acc,
            "neutral_original_minus_source": original_minus_source,
            "neutral_original_minus_erased": original_minus_erased,
            "neutral_original_minus_wrong": original_minus_wrong,
            "erased_residual_above_source": erased_residual_above_source,
        },
        "interpretation": (
            "Neutralizing candidate text preserves perfect full-sidecar selection and makes wrong "
            "sidecar collapse, so command-identity sidecar has strong causal control. However, "
            "erased-sidecar residual performance above source-overlap means slot-prior or structural "
            "residual signal remains; do not claim pure sidecar-only causality."
        ),
        "claim_boundary": (
            "Phase2AN is a diagnostic control over Phase2AM. It supports bounded sidecar-control "
            "evidence, not sealed transfer, open-ended debugging, production autonomy, or an "
            "epoch-making architecture."
        ),
        "blocked_actions": [
            "do_not_claim_pure_sidecar_dependency_if_erased_residual_above_source",
            "do_not_package_from_phase2an_diagnostic_control",
            "do_not_run_sealed_eval_from_phase2an_diagnostic_control",
            "do_not_claim_open_ended_debugging_generalization",
        ],
        "inputs": {
            "controls_manifest_json": str(Path(controls_manifest_json)),
            "neutral_original_eval_json": str(Path(neutral_original_eval_json)),
            "neutral_erased_eval_json": str(Path(neutral_erased_eval_json)),
            "neutral_wrong_eval_json": str(Path(neutral_wrong_eval_json)),
        },
    }
    if output_json:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AN candidate artifact controls.")
    parser.add_argument("--controls-manifest-json", required=True)
    parser.add_argument("--neutral-original-eval-json", required=True)
    parser.add_argument("--neutral-erased-eval-json", required=True)
    parser.add_argument("--neutral-wrong-eval-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2an_candidate_artifact_postflight(
        controls_manifest_json=args.controls_manifest_json,
        neutral_original_eval_json=args.neutral_original_eval_json,
        neutral_erased_eval_json=args.neutral_erased_eval_json,
        neutral_wrong_eval_json=args.neutral_wrong_eval_json,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
