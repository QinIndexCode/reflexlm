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
    for payload in _dict(report.get("source_overlap_command_slot_baseline")).values():
        value = _dict(payload).get("accuracy")
        if isinstance(value, int | float):
            return float(value)
    return None


def audit_phase2ao_order_permutation_postflight(
    *,
    original_permutation_manifest_json: str | Path,
    erased_permutation_manifest_json: str | Path,
    residual_baseline_json: str | Path,
    original_permuted_eval_json: str | Path,
    erased_permuted_eval_json: str | Path,
    output_json: str | Path | None = None,
    min_original_accuracy: float = 0.85,
    min_original_minus_erased: float = 0.50,
    min_original_minus_source: float = 0.50,
) -> dict[str, Any]:
    original_manifest = _read_json(original_permutation_manifest_json)
    erased_manifest = _read_json(erased_permutation_manifest_json)
    residual = _read_json(residual_baseline_json)
    original_eval = _read_json(original_permuted_eval_json)
    erased_eval = _read_json(erased_permuted_eval_json)
    original_acc = _accuracy(original_eval)
    erased_acc = _accuracy(erased_eval)
    source_acc = _source_accuracy(original_eval)
    original_minus_erased = (
        original_acc - erased_acc if isinstance(original_acc, float) and isinstance(erased_acc, float) else None
    )
    original_minus_source = (
        original_acc - source_acc if isinstance(original_acc, float) and isinstance(source_acc, float) else None
    )
    residual_explained_by_prior = residual.get("passed") is False and _dict(
        residual.get("checks")
    ).get("model_exceeds_max_nonleaky_baseline") is False
    checks = {
        "original_permutation_manifest_passed": original_manifest.get("passed") is True,
        "erased_permutation_manifest_passed": erased_manifest.get("passed") is True,
        "residual_explained_by_nonleaky_prior_baseline": residual_explained_by_prior,
        "original_permuted_accuracy_min": isinstance(original_acc, float)
        and original_acc >= min_original_accuracy,
        "original_permuted_beats_source": isinstance(original_minus_source, float)
        and original_minus_source >= min_original_minus_source,
        "original_permuted_beats_erased_permuted": isinstance(original_minus_erased, float)
        and original_minus_erased >= min_original_minus_erased,
        "erased_permuted_not_above_source": isinstance(erased_acc, float)
        and isinstance(source_acc, float)
        and erased_acc <= source_acc,
        "evaluated_on_cuda": original_eval.get("device") == erased_eval.get("device") == "cuda:0",
        "pairwise_disabled": original_eval.get("use_pairwise_command_reranker") is False
        and erased_eval.get("use_pairwise_command_reranker") is False,
    }
    sidecar_order_robustness_supported = all(
        checks[name]
        for name in (
            "original_permutation_manifest_passed",
            "erased_permutation_manifest_passed",
            "original_permuted_accuracy_min",
            "original_permuted_beats_source",
            "original_permuted_beats_erased_permuted",
            "evaluated_on_cuda",
            "pairwise_disabled",
        )
    )
    strict_erased_residual_control_passed = checks["erased_permuted_not_above_source"]
    passed = sidecar_order_robustness_supported and strict_erased_residual_control_passed
    if strict_erased_residual_control_passed:
        interpretation = (
            "Full sidecar performance remains high after deterministic candidate-order permutation, "
            "while erased-sidecar performance falls below source-overlap. Combined with the residual "
            "baseline audit, this indicates the earlier erased residual was explainable by candidate "
            "count/position priors, not hidden sidecar-free mechanism ability."
        )
    else:
        interpretation = (
            "Full sidecar performance remains high after deterministic candidate-order permutation "
            "and strongly exceeds erased-sidecar performance, supporting order-robust sidecar control. "
            "However, erased-sidecar performance remains above source-overlap on this run, so the strict "
            "erased residual control is not clean and must be reported as residual position-prior risk."
        )
    report = {
        "artifact_family": "phase2ao_order_permutation_postflight",
        "passed": passed,
        "sidecar_order_robustness_supported": sidecar_order_robustness_supported,
        "strict_erased_residual_control_passed": strict_erased_residual_control_passed,
        "erased_residual_explained_by_position_prior": residual_explained_by_prior,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "metrics": {
            "original_permuted_accuracy": original_acc,
            "erased_permuted_accuracy": erased_acc,
            "source_overlap_accuracy": source_acc,
            "original_minus_erased_permuted": original_minus_erased,
            "original_minus_source": original_minus_source,
            "changed_gold_slot_rows": original_manifest.get("changed_gold_slot_rows"),
            "non_identity_permutation_rows": original_manifest.get(
                "non_identity_permutation_rows"
            ),
        },
        "interpretation": interpretation,
        "claim_boundary": (
            "Phase2AO strengthens bounded command-slot sidecar-control evidence. It still does not "
            "establish sealed transfer, production autonomy, open-ended debugging, or an epoch-making "
            "architecture."
        ),
        "blocked_actions": [
            "do_not_claim_sealed_transfer_from_phase2ao",
            "do_not_claim_open_ended_debugging_generalization",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
        "inputs": {
            "original_permutation_manifest_json": str(Path(original_permutation_manifest_json)),
            "erased_permutation_manifest_json": str(Path(erased_permutation_manifest_json)),
            "residual_baseline_json": str(Path(residual_baseline_json)),
            "original_permuted_eval_json": str(Path(original_permuted_eval_json)),
            "erased_permuted_eval_json": str(Path(erased_permuted_eval_json)),
        },
    }
    if output_json:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AO order permutation postflight.")
    parser.add_argument("--original-permutation-manifest-json", required=True)
    parser.add_argument("--erased-permutation-manifest-json", required=True)
    parser.add_argument("--residual-baseline-json", required=True)
    parser.add_argument("--original-permuted-eval-json", required=True)
    parser.add_argument("--erased-permuted-eval-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ao_order_permutation_postflight(
        original_permutation_manifest_json=args.original_permutation_manifest_json,
        erased_permutation_manifest_json=args.erased_permutation_manifest_json,
        residual_baseline_json=args.residual_baseline_json,
        original_permuted_eval_json=args.original_permuted_eval_json,
        erased_permuted_eval_json=args.erased_permuted_eval_json,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
