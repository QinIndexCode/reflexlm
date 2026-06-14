from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _aggregate(summary: dict[str, Any], label: str) -> dict[str, Any]:
    aggregate = summary.get("aggregate_by_label", {}).get(label)
    if not isinstance(aggregate, dict):
        raise KeyError(f"Missing aggregate for label: {label}")
    return aggregate


def _mean_metric(aggregate: dict[str, Any], name: str) -> float:
    return float(aggregate.get(name, 0.0))


def _mean_gate_passed(summary: dict[str, Any], candidate_label: str) -> bool:
    passed_labels = summary.get("passed_mean_labels", [])
    if candidate_label in passed_labels:
        return True
    aggregate = summary.get("aggregate_by_label", {}).get(candidate_label, {})
    return bool(isinstance(aggregate, dict) and aggregate.get("passed_mean_gate", False))


def _all_seed_gates_passed(summary: dict[str, Any], candidate_label: str) -> bool:
    comparisons = [
        row
        for row in summary.get("comparisons_by_seed", [])
        if row.get("label") == candidate_label
    ]
    return bool(comparisons) and all(bool(row.get("passed", False)) for row in comparisons)


def _mean_per_task_completion(aggregate: dict[str, Any], task_name: str) -> float:
    per_task = aggregate.get("mean_per_task_completion", {})
    if not isinstance(per_task, dict):
        return 0.0
    return float(per_task.get(task_name, 0.0))


def _non_regression(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    metric: str,
    tolerance: float,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    candidate_value = _mean_metric(candidate, metric)
    baseline_value = _mean_metric(baseline, metric)
    delta = candidate_value - baseline_value
    passed = delta + tolerance >= 0.0 if higher_is_better else delta <= tolerance
    return {
        "passed": passed,
        "metric": metric,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": delta,
        "tolerance": tolerance,
    }


def assess_promotion_readiness(
    *,
    all_task_summary: dict[str, Any],
    reflex_layer_summary: dict[str, Any],
    debug_cortex_summary: dict[str, Any],
    candidate_label: str,
    baseline_label: str,
    phase2_pause_lock: str | Path,
    debug_nonregression_tolerance: float = 1.0e-6,
    require_all_seed_gates: bool = False,
    min_reflex_layer_completion: float = 0.0,
    min_common_recovery_completion: float = 0.0,
) -> dict[str, Any]:
    all_candidate = _aggregate(all_task_summary, candidate_label)
    all_baseline = _aggregate(all_task_summary, baseline_label)
    reflex_candidate = _aggregate(reflex_layer_summary, candidate_label)
    reflex_baseline = _aggregate(reflex_layer_summary, baseline_label)
    debug_candidate = _aggregate(debug_cortex_summary, candidate_label)
    debug_baseline = _aggregate(debug_cortex_summary, baseline_label)

    debug_completion = _non_regression(
        candidate=debug_candidate,
        baseline=debug_baseline,
        metric="mean_total_completion",
        tolerance=debug_nonregression_tolerance,
    )
    debug_latency = _non_regression(
        candidate=debug_candidate,
        baseline=debug_baseline,
        metric="mean_reaction_latency_ms",
        tolerance=debug_nonregression_tolerance,
        higher_is_better=False,
    )
    debug_hallucination = _non_regression(
        candidate=debug_candidate,
        baseline=debug_baseline,
        metric="mean_state_hallucination_rate",
        tolerance=debug_nonregression_tolerance,
        higher_is_better=False,
    )
    debug_stale = _non_regression(
        candidate=debug_candidate,
        baseline=debug_baseline,
        metric="mean_stale_state_action_rate",
        tolerance=debug_nonregression_tolerance,
        higher_is_better=False,
    )

    pause_lock = Path(phase2_pause_lock)
    checks = {
        "all_task_mean_gate_passed": _mean_gate_passed(all_task_summary, candidate_label),
        "reflex_layer_mean_gate_passed": _mean_gate_passed(
            reflex_layer_summary,
            candidate_label,
        ),
        "all_task_all_seed_gates_passed": (
            _all_seed_gates_passed(all_task_summary, candidate_label)
            if require_all_seed_gates
            else True
        ),
        "reflex_layer_completion_floor": (
            _mean_metric(reflex_candidate, "mean_total_completion")
            >= min_reflex_layer_completion
        ),
        "common_recovery_completion_floor": (
            _mean_per_task_completion(
                all_candidate,
                "common_error_recovery_routine",
            )
            >= min_common_recovery_completion
        ),
        "debug_cortex_completion_non_regression": bool(debug_completion["passed"]),
        "debug_cortex_latency_non_regression": bool(debug_latency["passed"]),
        "debug_cortex_hallucination_non_regression": bool(debug_hallucination["passed"]),
        "debug_cortex_stale_state_non_regression": bool(debug_stale["passed"]),
        "phase2_7b_pause_lock_present": pause_lock.exists(),
    }
    ready = all(checks.values())

    return {
        "ready_for_7b_validation": ready,
        "selected_candidate": candidate_label,
        "baseline_label": baseline_label,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "scope_contract": {
            "single_model": True,
            "reflex_layer_claim": "strict gain required",
            "debug_cortex_claim": "non-regression required; gain not claimed yet",
            "phase2_action": "do not remove pause lock or start 7B without explicit user confirmation",
            "require_all_seed_gates": require_all_seed_gates,
        },
        "all_task": {
            "baseline_completion": _mean_metric(all_baseline, "mean_total_completion"),
            "candidate_completion": _mean_metric(all_candidate, "mean_total_completion"),
            "completion_gain": _mean_metric(
                all_candidate,
                "mean_total_completion_gain_vs_baseline",
            ),
            "baseline_hard_completion": _mean_metric(all_baseline, "mean_hard_completion"),
            "candidate_hard_completion": _mean_metric(all_candidate, "mean_hard_completion"),
            "hard_gain": _mean_metric(
                all_candidate,
                "mean_hard_completion_gain_vs_baseline",
            ),
            "latency_delta_ms": _mean_metric(
                all_candidate,
                "mean_latency_delta_ms_vs_baseline",
            ),
            "mean_per_task_completion": all_candidate.get("mean_per_task_completion", {}),
        },
        "reflex_layer": {
            "baseline_completion": _mean_metric(reflex_baseline, "mean_total_completion"),
            "candidate_completion": _mean_metric(reflex_candidate, "mean_total_completion"),
            "completion_gain": _mean_metric(
                reflex_candidate,
                "mean_total_completion_gain_vs_baseline",
            ),
            "baseline_hard_completion": _mean_metric(reflex_baseline, "mean_hard_completion"),
            "candidate_hard_completion": _mean_metric(reflex_candidate, "mean_hard_completion"),
            "hard_gain": _mean_metric(
                reflex_candidate,
                "mean_hard_completion_gain_vs_baseline",
            ),
            "latency_delta_ms": _mean_metric(
                reflex_candidate,
                "mean_latency_delta_ms_vs_baseline",
            ),
            "minimum_completion_floor": min_reflex_layer_completion,
        },
        "common_recovery": {
            "candidate_completion": _mean_per_task_completion(
                all_candidate,
                "common_error_recovery_routine",
            ),
            "minimum_completion_floor": min_common_recovery_completion,
        },
        "debug_cortex": {
            "completion_non_regression": debug_completion,
            "latency_non_regression": debug_latency,
            "hallucination_non_regression": debug_hallucination,
            "stale_state_non_regression": debug_stale,
            "gain_claimed": False,
        },
        "phase2_pause_lock": str(pause_lock.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether the Phase 1B small-model evidence is ready for 7B validation."
    )
    parser.add_argument("--all-summary", required=True)
    parser.add_argument("--reflex-layer-summary", required=True)
    parser.add_argument("--debug-cortex-summary", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-label", default="flat_v3_slot_focus")
    parser.add_argument(
        "--phase2-pause-lock",
        default="artifacts/control/phase2_7b.paused",
    )
    parser.add_argument("--debug-nonregression-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--require-all-seed-gates", action="store_true")
    parser.add_argument("--min-reflex-layer-completion", type=float, default=0.0)
    parser.add_argument("--min-common-recovery-completion", type=float, default=0.0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = assess_promotion_readiness(
        all_task_summary=_load(args.all_summary),
        reflex_layer_summary=_load(args.reflex_layer_summary),
        debug_cortex_summary=_load(args.debug_cortex_summary),
        candidate_label=args.candidate_label,
        baseline_label=args.baseline_label,
        phase2_pause_lock=args.phase2_pause_lock,
        debug_nonregression_tolerance=args.debug_nonregression_tolerance,
        require_all_seed_gates=args.require_all_seed_gates,
        min_reflex_layer_completion=args.min_reflex_layer_completion,
        min_common_recovery_completion=args.min_common_recovery_completion,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["ready_for_7b_validation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
