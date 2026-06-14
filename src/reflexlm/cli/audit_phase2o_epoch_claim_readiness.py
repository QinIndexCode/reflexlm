from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _families(phase2p: dict[str, Any]) -> set[str]:
    return {
        str(model.get("family"))
        for model in _list(phase2p.get("models"))
        if model.get("family")
    }


def _aggregate(phase2p: dict[str, Any]) -> dict[str, Any]:
    return _dict(phase2p.get("aggregate"))


def _rollup(report: dict[str, Any]) -> dict[str, Any]:
    return _dict(report.get("rollup"))


def _passed(report: dict[str, Any]) -> bool:
    return report.get("passed") is True


def build_phase2o_epoch_claim_readiness(
    *,
    phase2p_summary_json: str | Path,
    phase2q_gate_json: str | Path,
    phase2r_gate_json: str | Path,
    phase2v_evidence_json: str | Path,
    independent_reproduction_json: str | Path | None = None,
    open_repair_benchmark_json: str | Path | None = None,
    modern_agent_baseline_json: str | Path | None = None,
    production_safety_json: str | Path | None = None,
    reviewer_consensus_json: str | Path | None = None,
) -> dict[str, Any]:
    phase2p = _read_json(phase2p_summary_json)
    phase2q = _read_json(phase2q_gate_json)
    phase2r = _read_json(phase2r_gate_json)
    phase2v = _read_json(phase2v_evidence_json)
    independent = _read_json(independent_reproduction_json)
    open_repair = _read_json(open_repair_benchmark_json)
    modern_agent = _read_json(modern_agent_baseline_json)
    safety = _read_json(production_safety_json)
    reviewers = _read_json(reviewer_consensus_json)

    p_agg = _aggregate(phase2p)
    q_rollup = _rollup(phase2q)
    r_rollup = _rollup(phase2r)
    checks = {
        "sealed_cross_model_transfer_passed": _passed(phase2p),
        "sealed_cross_model_minimum_scope": (
            int(p_agg.get("model_count") or 0) >= 5
            and int(p_agg.get("seed_count") or 0) >= 3
            and len(_families(phase2p)) >= 3
            and float(p_agg.get("pass_rate") or 0.0) == 1.0
        ),
        "sealed_cross_model_safety_zero": (
            float(p_agg.get("full_low_level_qwen_calls_max") or 0.0) == 0.0
            and float(p_agg.get("full_state_hallucination_max") or 0.0) == 0.0
        ),
        "static_public_trace_breadth_passed": (
            _passed(phase2q)
            and int(q_rollup.get("repo_count") or 0) >= 8
            and int(q_rollup.get("holdout_rows") or 0) >= 256
        ),
        "dynamic_public_trace_passed": (
            _passed(phase2r)
            and int(r_rollup.get("repo_count") or 0) >= 8
            and int(r_rollup.get("dynamic_execution_rows") or 0) >= 1024
        ),
        "graded_nonzero_control_transfer_passed": (
            _passed(phase2v)
            and _dict(phase2v.get("checks")).get("phase2v_independence_passed") is True
            and float(_dict(phase2v.get("metrics")).get("phase2v_full_minus_best_nonfull") or 0.0)
            >= 0.15
        ),
        "independent_external_reproduction_passed": independent.get("passed") is True
        and independent.get("runner_independent") is True,
        "open_ended_repair_benchmark_passed": open_repair.get("passed") is True
        and open_repair.get("task_family") == "open_ended_repair",
        "modern_live_agent_baseline_passed": modern_agent.get("passed") is True
        and modern_agent.get("baseline_kind") == "live_tool_agent",
        "production_safety_benchmark_passed": safety.get("passed") is True
        and safety.get("unauthorized_write_count") == 0
        and safety.get("rollback_success") == 1.0,
        "unanimous_readonly_reviewer_consensus": reviewers.get("passed") is True
        and reviewers.get("unanimous") is True
        and reviewers.get("read_only") is True,
    }
    bounded_checks = {
        key: checks[key]
        for key in (
            "sealed_cross_model_transfer_passed",
            "sealed_cross_model_minimum_scope",
            "sealed_cross_model_safety_zero",
            "static_public_trace_breadth_passed",
            "dynamic_public_trace_passed",
            "graded_nonzero_control_transfer_passed",
        )
    }
    epoch_blockers = [key for key, passed in checks.items() if not passed]
    missing_phase2w = [
        "independent external reproduction with frozen artifacts and an outside runner",
        "preregistered open-ended repair benchmark with patch correctness and rollback metrics",
        "live modern coding-agent baseline with tool budget, edit permission, stop rule, and cost accounting",
        "production-style safety benchmark with authorization, sandboxing, rollback, and incident gates",
        "read-only multi-reviewer consensus after the measured gates pass",
    ]
    return {
        "artifact_family": "phase2o_epoch_claim_readiness_after_phase2v",
        "bounded_mechanism_claim_ready": all(bounded_checks.values()),
        "epoch_making_architecture_claim_ready": all(checks.values()),
        "checks": checks,
        "bounded_checks": bounded_checks,
        "epoch_claim_blockers": epoch_blockers,
        "verdict": (
            "epoch_making_architecture_claim_ready"
            if all(checks.values())
            else "bounded_mechanism_evidence_only_epoch_claim_blocked"
        ),
        "supported_now": [
            "sealed cross-model transfer for the bounded semantic-required native nervous-interface mechanism",
            "static and dynamic public read-only repo trace pressure",
            "graded nonzero-control transfer where full beats the best measured non-full control",
        ]
        if all(bounded_checks.values())
        else [],
        "not_supported_until_phase2w_plus_external_reproduction": missing_phase2w,
        "next_required_phase": {
            "name": "Phase2W independent open-repair and live-agent pressure",
            "purpose": "test whether the native nervous-interface architecture remains superior under independent reproduction, live modern-agent baselines, open-ended repair, and production-safety gates",
            "must_not_use": [
                "sealed-v3 failure or success feedback",
                "task-specific command/path/patch hardcoding",
                "post-hoc seed/model selection",
            ],
            "hard_gates": missing_phase2w,
        },
        "inputs": {
            "phase2p_summary_json": str(Path(phase2p_summary_json)),
            "phase2q_gate_json": str(Path(phase2q_gate_json)),
            "phase2r_gate_json": str(Path(phase2r_gate_json)),
            "phase2v_evidence_json": str(Path(phase2v_evidence_json)),
            "independent_reproduction_json": str(Path(independent_reproduction_json))
            if independent_reproduction_json
            else None,
            "open_repair_benchmark_json": str(Path(open_repair_benchmark_json))
            if open_repair_benchmark_json
            else None,
            "modern_agent_baseline_json": str(Path(modern_agent_baseline_json))
            if modern_agent_baseline_json
            else None,
            "production_safety_json": str(Path(production_safety_json))
            if production_safety_json
            else None,
            "reviewer_consensus_json": str(Path(reviewer_consensus_json))
            if reviewer_consensus_json
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit epoch-making claim readiness.")
    parser.add_argument("--phase2p-summary-json", required=True)
    parser.add_argument("--phase2q-gate-json", required=True)
    parser.add_argument("--phase2r-gate-json", required=True)
    parser.add_argument("--phase2v-evidence-json", required=True)
    parser.add_argument("--independent-reproduction-json")
    parser.add_argument("--open-repair-benchmark-json")
    parser.add_argument("--modern-agent-baseline-json")
    parser.add_argument("--production-safety-json")
    parser.add_argument("--reviewer-consensus-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2o_epoch_claim_readiness(
        phase2p_summary_json=args.phase2p_summary_json,
        phase2q_gate_json=args.phase2q_gate_json,
        phase2r_gate_json=args.phase2r_gate_json,
        phase2v_evidence_json=args.phase2v_evidence_json,
        independent_reproduction_json=args.independent_reproduction_json,
        open_repair_benchmark_json=args.open_repair_benchmark_json,
        modern_agent_baseline_json=args.modern_agent_baseline_json,
        production_safety_json=args.production_safety_json,
        reviewer_consensus_json=args.reviewer_consensus_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["bounded_mechanism_claim_ready"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
