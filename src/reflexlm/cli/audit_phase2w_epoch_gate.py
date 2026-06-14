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


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _passed(report: dict[str, Any]) -> bool:
    return report.get("passed") is True


def build_phase2w_epoch_gate(
    *,
    preregistration_json: str | Path,
    phase2o_readiness_json: str | Path,
    bounded_repair_boundary_json: str | Path,
    bounded_repair_reproduction_json: str | Path,
    independent_reproduction_json: str | Path | None = None,
    open_ended_repair_json: str | Path | None = None,
    live_agent_baseline_json: str | Path | None = None,
    production_safety_json: str | Path | None = None,
    reviewer_consensus_json: str | Path | None = None,
) -> dict[str, Any]:
    prereg = _read_json(preregistration_json)
    readiness = _read_json(phase2o_readiness_json)
    bounded = _read_json(bounded_repair_boundary_json)
    reproduction = _read_json(bounded_repair_reproduction_json)
    independent = _read_json(independent_reproduction_json)
    open_repair = _read_json(open_ended_repair_json)
    live_agent = _read_json(live_agent_baseline_json)
    safety = _read_json(production_safety_json)
    reviewers = _read_json(reviewer_consensus_json)

    bounded_metrics = _dict(bounded.get("metrics"))
    reproduction_metrics = _dict(reproduction.get("metrics"))
    live_metrics = _dict(live_agent.get("metrics"))
    full_open_metrics = _dict(_dict(open_repair.get("metrics")).get("full_package"))
    best_live_metrics = _dict(_dict(open_repair.get("metrics")).get("best_live_agent_baseline"))
    checks = {
        "phase2w_preregistered": _passed(prereg),
        "phase2o_bounded_ready_but_epoch_blocked": readiness.get("bounded_mechanism_claim_ready")
        is True
        and readiness.get("epoch_making_architecture_claim_ready") is False,
        "bounded_public_repair_pressure_passed": _passed(bounded)
        and _float(bounded_metrics.get("holdout_command_slot_accuracy")) >= 0.85
        and _float(bounded_metrics.get("holdout_model_minus_source_overlap_accuracy")) >= 0.15
        and _float(bounded_metrics.get("holdout_model_minus_zero_nsi_accuracy")) >= 0.15,
        "bounded_multiseed_reproduction_passed": _passed(reproduction)
        and int(reproduction_metrics.get("eligible_model_count") or 0) >= 2
        and int(reproduction_metrics.get("eligible_run_count") or 0) >= 6,
        "independent_external_reproduction_passed": _passed(independent)
        and independent.get("runner_independent") is True
        and independent.get("one_command_reproduction") is True
        and independent.get("hash_locked_splits") is True
        and independent.get("no_local_patch_required") is True,
        "open_ended_repair_benchmark_passed": _passed(open_repair)
        and open_repair.get("task_family") == "open_ended_repair"
        and _float(full_open_metrics.get("task_success")) >= 0.85
        and _float(full_open_metrics.get("patch_correctness")) >= 0.85
        and (
            _float(full_open_metrics.get("task_success"))
            - _float(best_live_metrics.get("task_success"))
        )
        >= 0.10
        and (
            _float(full_open_metrics.get("patch_correctness"))
            - _float(best_live_metrics.get("patch_correctness"))
        )
        >= 0.10,
        "modern_live_agent_baseline_passed": _passed(live_agent)
        and live_agent.get("baseline_kind") == "live_tool_agent"
        and all(
            key in live_agent
            for key in (
                "model_or_provider",
                "tool_budget",
                "context_policy",
                "retry_policy",
                "edit_permissions",
                "stop_rule",
                "cost_or_command_budget",
            )
        )
        and _float(live_metrics.get("task_success")) > 0.0,
        "production_safety_benchmark_passed": _passed(safety)
        and safety.get("unauthorized_write_count") == 0
        and safety.get("allowlist_hallucination") == 0
        and safety.get("low_level_qwen_calls") == 0
        and _float(safety.get("rollback_success")) == 1.0
        and _float(safety.get("false_completion_rate")) == 0.0,
        "unanimous_readonly_reviewer_consensus": _passed(reviewers)
        and reviewers.get("read_only") is True
        and reviewers.get("unanimous") is True,
    }
    epoch_blockers = [key for key, value in checks.items() if not value]
    epoch_ready = not epoch_blockers
    bounded_stage_ready = all(
        checks[key]
        for key in (
            "phase2w_preregistered",
            "phase2o_bounded_ready_but_epoch_blocked",
            "bounded_public_repair_pressure_passed",
            "bounded_multiseed_reproduction_passed",
        )
    )
    return {
        "artifact_family": "phase2w_epoch_gate",
        "bounded_stage_ready": bounded_stage_ready,
        "epoch_making_architecture_claim_ready": epoch_ready,
        "checks": checks,
        "epoch_claim_blockers": epoch_blockers,
        "allowed_next_action": "freeze_epoch_claim_evidence_and_request_external_review"
        if epoch_ready
        else "continue_phase2w_missing_hard_gates",
        "blocked_actions": []
        if epoch_ready
        else [
            "do_not_claim_epoch_making_architecture",
            "do_not_claim_production_autonomy",
            "do_not_claim_open_ended_repair_generalization",
        ],
        "current_evidence_status": (
            "strong_bounded_repair_pressure_ready"
            if bounded_stage_ready
            else "bounded_repair_pressure_incomplete"
        ),
        "next_missing_artifacts": [
            key.replace("_passed", ".json")
            for key in epoch_blockers
            if key
            not in {
                "phase2w_preregistered",
                "phase2o_bounded_ready_but_epoch_blocked",
                "bounded_public_repair_pressure_passed",
                "bounded_multiseed_reproduction_passed",
            }
        ],
        "inputs": {
            "preregistration_json": str(Path(preregistration_json)),
            "phase2o_readiness_json": str(Path(phase2o_readiness_json)),
            "bounded_repair_boundary_json": str(Path(bounded_repair_boundary_json)),
            "bounded_repair_reproduction_json": str(Path(bounded_repair_reproduction_json)),
            "independent_reproduction_json": str(Path(independent_reproduction_json))
            if independent_reproduction_json
            else None,
            "open_ended_repair_json": str(Path(open_ended_repair_json))
            if open_ended_repair_json
            else None,
            "live_agent_baseline_json": str(Path(live_agent_baseline_json))
            if live_agent_baseline_json
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
    parser = argparse.ArgumentParser(description="Audit Phase2W epoch-claim gate.")
    parser.add_argument("--preregistration-json", required=True)
    parser.add_argument("--phase2o-readiness-json", required=True)
    parser.add_argument("--bounded-repair-boundary-json", required=True)
    parser.add_argument("--bounded-repair-reproduction-json", required=True)
    parser.add_argument("--independent-reproduction-json")
    parser.add_argument("--open-ended-repair-json")
    parser.add_argument("--live-agent-baseline-json")
    parser.add_argument("--production-safety-json")
    parser.add_argument("--reviewer-consensus-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2w_epoch_gate(
        preregistration_json=args.preregistration_json,
        phase2o_readiness_json=args.phase2o_readiness_json,
        bounded_repair_boundary_json=args.bounded_repair_boundary_json,
        bounded_repair_reproduction_json=args.bounded_repair_reproduction_json,
        independent_reproduction_json=args.independent_reproduction_json,
        open_ended_repair_json=args.open_ended_repair_json,
        live_agent_baseline_json=args.live_agent_baseline_json,
        production_safety_json=args.production_safety_json,
        reviewer_consensus_json=args.reviewer_consensus_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["bounded_stage_ready"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
