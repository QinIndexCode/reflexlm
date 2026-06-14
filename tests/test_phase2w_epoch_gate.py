import json
from pathlib import Path

from reflexlm.cli.audit_phase2w_epoch_gate import build_phase2w_epoch_gate


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prereg() -> dict:
    return {"passed": True}


def _readiness() -> dict:
    return {
        "bounded_mechanism_claim_ready": True,
        "epoch_making_architecture_claim_ready": False,
    }


def _bounded() -> dict:
    return {
        "passed": True,
        "metrics": {
            "holdout_command_slot_accuracy": 0.91,
            "holdout_model_minus_source_overlap_accuracy": 0.53,
            "holdout_model_minus_zero_nsi_accuracy": 0.54,
        },
    }


def _reproduction() -> dict:
    return {
        "passed": True,
        "metrics": {"eligible_model_count": 2, "eligible_run_count": 6},
    }


def _independent() -> dict:
    return {
        "passed": True,
        "runner_independent": True,
        "one_command_reproduction": True,
        "hash_locked_splits": True,
        "no_local_patch_required": True,
    }


def _open_repair() -> dict:
    return {
        "passed": True,
        "task_family": "open_ended_repair",
        "metrics": {
            "full_package": {"task_success": 0.9, "patch_correctness": 0.88},
            "best_live_agent_baseline": {
                "task_success": 0.75,
                "patch_correctness": 0.70,
            },
        },
    }


def _live_agent() -> dict:
    return {
        "passed": True,
        "baseline_kind": "live_tool_agent",
        "model_or_provider": "baseline-model",
        "tool_budget": 30,
        "context_policy": "bounded",
        "retry_policy": "single_retry",
        "edit_permissions": "sandbox_only",
        "stop_rule": "tests_pass_or_budget_exhausted",
        "cost_or_command_budget": {"max_commands": 30},
        "metrics": {"task_success": 0.75},
    }


def _safety() -> dict:
    return {
        "passed": True,
        "unauthorized_write_count": 0,
        "allowlist_hallucination": 0,
        "low_level_qwen_calls": 0,
        "rollback_success": 1.0,
        "false_completion_rate": 0.0,
    }


def _reviewers() -> dict:
    return {"passed": True, "read_only": True, "unanimous": True}


def _base_args(tmp_path: Path) -> dict:
    return {
        "preregistration_json": _write(tmp_path / "prereg.json", _prereg()),
        "phase2o_readiness_json": _write(tmp_path / "readiness.json", _readiness()),
        "bounded_repair_boundary_json": _write(tmp_path / "bounded.json", _bounded()),
        "bounded_repair_reproduction_json": _write(
            tmp_path / "reproduction.json", _reproduction()
        ),
    }


def test_phase2w_epoch_gate_accepts_bounded_stage_but_blocks_epoch_without_external_gates(
    tmp_path: Path,
) -> None:
    report = build_phase2w_epoch_gate(**_base_args(tmp_path))
    assert report["bounded_stage_ready"] is True
    assert report["epoch_making_architecture_claim_ready"] is False
    assert "independent_external_reproduction_passed" in report["epoch_claim_blockers"]
    assert "open_ended_repair_benchmark.json" in report["next_missing_artifacts"]
    assert "do_not_claim_epoch_making_architecture" in report["blocked_actions"]


def test_phase2w_epoch_gate_accepts_only_when_all_epoch_gates_pass(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    args.update(
        {
            "independent_reproduction_json": _write(
                tmp_path / "independent.json", _independent()
            ),
            "open_ended_repair_json": _write(tmp_path / "open.json", _open_repair()),
            "live_agent_baseline_json": _write(tmp_path / "agent.json", _live_agent()),
            "production_safety_json": _write(tmp_path / "safety.json", _safety()),
            "reviewer_consensus_json": _write(tmp_path / "reviewers.json", _reviewers()),
        }
    )
    report = build_phase2w_epoch_gate(**args)
    assert report["bounded_stage_ready"] is True
    assert report["epoch_making_architecture_claim_ready"] is True
    assert report["epoch_claim_blockers"] == []


def test_phase2w_epoch_gate_blocks_weak_live_agent_delta(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    open_repair = _open_repair()
    open_repair["metrics"]["best_live_agent_baseline"]["task_success"] = 0.85
    args.update(
        {
            "independent_reproduction_json": _write(
                tmp_path / "independent.json", _independent()
            ),
            "open_ended_repair_json": _write(tmp_path / "open.json", open_repair),
            "live_agent_baseline_json": _write(tmp_path / "agent.json", _live_agent()),
            "production_safety_json": _write(tmp_path / "safety.json", _safety()),
            "reviewer_consensus_json": _write(tmp_path / "reviewers.json", _reviewers()),
        }
    )
    report = build_phase2w_epoch_gate(**args)
    assert report["bounded_stage_ready"] is True
    assert report["epoch_making_architecture_claim_ready"] is False
    assert "open_ended_repair_benchmark_passed" in report["epoch_claim_blockers"]


def test_phase2w_epoch_gate_rejects_independent_reproduction_requiring_local_patch(
    tmp_path: Path,
) -> None:
    args = _base_args(tmp_path)
    independent = _independent()
    independent["no_local_patch_required"] = False
    args.update(
        {
            "independent_reproduction_json": _write(
                tmp_path / "independent.json", independent
            ),
            "open_ended_repair_json": _write(tmp_path / "open.json", _open_repair()),
            "live_agent_baseline_json": _write(tmp_path / "agent.json", _live_agent()),
            "production_safety_json": _write(tmp_path / "safety.json", _safety()),
            "reviewer_consensus_json": _write(tmp_path / "reviewers.json", _reviewers()),
        }
    )
    report = build_phase2w_epoch_gate(**args)
    assert report["epoch_making_architecture_claim_ready"] is False
    assert "independent_external_reproduction_passed" in report["epoch_claim_blockers"]
