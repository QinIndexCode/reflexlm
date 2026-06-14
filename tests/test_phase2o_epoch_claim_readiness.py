import json
from pathlib import Path

from reflexlm.cli.audit_phase2o_epoch_claim_readiness import (
    build_phase2o_epoch_claim_readiness,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _phase2p() -> dict:
    return {
        "passed": True,
        "models": [
            {"family": "Qwen2.5"},
            {"family": "Qwen2.5"},
            {"family": "Qwen2.5"},
            {"family": "TinyLlama"},
            {"family": "SmolLM2"},
        ],
        "aggregate": {
            "model_count": 5,
            "seed_count": 3,
            "pass_rate": 1.0,
            "full_low_level_qwen_calls_max": 0,
            "full_state_hallucination_max": 0.0,
        },
    }


def _phase2q() -> dict:
    return {"passed": True, "rollup": {"repo_count": 8, "holdout_rows": 256}}


def _phase2r() -> dict:
    return {
        "passed": True,
        "rollup": {"repo_count": 8, "dynamic_execution_rows": 1024},
    }


def _phase2v() -> dict:
    return {
        "passed": True,
        "checks": {"phase2v_independence_passed": True},
        "metrics": {"phase2v_full_minus_best_nonfull": 0.48},
    }


def test_epoch_claim_readiness_keeps_epoch_claim_blocked_without_phase2w_gates(
    tmp_path: Path,
) -> None:
    report = build_phase2o_epoch_claim_readiness(
        phase2p_summary_json=_write(tmp_path / "phase2p.json", _phase2p()),
        phase2q_gate_json=_write(tmp_path / "phase2q.json", _phase2q()),
        phase2r_gate_json=_write(tmp_path / "phase2r.json", _phase2r()),
        phase2v_evidence_json=_write(tmp_path / "phase2v.json", _phase2v()),
    )
    assert report["bounded_mechanism_claim_ready"] is True
    assert report["epoch_making_architecture_claim_ready"] is False
    assert "independent_external_reproduction_passed" in report["epoch_claim_blockers"]
    assert "open_ended_repair_benchmark_passed" in report["epoch_claim_blockers"]
    assert report["next_required_phase"]["name"].startswith("Phase2W")


def test_epoch_claim_readiness_accepts_only_after_all_strong_gates_pass(tmp_path: Path) -> None:
    report = build_phase2o_epoch_claim_readiness(
        phase2p_summary_json=_write(tmp_path / "phase2p.json", _phase2p()),
        phase2q_gate_json=_write(tmp_path / "phase2q.json", _phase2q()),
        phase2r_gate_json=_write(tmp_path / "phase2r.json", _phase2r()),
        phase2v_evidence_json=_write(tmp_path / "phase2v.json", _phase2v()),
        independent_reproduction_json=_write(
            tmp_path / "independent.json",
            {"passed": True, "runner_independent": True},
        ),
        open_repair_benchmark_json=_write(
            tmp_path / "open_repair.json",
            {"passed": True, "task_family": "open_ended_repair"},
        ),
        modern_agent_baseline_json=_write(
            tmp_path / "agent.json",
            {"passed": True, "baseline_kind": "live_tool_agent"},
        ),
        production_safety_json=_write(
            tmp_path / "safety.json",
            {"passed": True, "unauthorized_write_count": 0, "rollback_success": 1.0},
        ),
        reviewer_consensus_json=_write(
            tmp_path / "reviewers.json",
            {"passed": True, "unanimous": True, "read_only": True},
        ),
    )
    assert report["bounded_mechanism_claim_ready"] is True
    assert report["epoch_making_architecture_claim_ready"] is True
    assert report["epoch_claim_blockers"] == []


def test_epoch_claim_readiness_blocks_weak_phase2v_delta(tmp_path: Path) -> None:
    phase2v = _phase2v()
    phase2v["metrics"]["phase2v_full_minus_best_nonfull"] = 0.01
    report = build_phase2o_epoch_claim_readiness(
        phase2p_summary_json=_write(tmp_path / "phase2p.json", _phase2p()),
        phase2q_gate_json=_write(tmp_path / "phase2q.json", _phase2q()),
        phase2r_gate_json=_write(tmp_path / "phase2r.json", _phase2r()),
        phase2v_evidence_json=_write(tmp_path / "phase2v.json", phase2v),
    )
    assert report["bounded_mechanism_claim_ready"] is False
    assert "graded_nonzero_control_transfer_passed" in report["epoch_claim_blockers"]
