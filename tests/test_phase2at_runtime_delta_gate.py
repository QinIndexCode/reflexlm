import json
from pathlib import Path

from reflexlm.cli.audit_phase2at_runtime_delta_gate import (
    audit_phase2at_runtime_delta_gate,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _audit(
    *,
    passed: bool,
    policy_loaded: bool,
    rows: int = 256,
    success_rate: float = 1.0,
) -> dict:
    return {
        "passed": passed,
        "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
        "checks": {
            "all_rows_policy_loaded": policy_loaded,
            "sealed_feedback_absent": True,
        },
        "metrics": {
            "row_count": rows,
            "success_rate": success_rate,
            "failure_reasons": {},
        },
        "blocked_actions": [
            "do_not_claim_freeform_model_generated_patch_repair",
            "do_not_claim_open_ended_debugging_generalization",
        ],
    }


def test_phase2at_runtime_delta_gate_rejects_symbolic_runner_tie(
    tmp_path: Path,
) -> None:
    report = audit_phase2at_runtime_delta_gate(
        full_runtime_audit_json=_write(
            tmp_path / "full.json", _audit(passed=True, policy_loaded=True, success_rate=1.0)
        ),
        control_runtime_audit_json=_write(
            tmp_path / "control.json",
            _audit(passed=False, policy_loaded=False, success_rate=1.0),
        ),
    )

    assert report["passed"] is False
    assert report["metrics"]["full_minus_control_success_rate"] == 0.0
    assert "do_not_claim_package_runtime_mechanism_delta" in report["blocked_actions"]
    assert "phase2at_package_runtime_delta_supported" not in report["supported_claims"]


def test_phase2at_runtime_delta_gate_accepts_policy_delta(
    tmp_path: Path,
) -> None:
    report = audit_phase2at_runtime_delta_gate(
        full_runtime_audit_json=_write(
            tmp_path / "full.json", _audit(passed=True, policy_loaded=True, success_rate=0.92)
        ),
        control_runtime_audit_json=_write(
            tmp_path / "control.json",
            _audit(passed=False, policy_loaded=False, success_rate=0.70),
        ),
        min_full_minus_control=0.15,
    )

    assert report["passed"] is True
    assert report["metrics"]["full_minus_control_success_rate"] >= 0.15
    assert "phase2at_package_runtime_delta_supported" in report["supported_claims"]


def test_phase2at_runtime_delta_gate_rejects_boundary_drift(
    tmp_path: Path,
) -> None:
    control = _audit(passed=False, policy_loaded=False, success_rate=0.70)
    control["claim_boundary"] = "open_ended_repair"
    report = audit_phase2at_runtime_delta_gate(
        full_runtime_audit_json=_write(
            tmp_path / "full.json", _audit(passed=True, policy_loaded=True, success_rate=0.92)
        ),
        control_runtime_audit_json=_write(tmp_path / "control.json", control),
    )

    assert report["passed"] is False
    assert report["checks"]["same_execution_boundary"] is False
