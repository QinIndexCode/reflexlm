import json
from pathlib import Path

from reflexlm.cli.audit_phase2au_runtime_delta_gate import (
    PHASE2AU_RUNTIME_BOUNDARY,
    audit_phase2au_runtime_delta_gate,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _package_gate(*, passed: bool = True) -> dict:
    return {
        "artifact_family": "phase2au_policy_required_package_gate",
        "passed": passed,
        "ready_for_phase2au_runtime_delta_eval": passed,
    }


def _runtime_audit(
    *,
    passed: bool,
    policy_loaded: bool,
    rows: int = 20,
    success_rate: float = 1.0,
) -> dict:
    return {
        "artifact_family": "phase2au_policy_required_runtime_audit",
        "passed": passed,
        "claim_boundary": PHASE2AU_RUNTIME_BOUNDARY,
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
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_open_ended_debugging_generalization",
        ],
    }


def test_phase2au_runtime_delta_gate_accepts_bounded_package_delta(
    tmp_path: Path,
) -> None:
    report = audit_phase2au_runtime_delta_gate(
        full_runtime_audit_json=_write(
            tmp_path / "full.json",
            _runtime_audit(passed=True, policy_loaded=True, success_rate=0.9),
        ),
        control_runtime_audit_json=_write(
            tmp_path / "control.json",
            _runtime_audit(passed=False, policy_loaded=False, success_rate=0.65),
        ),
        package_gate_json=_write(tmp_path / "package.json", _package_gate()),
    )

    assert report["passed"] is True
    assert report["metrics"]["full_minus_control_success_rate"] == 0.25
    assert "phase2au_bounded_package_runtime_delta_supported" in report[
        "supported_claims"
    ]
    assert "epoch_making_architecture" in report["unsupported_claims"]


def test_phase2au_runtime_delta_gate_rejects_no_policy_tie(tmp_path: Path) -> None:
    report = audit_phase2au_runtime_delta_gate(
        full_runtime_audit_json=_write(
            tmp_path / "full.json",
            _runtime_audit(passed=True, policy_loaded=True, success_rate=0.9),
        ),
        control_runtime_audit_json=_write(
            tmp_path / "control.json",
            _runtime_audit(passed=False, policy_loaded=False, success_rate=0.9),
        ),
        package_gate_json=_write(tmp_path / "package.json", _package_gate()),
    )

    assert report["passed"] is False
    assert report["checks"]["full_minus_control_delta_met"] is False
    assert "do_not_claim_phase2au_package_runtime_delta" in report["blocked_actions"]


def test_phase2au_runtime_delta_gate_rejects_head_eval_postflight_as_runtime(
    tmp_path: Path,
) -> None:
    fake_full = _runtime_audit(passed=True, policy_loaded=True, success_rate=1.0)
    fake_full["artifact_family"] = "phase2au_policy_required_eval_postflight"
    report = audit_phase2au_runtime_delta_gate(
        full_runtime_audit_json=_write(tmp_path / "full.json", fake_full),
        control_runtime_audit_json=_write(
            tmp_path / "control.json",
            _runtime_audit(passed=False, policy_loaded=False, success_rate=0.5),
        ),
        package_gate_json=_write(tmp_path / "package.json", _package_gate()),
    )

    assert report["passed"] is False
    assert report["checks"]["not_head_eval_or_smoke_postflight"] is False


def test_phase2au_runtime_delta_gate_rejects_wrong_boundary(tmp_path: Path) -> None:
    control = _runtime_audit(passed=False, policy_loaded=False, success_rate=0.65)
    control["claim_boundary"] = "open_ended_repair"
    report = audit_phase2au_runtime_delta_gate(
        full_runtime_audit_json=_write(
            tmp_path / "full.json",
            _runtime_audit(passed=True, policy_loaded=True, success_rate=0.9),
        ),
        control_runtime_audit_json=_write(tmp_path / "control.json", control),
        package_gate_json=_write(tmp_path / "package.json", _package_gate()),
    )

    assert report["passed"] is False
    assert report["checks"]["same_phase2au_execution_boundary"] is False
