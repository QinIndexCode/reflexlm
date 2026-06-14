import json
from pathlib import Path

from reflexlm.cli.audit_phase2au_patch_execution_delta_gate import (
    audit_phase2au_patch_execution_delta_gate,
)
from reflexlm.cli.build_phase2aa_bounded_patch_candidates import CLAIM_BOUNDARY


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(
    index: int,
    *,
    policy_loaded: bool,
    selected_slot: int,
    expected_slot: int,
    success: bool,
) -> dict:
    correct = selected_slot == expected_slot
    return {
        "trace_id": f"holdout:repo:{index}",
        "source_kind": "public_repo",
        "claim_boundary": CLAIM_BOUNDARY,
        "native_policy_label": "phase2au_qwen0_5b_policy_required_identityrefv2_capacity_package",
        "policy_loaded": policy_loaded,
        "selected_patch_candidate_slot": selected_slot,
        "expected_patch_candidate_slot": expected_slot,
        "patch_candidate_selected_correctly": correct,
        "success": success,
        "patch_source": "selected_recorded_correct_patch_candidate"
        if correct
        else "selected_bounded_distractor_patch_candidate",
        "patch_generator": "bounded_patch_candidate_selector_v1",
        "recorded_patch_artifact_used": correct,
        "rollback_failure_restored": True,
        "unauthorized_write_count": 0,
        "false_completion": False,
        "oracle_trace_used": False,
        "sealed_feedback_used": False,
        "claim_bearing_freeform_patch_evidence": False,
        "freeform_patch_generation": False,
        "low_level_qwen_calls": 0,
    }


def _full_rows(count: int = 20) -> list[dict]:
    return [
        _row(
            index,
            policy_loaded=True,
            selected_slot=index % 4,
            expected_slot=index % 4,
            success=True,
        )
        for index in range(count)
    ]


def _control_rows(count: int = 20) -> list[dict]:
    return [
        _row(
            index,
            policy_loaded=False,
            selected_slot=0,
            expected_slot=index % 4,
            success=(index % 4 == 0),
        )
        for index in range(count)
    ]


def test_phase2au_patch_execution_delta_gate_accepts_bounded_delta(
    tmp_path: Path,
) -> None:
    report = audit_phase2au_patch_execution_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", _full_rows()),
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl", _control_rows()
        ),
    )

    assert report["passed"] is True
    assert report["metrics"]["full_success_rate"] == 1.0
    assert report["metrics"]["control_success_rate"] == 0.25
    assert "phase2au_bounded_recorded_patch_candidate_execution_delta_supported" in report[
        "supported_claims"
    ]
    assert "learned_freeform_patch_generation" in report["unsupported_claims"]


def test_phase2au_patch_execution_delta_gate_rejects_freeform_patch_claim(
    tmp_path: Path,
) -> None:
    full = _full_rows()
    full[0]["freeform_patch_generation"] = True

    report = audit_phase2au_patch_execution_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", full),
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl", _control_rows()
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["bounded_patch_execution_only"] is False
    assert "do_not_claim_freeform_patch_generation" in report["blocked_actions"]


def test_phase2au_patch_execution_delta_gate_rejects_non_phase2au_package_label(
    tmp_path: Path,
) -> None:
    full = _full_rows()
    full[0]["native_policy_label"] = "phase2aa_candidate_selector"

    report = audit_phase2au_patch_execution_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", full),
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl", _control_rows()
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["phase2au_policy_label_present"] is False


def test_phase2au_patch_execution_delta_gate_rejects_oracle_control(
    tmp_path: Path,
) -> None:
    oracle_control = [
        _row(
            index,
            policy_loaded=False,
            selected_slot=index % 4,
            expected_slot=index % 4,
            success=True,
        )
        for index in range(20)
    ]

    report = audit_phase2au_patch_execution_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", _full_rows()),
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl", oracle_control
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["control_is_fixed_non_oracle_slot"] is False


def test_phase2au_patch_execution_delta_gate_rejects_execution_safety_failure(
    tmp_path: Path,
) -> None:
    full = _full_rows()
    full[0]["rollback_failure_restored"] = False
    full[1]["unauthorized_write_count"] = 1

    report = audit_phase2au_patch_execution_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", full),
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl", _control_rows()
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["execution_safety_met"] is False
