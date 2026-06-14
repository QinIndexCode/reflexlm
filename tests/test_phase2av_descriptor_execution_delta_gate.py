import json
from pathlib import Path

from reflexlm.cli.audit_phase2av_descriptor_execution_delta_gate import (
    audit_phase2av_descriptor_execution_delta_gate,
)
from reflexlm.cli.run_phase2av_descriptor_selected_execution import CLAIM_BOUNDARY


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(
    index: int,
    *,
    mode: str,
    selected_slot: int,
    expected_slot: int,
    success: bool,
) -> dict:
    return {
        "trace_id": f"phase2av:holdout:{index:05d}",
        "source_kind": "public_repo",
        "selection_mode": mode,
        "selected_patch_candidate_slot": selected_slot,
        "expected_patch_candidate_slot": expected_slot,
        "patch_candidate_selected_correctly": selected_slot == expected_slot,
        "success": success,
        "patch_generator": "bounded_symbolic_structural_patch_v1",
        "rollback_failure_restored": True,
        "unauthorized_write_count": 0,
        "false_completion": False,
        "oracle_trace_used": False,
        "sealed_feedback_used": False,
        "claim_bearing_freeform_patch_evidence": False,
        "freeform_patch_generation": False,
        "low_level_qwen_calls": 0,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _full_rows(count: int = 20) -> list[dict]:
    return [
        _row(
            index,
            mode="adapter",
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
            mode="source_overlap",
            selected_slot=0,
            expected_slot=index % 4,
            success=index % 4 == 0,
        )
        for index in range(count)
    ]


def test_phase2av_descriptor_execution_delta_gate_accepts_nonzero_control_delta(
    tmp_path: Path,
) -> None:
    report = audit_phase2av_descriptor_execution_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", _full_rows()),
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl", _control_rows()
        ),
    )

    assert report["passed"] is True
    assert report["metrics"]["full_success_rate"] == 1.0
    assert report["metrics"]["control_success_rate"] == 0.25
    assert (
        "phase2av_bounded_descriptor_selected_symbolic_execution_delta_supported"
        in report["supported_claims"]
    )
    assert "learned_freeform_patch_generation" in report["unsupported_claims"]


def test_phase2av_descriptor_execution_delta_gate_rejects_all_zero_control(
    tmp_path: Path,
) -> None:
    control = [
        _row(index, mode="source_overlap", selected_slot=1, expected_slot=0, success=False)
        for index in range(20)
    ]

    report = audit_phase2av_descriptor_execution_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", _full_rows()),
        control_execution_jsonl=_write_jsonl(tmp_path / "control.jsonl", control),
    )

    assert report["passed"] is False
    assert report["checks"]["control_success_nonzero_met"] is False


def test_phase2av_descriptor_execution_delta_gate_rejects_ceiling_control(
    tmp_path: Path,
) -> None:
    control = [
        _row(
            index,
            mode="source_overlap",
            selected_slot=index % 4,
            expected_slot=index % 4,
            success=True,
        )
        for index in range(20)
    ]

    report = audit_phase2av_descriptor_execution_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", _full_rows()),
        control_execution_jsonl=_write_jsonl(tmp_path / "control.jsonl", control),
    )

    assert report["passed"] is False
    assert report["checks"]["control_success_not_ceiling_met"] is False
    assert report["checks"]["full_minus_control_success_delta_met"] is False


def test_phase2av_descriptor_execution_delta_gate_rejects_freeform_claim(
    tmp_path: Path,
) -> None:
    full = _full_rows()
    full[0]["freeform_patch_generation"] = True

    report = audit_phase2av_descriptor_execution_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", full),
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl", _control_rows()
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["bounded_symbolic_execution_only"] is False


def test_phase2av_descriptor_execution_delta_gate_rejects_oracle_control_mode(
    tmp_path: Path,
) -> None:
    control = _control_rows()
    control[0]["selection_mode"] = "gold_oracle"

    report = audit_phase2av_descriptor_execution_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", _full_rows()),
        control_execution_jsonl=_write_jsonl(tmp_path / "control.jsonl", control),
    )

    assert report["passed"] is False
    assert report["checks"]["control_non_oracle_selection_mode"] is False
