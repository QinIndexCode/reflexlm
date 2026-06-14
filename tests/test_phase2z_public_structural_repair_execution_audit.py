import json
from pathlib import Path

from reflexlm.cli.audit_phase2z_public_structural_repair_execution import (
    audit_phase2z_public_structural_repair_execution,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(success: bool, outputs: dict | None = None) -> dict:
    return {
        "success": success,
        "policy_loaded": True,
        "source_kind": "public_repo",
        "claim_boundary": "public_structural_recorded_patch_runtime_control_only_not_model_patch_generation",
        "claim_bearing_execution_evidence": False,
        "recorded_patch_artifact_used": True,
        "oracle_trace_used": True,
        "sealed_feedback_used": False,
        "policy_open_repair_outputs": outputs
        if outputs is not None
        else {"patch_proposal": 1, "bounded_edit_scope": 1, "rollback_safety": 1},
        "full_test_pass_rate": 1.0 if success else 0.0,
        "rollback_failure_restored": success,
    }


def _symbolic_row(success: bool) -> dict:
    row = _row(success)
    row.update(
        {
            "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
            "claim_bearing_execution_evidence": True,
            "recorded_patch_artifact_used": False,
            "oracle_trace_used": False,
            "patch_source": "package_runtime_symbolic_structural_patch_proposal",
            "patch_generator": "bounded_symbolic_structural_patch_v1",
        }
    )
    return row


def test_phase2z_public_structural_execution_audit_accepts_bounded_success(
    tmp_path: Path,
) -> None:
    results = _write_jsonl(
        tmp_path / "results.jsonl",
        [_row(True) for _ in range(7)] + [_row(False, {}) for _ in range(3)],
    )

    report = audit_phase2z_public_structural_repair_execution(
        execution_results_jsonl=results,
        min_rows=10,
        min_success_rate=0.7,
    )

    assert report["passed"] is True
    assert report["claim_bearing_execution_evidence"] is False
    assert report["metrics"]["failure_reasons"]["missing_open_repair_head_outputs"] == 3
    assert "do_not_claim_freeform_model_generated_patch_repair" in report["blocked_actions"]


def test_phase2z_public_structural_execution_audit_accepts_bounded_symbolic_structural(
    tmp_path: Path,
) -> None:
    results = _write_jsonl(
        tmp_path / "results.jsonl",
        [_symbolic_row(True) for _ in range(4)],
    )

    report = audit_phase2z_public_structural_repair_execution(
        execution_results_jsonl=results,
        min_rows=4,
        min_success_rate=1.0,
    )

    assert report["passed"] is True
    assert report["claim_bearing_execution_evidence"] is True
    assert report["checks"]["symbolic_structural_boundary_valid"] is True
    assert report["checks"]["no_rows_claim_freeform_model_patch_generation"] is True
    assert report["metrics"]["symbolic_structural_mode"] is True


def test_phase2z_public_structural_execution_audit_rejects_symbolic_boundary_drift(
    tmp_path: Path,
) -> None:
    row = _symbolic_row(True)
    row["patch_generator"] = "freeform_llm_patch"
    results = _write_jsonl(tmp_path / "results.jsonl", [row])

    report = audit_phase2z_public_structural_repair_execution(
        execution_results_jsonl=results,
        min_rows=1,
        min_success_rate=1.0,
    )

    assert report["passed"] is False
    assert report["checks"]["symbolic_structural_boundary_valid"] is False
    assert report["checks"]["no_rows_claim_freeform_model_patch_generation"] is False


def test_phase2z_public_structural_execution_audit_rejects_low_success(
    tmp_path: Path,
) -> None:
    results = _write_jsonl(
        tmp_path / "results.jsonl",
        [_row(True) for _ in range(6)] + [_row(False, {}) for _ in range(4)],
    )

    report = audit_phase2z_public_structural_repair_execution(
        execution_results_jsonl=results,
        min_rows=10,
        min_success_rate=0.7,
    )

    assert report["passed"] is False
    assert report["checks"]["success_rate_minimum_met"] is False
