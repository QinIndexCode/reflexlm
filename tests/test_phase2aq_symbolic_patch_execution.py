import json
from pathlib import Path

from reflexlm.cli.audit_phase2aq_symbolic_patch_execution import (
    audit_phase2aq_symbolic_patch_execution,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(**overrides: object) -> dict:
    row = {
        "success": True,
        "policy_loaded": True,
        "policy_open_repair_outputs": {
            "patch_proposal": 1,
            "bounded_edit_scope": 1,
            "rollback_safety": 1,
        },
        "patch_generator": "bounded_symbolic_text_membership_patch_v1",
        "patch_source": "package_runtime_symbolic_text_membership_patch_proposal",
        "recorded_patch_artifact_used": False,
        "recorded_patch_artifact_used_for_fault_injection": True,
        "claim_bearing_execution_evidence": True,
        "claim_boundary": "bounded_runtime_symbolic_patch_proposal_only_not_open_ended_repair",
        "full_test_pass_rate": 1.0,
        "rollback_failure_restored": True,
        "sealed_feedback_used": False,
        "false_completion": False,
        "unauthorized_write_count": 0,
    }
    row.update(overrides)
    return row


def test_phase2aq_audit_accepts_runtime_symbolic_patch_smoke(tmp_path: Path) -> None:
    report = audit_phase2aq_symbolic_patch_execution(
        execution_results_jsonl=_write_jsonl(
            tmp_path / "results.jsonl",
            [_row() for _ in range(4)],
        ),
        min_rows=4,
        min_success_rate=1.0,
    )

    assert report["passed"] is True
    assert report["evidence_level"] == "smoke"
    assert report["claim_bearing_execution_evidence"] is True
    assert "bounded_runtime_symbolic_text_membership_patch_proposal_smoke_supported" in report["supported_claims"]


def test_phase2aq_audit_labels_24_row_holdout_evidence(tmp_path: Path) -> None:
    report = audit_phase2aq_symbolic_patch_execution(
        execution_results_jsonl=_write_jsonl(
            tmp_path / "results.jsonl",
            [_row() for _ in range(24)],
        ),
        min_rows=24,
        min_success_rate=1.0,
    )

    assert report["passed"] is True
    assert report["evidence_level"] == "holdout24"
    assert (
        "bounded_runtime_symbolic_text_membership_patch_proposal_holdout24_supported"
        in report["supported_claims"]
    )


def test_phase2aq_audit_rejects_recorded_patch_replay(tmp_path: Path) -> None:
    report = audit_phase2aq_symbolic_patch_execution(
        execution_results_jsonl=_write_jsonl(
            tmp_path / "results.jsonl",
            [
                _row(
                    patch_source="recorded_public_structural_patch_diff_operator",
                    recorded_patch_artifact_used=True,
                )
                for _ in range(4)
            ],
        ),
        min_rows=4,
        min_success_rate=1.0,
    )

    assert report["passed"] is False
    assert report["checks"]["no_rows_use_recorded_patch_as_proposal"] is False
    assert report["checks"]["all_rows_runtime_patch_source"] is False
