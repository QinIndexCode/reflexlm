import json
from pathlib import Path

from reflexlm.cli.audit_phase2ar_symbolic_structural_execution import (
    audit_phase2ar_symbolic_structural_execution,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(kind: str, **overrides: object) -> dict:
    row = {
        "success": True,
        "policy_loaded": True,
        "patch_generator": "bounded_symbolic_structural_patch_v1",
        "patch_source": "package_runtime_symbolic_structural_patch_proposal",
        "symbolic_patch_kinds": [kind],
        "recorded_patch_artifact_used": False,
        "recorded_patch_artifact_used_for_fault_injection": True,
        "claim_bearing_execution_evidence": True,
        "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
        "rollback_failure_restored": True,
        "sealed_feedback_used": False,
        "false_completion": False,
        "unauthorized_write_count": 0,
    }
    row.update(overrides)
    return row


def test_phase2ar_execution_audit_accepts_diverse_successful_holdout(tmp_path: Path) -> None:
    rows = [
        _row("text_membership" if index % 2 == 0 else "ast_attribute_restoration")
        for index in range(8)
    ]

    report = audit_phase2ar_symbolic_structural_execution(
        execution_results_jsonl=_write_jsonl(tmp_path / "results.jsonl", rows),
        min_rows=8,
        min_success_rate=1.0,
    )

    assert report["passed"] is True
    assert report["checks"]["required_patch_kinds_present"] is True


def test_phase2ar_execution_audit_rejects_recorded_patch_replay(tmp_path: Path) -> None:
    rows = [
        _row(
            "text_membership",
            patch_source="recorded_public_structural_patch_diff_operator",
            recorded_patch_artifact_used=True,
        )
        for _ in range(8)
    ]

    report = audit_phase2ar_symbolic_structural_execution(
        execution_results_jsonl=_write_jsonl(tmp_path / "results.jsonl", rows),
        min_rows=8,
        min_success_rate=1.0,
    )

    assert report["passed"] is False
    assert report["checks"]["no_rows_use_recorded_patch_as_proposal"] is False
    assert report["checks"]["all_rows_runtime_structural_patch_source"] is False
