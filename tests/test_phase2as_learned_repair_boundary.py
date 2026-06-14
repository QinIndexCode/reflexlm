import json
from pathlib import Path

from reflexlm.cli.audit_phase2as_learned_repair_boundary import (
    audit_phase2as_learned_repair_boundary,
    classify_repair_evidence_row,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _symbolic_row(index: int = 0) -> dict:
    return {
        "trace_id": f"symbolic:{index}",
        "success": True,
        "patch_source": "package_runtime_symbolic_structural_patch_proposal",
        "patch_generator": "bounded_symbolic_structural_patch_v1",
        "recorded_patch_artifact_used": False,
        "recorded_patch_artifact_used_for_fault_injection": True,
        "claim_bearing_execution_evidence": True,
        "claim_bearing_freeform_patch_evidence": False,
        "freeform_patch_generation": False,
        "sealed_feedback_used": False,
    }


def _candidate_row(index: int = 0) -> dict:
    return {
        "trace_id": f"candidate:{index}",
        "success": True,
        "patch_source": "selected_recorded_correct_patch_candidate",
        "patch_generator": "bounded_patch_candidate_selector_v1",
        "patch_candidate_selected_correctly": True,
        "claim_bearing_candidate_selection_evidence": True,
        "claim_bearing_freeform_patch_evidence": False,
        "freeform_patch_generation": False,
        "sealed_feedback_used": False,
    }


def test_phase2as_classifier_separates_symbolic_from_candidate_selection() -> None:
    assert classify_repair_evidence_row(_symbolic_row()) == "symbolic_runtime_generator"
    assert classify_repair_evidence_row(_candidate_row()) == "bounded_patch_candidate_selection"
    assert (
        classify_repair_evidence_row(
            {
                "patch_source": "recorded_public_structural_patch_diff_operator",
                "patch_generator": "public_structural_recorded_diff_operator_v1",
                "recorded_patch_artifact_used": True,
            }
        )
        == "recorded_patch_control"
    )
    assert (
        classify_repair_evidence_row(
            {
                "patch_source": "control_runtime_symbolic_ast_attribute_patch",
                "control_execution_evidence": True,
                "claim_bearing_execution_evidence": False,
            }
        )
        == "restricted_symbolic_control"
    )


def test_phase2as_accepts_symbolic_claim_but_blocks_learned_relabel(tmp_path: Path) -> None:
    rows = _write_jsonl(tmp_path / "symbolic.jsonl", [_symbolic_row(i) for i in range(8)])

    symbolic_report = audit_phase2as_learned_repair_boundary(
        execution_results_jsonl=rows,
        claimed_capability="symbolic_structural_patch_proposal",
        min_rows=8,
        min_success_rate=1.0,
    )
    learned_report = audit_phase2as_learned_repair_boundary(
        execution_results_jsonl=rows,
        claimed_capability="learned_patch_generation",
        min_rows=8,
        min_success_rate=1.0,
    )

    assert symbolic_report["passed"] is True
    assert (
        "phase2as_symbolic_runtime_generator_boundary_preserved"
        in symbolic_report["supported_claims"]
    )
    assert learned_report["passed"] is False
    assert learned_report["checks"]["no_rows_symbolic_runtime_generator"] is False
    assert "learned_patch_generation" in learned_report["unsupported_claims"]
    assert (
        "do_not_relabel_symbolic_phase2ar_as_learned_patch_generation"
        in learned_report["blocked_actions"]
    )


def test_phase2as_accepts_candidate_selection_without_patch_authorship_claim(
    tmp_path: Path,
) -> None:
    rows = _write_jsonl(tmp_path / "candidate.jsonl", [_candidate_row(i) for i in range(8)])

    report = audit_phase2as_learned_repair_boundary(
        execution_results_jsonl=rows,
        claimed_capability="bounded_patch_candidate_selection",
        min_rows=8,
        min_success_rate=1.0,
        min_selection_accuracy=1.0,
    )

    assert report["passed"] is True
    assert report["checks"]["all_rows_bounded_patch_candidate_selection"] is True
    assert report["checks"]["candidate_selection_does_not_claim_patch_authorship"] is True
    assert report["metrics"]["evidence_class_counts"] == {
        "bounded_patch_candidate_selection": 8
    }
    assert "freeform_patch_generation" in report["unsupported_claims"]


def test_phase2as_rejects_candidate_selection_when_relabelled_as_generation(
    tmp_path: Path,
) -> None:
    rows = _write_jsonl(tmp_path / "candidate.jsonl", [_candidate_row(i) for i in range(8)])

    report = audit_phase2as_learned_repair_boundary(
        execution_results_jsonl=rows,
        claimed_capability="learned_patch_generation",
        min_rows=8,
        min_success_rate=1.0,
    )

    assert report["passed"] is False
    assert report["checks"]["all_rows_have_explicit_learned_patch_generator"] is False
    assert (
        "do_not_relabel_bounded_candidate_selection_as_patch_generation"
        in report["blocked_actions"]
    )
