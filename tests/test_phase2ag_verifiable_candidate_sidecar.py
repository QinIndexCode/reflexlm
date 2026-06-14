import json
from pathlib import Path

from reflexlm.cli.audit_phase2ag_verifiable_candidate_sidecar import (
    audit_phase2ag_verifiable_candidate_sidecar,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(expected_slot: int = 1) -> dict:
    candidates = [
        {
            "repair_action": "wrong",
            "edit_scope": "pkg/a.py",
            "target_line": 10,
            "target_col": 1,
            "target_literal_hash": "wrong-literal",
            "structural_probe_hash": "wrong-probe",
        },
        {
            "repair_action": "correct",
            "edit_scope": "pkg/b.py",
            "target_line": 20,
            "target_col": 2,
            "target_literal_hash": "correct-literal",
            "structural_probe_hash": "correct-probe",
        },
    ]
    return {
        "trace_id": "row-1",
        "repo_id": "repo-a",
        "current_visible_text": "public repair row without oracle markers",
        "runtime_visible_evidence": {
            "structural_probe_hashes": ["correct-probe"],
            "expected_literal_hash": "correct-literal",
            "target_location": {"path": "pkg/b.py", "line": 20, "col": 2},
        },
        "repair_candidates": candidates,
        "expected_repair_action": candidates[expected_slot]["repair_action"],
    }


def test_phase2ag_verifiable_candidate_sidecar_accepts_unique_runtime_probe(
    tmp_path: Path,
) -> None:
    path = _write_jsonl(tmp_path / "rows.jsonl", [_row()])

    report = audit_phase2ag_verifiable_candidate_sidecar(
        jsonl=path,
        split_name="val",
        output_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["probe_accuracy"] == 1.0
    assert report["blocked_actions"] == []


def test_phase2ag_verifiable_candidate_sidecar_rejects_unresolved_probe(
    tmp_path: Path,
) -> None:
    row = _row()
    row["runtime_visible_evidence"] = {"pytest_before_patch": {"stdout_excerpt": "no probe"}}
    row["repair_candidates"][0].pop("target_literal_hash")
    row["repair_candidates"][1].pop("target_literal_hash")
    path = _write_jsonl(tmp_path / "rows.jsonl", [row])

    report = audit_phase2ag_verifiable_candidate_sidecar(
        jsonl=path,
        split_name="val",
        output_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["metrics"]["unresolved_probe_rows"] == 1
    assert report["unresolved_rows"][0]["probe_overlaps"] == [[], []]
    assert "do_not_train_claim_bearing_phase2ag_adapter" in report["blocked_actions"]


def test_phase2ag_verifiable_candidate_sidecar_rejects_wrong_probe(
    tmp_path: Path,
) -> None:
    row = _row()
    row["runtime_visible_evidence"]["structural_probe_hashes"] = ["wrong-probe"]
    row["runtime_visible_evidence"]["expected_literal_hash"] = "wrong-literal"
    row["runtime_visible_evidence"]["target_location"] = {"path": "pkg/a.py", "line": 10, "col": 1}
    path = _write_jsonl(tmp_path / "rows.jsonl", [row])

    report = audit_phase2ag_verifiable_candidate_sidecar(
        jsonl=path,
        split_name="val",
        output_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["metrics"]["incorrect_probe_rows"] == 1
    assert report["incorrect_rows"][0]["probe_prediction"] == 0


def test_phase2ag_verifiable_candidate_sidecar_rejects_markers_and_sealed_refs(
    tmp_path: Path,
) -> None:
    row = _row()
    row["current_visible_text"] = "gold candidate_0 leaked"
    row["normalization"] = {"sealed_feedback_used": True}
    path = _write_jsonl(tmp_path / "rows.jsonl", [row])

    report = audit_phase2ag_verifiable_candidate_sidecar(
        jsonl=path,
        split_name="val",
        output_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["metrics"]["marker_leak_rows"] == 1
    assert report["metrics"]["sealed_reference_rows"] == 1
