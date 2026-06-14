import json
from pathlib import Path

from reflexlm.cli.build_phase2w_open_repair_missing_report import (
    build_phase2w_open_repair_missing_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2w_open_repair_missing_report_blocks_relabeling_bounded_repair(
    tmp_path: Path,
) -> None:
    report = build_phase2w_open_repair_missing_report(
        bounded_repair_boundary_json=_write(
            tmp_path / "bounded.json",
            {
                "passed": True,
                "active_evidence_boundary": "phase2s_public_repo_origin_disjoint_repair_positive_bounded_mechanism",
            },
        ),
        intended_results_jsonl=tmp_path / "open_results.jsonl",
    )
    assert report["passed"] is False
    assert report["claim_boundary"] == "missing_report_only_not_evidence"
    assert "do_not_relabel_phase2s_bounded_repair_as_open_ended" in report["blocked_actions"]
    assert report["bounded_repair_boundary_passed"] is True
    assert "repo_origin" in report["required_row_fields"]
    assert "full_transcript_sha256" in report["required_row_fields"]
