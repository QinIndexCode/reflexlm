import json
from pathlib import Path

from reflexlm.cli.build_phase2af_shortcut_bucket_gap_report import (
    build_phase2af_shortcut_bucket_gap_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, *, expected_slot: int, visible_symbol_slot: int) -> dict:
    candidates = [
        {
            "repair_action": f"repair_action_{index}_{slot}",
            "intent": "apply_patch_and_rerun_tests",
            "edit_scope": "pkg/module.py",
            "target_symbol": f"symbol_{index}_{slot}",
            "verification_command": "python -m pytest -q <generated_repair_test> --maxfail=1",
        }
        for slot in range(4)
    ]
    return {
        "trace_id": f"val:repo_{index}:{index}",
        "split": "val",
        "source_kind": "public_repo",
        "repo_id": f"repo_{index % 3}",
        "repo_url_or_origin": f"https://example.invalid/repo_{index % 3}.git",
        "current_visible_text": "public runtime repair evidence without oracle markers",
        "runtime_visible_evidence": {
            "changed_files": ["pkg/module.py"],
            "traceback_symbols": [f"symbol_{index}_{visible_symbol_slot}"],
            "target_location": {"path": "pkg/module.py"},
            "watched_files": ["tests/test_generated.py"],
            "pytest_before_patch": {"stdout_excerpt": "AssertionError"},
        },
        "repair_candidates": candidates,
        "expected_repair_action": candidates[expected_slot]["repair_action"],
        "expected_repair_result": {"test_target": "phase2s_repair_tests/test_case.py"},
        "normalization": {"sealed_feedback_absent": True},
    }


def test_phase2af_shortcut_bucket_gap_report_rejects_missing_identity_wrong_source_correct(
    tmp_path: Path,
) -> None:
    rows = [_row(index, expected_slot=1, visible_symbol_slot=1) for index in range(4)]
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    report = build_phase2af_shortcut_bucket_gap_report(
        jsonls=[source],
        output_json=tmp_path / "report.json",
    )

    assert report["passed"] is False
    assert report["checks"]["identity_wrong_source_correct_bucket_present"] is False
    assert "do_not_claim_model_beats_runtime_identity" in report["blocked_actions"]


def test_phase2af_shortcut_bucket_gap_report_accepts_identity_wrong_source_correct(
    tmp_path: Path,
) -> None:
    rows = [
        _row(index, expected_slot=1, visible_symbol_slot=1)
        for index in range(4)
    ]
    for row in rows:
        for slot, candidate in enumerate(row["repair_candidates"]):
            candidate["edit_scope"] = f"pkg/module_{slot}.py"
            candidate["target_literal_hash"] = f"literal_{slot}"
            candidate["target_line"] = 10 + slot
            candidate["target_col"] = 3
        row["runtime_visible_evidence"]["changed_files"] = ["pkg/module_1.py"]
        row["runtime_visible_evidence"]["traceback_symbols"] = ["symbol_999_0"]
        row["runtime_visible_evidence"]["expected_literal_hash"] = "literal_0"
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    report = build_phase2af_shortcut_bucket_gap_report(
        jsonls=[source],
        output_json=tmp_path / "report.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["bucket_counts"]["source_1_identity_0"] == 4
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["passed"] is True
