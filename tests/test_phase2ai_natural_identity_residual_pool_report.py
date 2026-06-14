import json
from pathlib import Path

from reflexlm.cli.build_phase2ah_identity_contrast_pressure import (
    phase2ah_identity_contrast_row,
)
from reflexlm.cli.build_phase2ai_natural_identity_residual_pool_report import (
    build_phase2ai_natural_identity_residual_pool_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int) -> dict:
    candidates = [
        {
            "repair_action": f"repair_action_{index}_{slot}",
            "intent": "apply_patch_and_rerun_tests",
            "edit_scope": f"pkg/module_{slot}.py",
            "target_symbol": f"symbol_{index}_{slot}",
            "verification_command": "python -m pytest -q <generated_repair_test> --maxfail=1",
        }
        for slot in range(3)
    ]
    return {
        "trace_id": f"val:repo_{index}:{index}",
        "split": "val",
        "source_kind": "public_repo",
        "repo_id": f"repo_{index % 3}",
        "repo_url_or_origin": f"https://example.invalid/repo_{index % 3}.git",
        "current_visible_text": "public runtime repair evidence without oracle markers",
        "runtime_visible_evidence": {
            "changed_files": ["pkg/module_1.py"],
            "traceback_symbols": ["unhelpful_symbol"],
            "watched_files": ["tests/test_generated.py"],
            "pytest_before_patch": {"stdout_excerpt": "AssertionError"},
        },
        "repair_candidates": candidates,
        "expected_repair_action": candidates[1]["repair_action"],
        "expected_repair_result": {"test_target": "phase2s_repair_tests/test_case.py"},
        "normalization": {"sealed_feedback_absent": True},
    }


def _natural_source_correct_identity_wrong_row(index: int = 0) -> dict:
    row = phase2ah_identity_contrast_row(_row(index))
    assert row is not None
    row.pop("claim_boundary", None)
    row.pop("benchmark_family", None)
    row.pop("unsupported_claims", None)
    row["selection_origin"] = "unit_test_nonsealed_public_trace_fixture"
    return row


def test_phase2ai_pool_report_accepts_natural_source_correct_identity_wrong(
    tmp_path: Path,
) -> None:
    source = _write_jsonl(
        tmp_path / "pool" / "rows.jsonl",
        [_natural_source_correct_identity_wrong_row(index) for index in range(3)],
    )

    report = build_phase2ai_natural_identity_residual_pool_report(
        roots=[source],
        min_source_correct_identity_wrong_rows=3,
        min_unique_source_correct_identity_wrong_traces=3,
        min_source_correct_identity_wrong_repos=1,
    )

    assert report["passed"] is True
    assert report["metrics"]["bucket_counts"]["source_1_identity_0"] == 3
    assert report["metrics"]["source_correct_identity_wrong_unique_traces"] == 3
    assert report["allowed_next_action"] == "build_repo_origin_disjoint_phase2ai_split"


def test_phase2ai_pool_report_excludes_phase2ah_adversarial_rows_by_default(
    tmp_path: Path,
) -> None:
    adversarial = phase2ah_identity_contrast_row(_row(0))
    assert adversarial is not None
    source = _write_jsonl(tmp_path / "phase2ah_identity_contrast" / "rows.jsonl", [adversarial])

    report = build_phase2ai_natural_identity_residual_pool_report(roots=[source])

    assert report["passed"] is False
    assert report["metrics"]["source_correct_identity_wrong_rows"] == 0
    assert report["metrics"]["rejected_counts"]["adversarial_identity_contrast"] == 1


def test_phase2ai_pool_report_excludes_head_datasets_by_default(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "phase2s_heads" / "val.jsonl",
        [_natural_source_correct_identity_wrong_row()],
    )

    report = build_phase2ai_natural_identity_residual_pool_report(roots=[tmp_path])

    assert report["passed"] is False
    assert report["metrics"]["jsonl_inputs_scanned"] == 0
    assert report["checks"]["head_datasets_excluded"] is True
