from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflexlm.cli.build_phase2av_operation_balanced_subset import (
    build_phase2av_operation_balanced_subset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, operation: str, *, evidence: str = "AssertionError") -> dict:
    return {
        "task_id": f"task-{index}",
        "repo_origin": f"https://example.test/repo-{index % 2}.git",
        "expected_policy": {"patch_operation": operation},
        "runtime_visible_evidence": {
            "pytest_before_patch": {
                "stdout_excerpt": evidence,
                "stderr_excerpt": "",
            }
        },
    }


def test_phase2av_operation_balanced_subset_downsamples_majority(tmp_path: Path) -> None:
    rows = [_row(0, "insert_import"), _row(1, "insert_import"), _row(2, "replace_attribute")]
    train = _write_jsonl(tmp_path / "train.jsonl", rows)
    val = _write_jsonl(tmp_path / "val.jsonl", rows)
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", rows)

    report = build_phase2av_operation_balanced_subset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        holdout_tasks_jsonl=holdout,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
    )

    assert report["passed"] is True
    assert report["split_counts"] == {"train": 2, "val": 2, "holdout": 2}
    assert report["split_reports"]["train"]["selected_operation_counts"] == {
        "insert_import": 1,
        "replace_attribute": 1,
    }
    assert report["full_training_allowed"] is False


def test_phase2av_operation_balanced_subset_can_keep_all_eligible_without_claiming_balance(
    tmp_path: Path,
) -> None:
    rows = [_row(0, "insert_import"), _row(1, "insert_import"), _row(2, "replace_attribute")]
    train = _write_jsonl(tmp_path / "train.jsonl", rows)
    val = _write_jsonl(tmp_path / "val.jsonl", rows)
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", rows)

    report = build_phase2av_operation_balanced_subset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        holdout_tasks_jsonl=holdout,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        keep_all_eligible=True,
    )

    assert report["passed"] is True
    assert report["operation_balanced"] is False
    assert report["keep_all_eligible"] is True
    assert report["split_counts"] == {"train": 3, "val": 3, "holdout": 3}
    assert report["split_reports"]["train"]["operation_limit"] is None
    assert report["split_reports"]["train"]["operation_minimum_count"] == 1
    assert report["split_reports"]["train"]["selected_operation_counts"] == {
        "insert_import": 2,
        "replace_attribute": 1,
    }
    assert report["full_training_allowed"] is False


def test_phase2av_operation_balanced_subset_rejects_single_operation(tmp_path: Path) -> None:
    rows = [_row(0, "insert_import"), _row(1, "insert_import")]
    train = _write_jsonl(tmp_path / "train.jsonl", rows)

    with pytest.raises(ValueError, match="at least two operations"):
        build_phase2av_operation_balanced_subset(
            train_tasks_jsonl=train,
            val_tasks_jsonl=train,
            holdout_tasks_jsonl=train,
            output_dir=tmp_path / "out",
            manifest_json=tmp_path / "manifest.json",
        )


def test_phase2av_operation_balanced_subset_can_require_known_failure_family(
    tmp_path: Path,
) -> None:
    rows = [
        _row(0, "insert_import", evidence="NameError: name 'os' is not defined"),
        _row(1, "insert_import", evidence="NameError: name 'sys' is not defined"),
        _row(2, "replace_attribute", evidence="AttributeError: 'str' object has no attribute 'x'"),
        _row(3, "replace_attribute", evidence="AttributeError: 'str' object has no attribute 'y'"),
        _row(4, "replace_attribute", evidence="failed without exception class"),
    ]
    train = _write_jsonl(tmp_path / "train.jsonl", rows)

    report = build_phase2av_operation_balanced_subset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=train,
        holdout_tasks_jsonl=train,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        require_known_failure_family=True,
    )

    assert report["passed"] is True
    assert report["split_reports"]["train"]["excluded_unknown_failure_family"] == 1
    assert report["split_reports"]["train"]["selected_failure_family_counts"] == {
        "attribute_missing_runtime": 2,
        "missing_import_or_symbol_runtime": 2,
    }


def test_phase2av_operation_balanced_subset_rejects_when_known_family_filter_removes_operation(
    tmp_path: Path,
) -> None:
    rows = [
        _row(0, "insert_import", evidence="failed without exception class"),
        _row(1, "replace_attribute", evidence="AttributeError: object has no attribute x"),
    ]
    train = _write_jsonl(tmp_path / "train.jsonl", rows)

    with pytest.raises(ValueError, match="at least two operations"):
        build_phase2av_operation_balanced_subset(
            train_tasks_jsonl=train,
            val_tasks_jsonl=train,
            holdout_tasks_jsonl=train,
            output_dir=tmp_path / "out",
            manifest_json=tmp_path / "manifest.json",
            require_known_failure_family=True,
        )
