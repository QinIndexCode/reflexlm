import json
from pathlib import Path

from reflexlm.cli.audit_phase2av_descriptor_pool_gap import (
    audit_phase2av_descriptor_pool_gap,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    return path


def _row(
    index: int,
    *,
    split: str,
    operation: str,
    template: str,
    repo: str,
    family: str = "missing_import_or_symbol_runtime",
) -> dict:
    return {
        "task_id": f"{split}:{index}",
        "split": split,
        "repo_origin": repo,
        "learned_patch_descriptor_target": {
            "operation": operation,
            "after_fragment_template_id": template,
        },
        "runtime_visible_evidence": {"descriptor_failure_family": family},
    }


def test_phase2av_descriptor_pool_gap_accepts_diverse_pool(tmp_path: Path) -> None:
    paths = []
    for split in ("train", "val", "holdout"):
        rows = []
        for operation, template in [
            ("insert_import", "import_restoration"),
            ("replace_attribute", "call_attribute_restoration"),
            ("replace_literal", "literal_restoration"),
        ]:
            for index in range(5):
                rows.append(
                    _row(
                        index,
                        split=split,
                        operation=operation,
                        template=template,
                        repo=f"repo-{index % 3}",
                    )
                )
        paths.append(_write_jsonl(tmp_path / f"{split}.jsonl", rows))

    report = audit_phase2av_descriptor_pool_gap(
        jsonl_paths=paths,
        min_total_rows=45,
        min_rows_per_split=15,
        min_operations_per_split=3,
        min_examples_per_operation=5,
        min_repo_origins_per_split=3,
    )

    assert report["passed"] is True
    assert report["ready_for_phase2av_full_data_construction"] is True
    assert report["blocking_reasons"] == []


def test_phase2av_descriptor_pool_gap_rejects_small_two_operation_pool(
    tmp_path: Path,
) -> None:
    train = [
        _row(i, split="train", operation="insert_import", template="import_restoration", repo="a")
        for i in range(12)
    ] + [
        _row(
            i,
            split="train",
            operation="replace_attribute",
            template="call_attribute_restoration",
            repo="b",
            family="attribute_missing_runtime",
        )
        for i in range(3)
    ]
    val = train[:10]
    holdout = train[:12]

    report = audit_phase2av_descriptor_pool_gap(
        jsonl_paths=[
            _write_jsonl(tmp_path / "train.jsonl", train),
            _write_jsonl(tmp_path / "val.jsonl", val),
            _write_jsonl(tmp_path / "holdout.jsonl", holdout),
        ],
        min_total_rows=64,
        min_rows_per_split=16,
        min_operations_per_split=3,
        min_examples_per_operation=5,
        min_repo_origins_per_split=3,
    )

    assert report["passed"] is False
    assert "total_rows_below_full_data_threshold" in report["blocking_reasons"]
    assert "train_operation_diversity_below_threshold" in report["blocking_reasons"]
    assert "train_repo_origin_below_threshold" in report["blocking_reasons"]
    assert "phase2av_full_training_ready" in report["unsupported_claims"]


def test_phase2av_descriptor_pool_gap_rejects_single_repo_operation_coverage(
    tmp_path: Path,
) -> None:
    paths = []
    for split in ("train", "val", "holdout"):
        rows = []
        for operation, template in [
            ("insert_import", "import_restoration"),
            ("replace_literal", "literal_restoration"),
        ]:
            for index in range(6):
                rows.append(
                    _row(
                        index,
                        split=split,
                        operation=operation,
                        template=template,
                        repo=f"repo-{index % 3}",
                    )
                )
        for index in range(6):
            rows.append(
                _row(
                    index,
                    split=split,
                    operation="replace_attribute",
                    template="call_attribute_restoration",
                    repo="single-attribute-repo",
                    family="attribute_missing_runtime",
                )
            )
        paths.append(_write_jsonl(tmp_path / f"{split}.jsonl", rows))

    report = audit_phase2av_descriptor_pool_gap(
        jsonl_paths=paths,
        min_total_rows=54,
        min_rows_per_split=18,
        min_operations_per_split=3,
        min_examples_per_operation=5,
        min_repo_origins_per_split=3,
        min_repo_origins_per_operation=2,
    )

    assert report["passed"] is False
    assert "train_operation_repo_origin_below_threshold" in report["blocking_reasons"]
    assert report["split_reports"]["train"]["operations_with_too_few_repo_origins"] == [
        "replace_attribute"
    ]
