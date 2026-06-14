import json
from pathlib import Path

import pytest

from reflexlm.cli.build_phase2av_accepted_descriptor_subset import (
    build_phase2av_accepted_descriptor_subset,
)


HASH = "a" * 64


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows),
        encoding="utf-8",
    )
    return path


def _manifest(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "manifest.json",
        {
            "artifact_family": "phase2at_learned_patch_candidate_split",
            "schema_version": "phase2at.learned_bounded_patch_candidate.v1",
            "split_counts": {"train": 2, "val": 1, "holdout": 1},
            "split_hashes": {"train": HASH, "val": HASH, "holdout": HASH},
            "freeform_patch_generation": False,
            "recorded_patch_artifact_as_generation_target": False,
            "symbolic_generator_as_generation_target": False,
            "sealed_feedback_used": False,
        },
    )


def _descriptor(path: Path, split: str, trace_ids: list[str]) -> Path:
    rows = [
        {
            "trace_id": trace_id,
            "split": split,
            "learned_patch_candidate_target": {"operation": "replace_literal"},
        }
        for trace_id in trace_ids
    ]
    return _write_jsonl(path, rows)


def _runtime(path: Path, split: str, source_ids: list[str]) -> Path:
    rows = [
        {
            "task_id": f"phase2av:{split}:{index:05d}",
            "source": {"source_task_id": source_id},
        }
        for index, source_id in enumerate(source_ids)
    ]
    return _write_jsonl(path, rows)


def test_builds_descriptor_subset_from_runtime_accepted_ids(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    train_desc = _descriptor(tmp_path / "phase2at" / "train.jsonl", "train", ["a", "b"])
    val_desc = _descriptor(tmp_path / "phase2at" / "val.jsonl", "val", ["c"])
    holdout_desc = _descriptor(tmp_path / "phase2at" / "holdout.jsonl", "holdout", ["d"])
    train_runtime = _runtime(tmp_path / "runtime" / "train.jsonl", "train", ["b"])
    val_runtime = _runtime(tmp_path / "runtime" / "val.jsonl", "val", ["c"])
    holdout_runtime = _runtime(tmp_path / "runtime" / "holdout.jsonl", "holdout", ["d"])

    report = build_phase2av_accepted_descriptor_subset(
        descriptor_manifest_json=manifest,
        train_descriptor_jsonl=train_desc,
        val_descriptor_jsonl=val_desc,
        holdout_descriptor_jsonl=holdout_desc,
        train_runtime_tasks_jsonl=train_runtime,
        val_runtime_tasks_jsonl=val_runtime,
        holdout_runtime_tasks_jsonl=holdout_runtime,
        output_dir=tmp_path / "accepted_phase2at",
        manifest_json=tmp_path / "accepted_manifest.json",
    )

    assert report["split_counts"] == {"train": 1, "val": 1, "holdout": 1}
    assert report["accepted_subset"]["train"]["rejected_trace_ids"] == ["a"]
    assert report["sealed_feedback_used"] is False
    rows = [
        json.loads(line)
        for line in (tmp_path / "accepted_phase2at" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["trace_id"] for row in rows] == ["b"]


def test_rejects_runtime_id_missing_from_descriptor(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    train_desc = _descriptor(tmp_path / "phase2at" / "train.jsonl", "train", ["a"])
    val_desc = _descriptor(tmp_path / "phase2at" / "val.jsonl", "val", ["c"])
    holdout_desc = _descriptor(tmp_path / "phase2at" / "holdout.jsonl", "holdout", ["d"])
    train_runtime = _runtime(tmp_path / "runtime" / "train.jsonl", "train", ["missing"])
    val_runtime = _runtime(tmp_path / "runtime" / "val.jsonl", "val", ["c"])
    holdout_runtime = _runtime(tmp_path / "runtime" / "holdout.jsonl", "holdout", ["d"])

    with pytest.raises(ValueError, match="missing from descriptor"):
        build_phase2av_accepted_descriptor_subset(
            descriptor_manifest_json=manifest,
            train_descriptor_jsonl=train_desc,
            val_descriptor_jsonl=val_desc,
            holdout_descriptor_jsonl=holdout_desc,
            train_runtime_tasks_jsonl=train_runtime,
            val_runtime_tasks_jsonl=val_runtime,
            holdout_runtime_tasks_jsonl=holdout_runtime,
            output_dir=tmp_path / "accepted_phase2at",
            manifest_json=tmp_path / "accepted_manifest.json",
        )
