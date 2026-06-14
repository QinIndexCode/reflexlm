import json
from pathlib import Path

from reflexlm.cli.build_phase2aw_split_clean_candidate_pool import (
    build_phase2aw_split_clean_candidate_pool,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(split: str, repo: str = "psf_black") -> dict:
    return {
        "trace_id": f"{split}:{repo}:0",
        "split": split,
        "repo_url_or_origin": f"https://github.com/{repo.replace('_', '/')}.git",
        "artifact_paths": {
            "generated_test": f"artifacts/{split}/{repo}/row_00000/generated_test.py"
        },
        "source_kind": "public_repo",
    }


def _source_artifacts(root: Path, split: str, repo: str = "psf_black") -> None:
    artifact_dir = root / "artifacts" / split / repo / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "generated_test.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (artifact_dir / "patch.diff").write_text("diff --git a/x b/x\n", encoding="utf-8")


def test_phase2aw_split_clean_candidate_pool_rewrites_and_copies_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    for split in ("train", "val", "holdout"):
        _source_artifacts(source, split)
    train = _write_jsonl(tmp_path / "train.jsonl", [_row("train")])
    val = _write_jsonl(tmp_path / "val.jsonl", [_row("val")])
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", [_row("holdout")])

    report = build_phase2aw_split_clean_candidate_pool(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        source_dataset_root=source,
        output_jsonl_dir=tmp_path / "out_jsonl",
        output_dataset_root=tmp_path / "out_data",
        manifest_json=tmp_path / "manifest.json",
    )

    assert report["passed"] is True
    rewritten = json.loads((tmp_path / "out_jsonl" / "holdout.jsonl").read_text())
    assert rewritten["artifact_paths"]["generated_test"].startswith(
        "artifacts/holdout/"
    )
    assert rewritten["artifact_paths"]["patch_diff"].endswith("patch.diff")
    assert (
        tmp_path / "out_data" / rewritten["artifact_paths"]["generated_test"]
    ).exists()
    assert (tmp_path / "out_data" / rewritten["artifact_paths"]["patch_diff"]).exists()
    assert rewritten["phase2aw_split_clean_artifact_rewrite"]["enabled"] is True


def test_phase2aw_split_clean_candidate_pool_rejects_missing_patch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    artifact_dir = source / "artifacts" / "train" / "psf_black" / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "generated_test.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    train = _write_jsonl(tmp_path / "train.jsonl", [_row("train")])

    report = build_phase2aw_split_clean_candidate_pool(
        train_jsonl=train,
        val_jsonl=train,
        holdout_jsonl=train,
        source_dataset_root=source,
        output_jsonl_dir=tmp_path / "out_jsonl",
        output_dataset_root=tmp_path / "out_data",
        manifest_json=tmp_path / "manifest.json",
    )

    assert report["passed"] is False
    assert report["reject_counts"]["patch_diff_copy_failed"] == 3

