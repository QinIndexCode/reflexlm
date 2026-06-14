import json
from pathlib import Path

from reflexlm.cli.build_phase2ag_verifiable_candidate_sidecar_split import (
    CLAIM_BOUNDARY,
    build_phase2ag_verifiable_candidate_sidecar_split,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(repo: str, index: int, *, expected_slot: int = 1, good: bool = True) -> dict:
    expected_probe = f"probe-{repo}-{index}"
    wrong_probe = f"wrong-{repo}-{index}"
    candidates = [
        {
            "repair_action": f"wrong-{index}",
            "edit_scope": "pkg/a.py",
            "structural_probe_hash": wrong_probe,
        },
        {
            "repair_action": f"correct-{index}",
            "edit_scope": "pkg/b.py",
            "structural_probe_hash": expected_probe,
        },
    ]
    if expected_slot == 0:
        candidates = list(reversed(candidates))
    return {
        "trace_id": f"{repo}:{index}",
        "repo_id": repo,
        "current_visible_text": "public repair candidate verification row",
        "runtime_visible_evidence": {
            "structural_probe_hashes": [expected_probe] if good else [],
        },
        "repair_candidates": candidates,
        "expected_repair_action": f"correct-{index}",
    }


def test_phase2ag_split_builder_filters_and_records_unique_probe_rows(
    tmp_path: Path,
) -> None:
    train = _write_jsonl(
        tmp_path / "train.jsonl",
        [_row("train_repo", 0, expected_slot=0)]
        + [_row("train_repo", index) for index in range(1, 3)]
        + [_row("train_repo", 99, good=False)],
    )
    val = _write_jsonl(tmp_path / "val.jsonl", [_row("val_repo", index) for index in range(2)])
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        [_row("holdout_repo", index) for index in range(2)],
    )

    manifest = build_phase2ag_verifiable_candidate_sidecar_split(
        train_jsonl=[train],
        val_jsonl=[val],
        holdout_jsonl=[holdout],
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_train_rows=3,
        min_val_rows=2,
        min_holdout_rows=2,
    )

    assert manifest["passed"] is True
    assert manifest["claim_boundary"] == CLAIM_BOUNDARY
    assert manifest["split_counts"] == {"train": 3, "val": 2, "holdout": 2}
    assert manifest["filter_reports"]["train"]["rejected"] == 1
    assert manifest["checks"]["repo_disjoint"] is True
    assert manifest["checks"]["train_covers_val_and_holdout_slots"] is True
    built_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(row["benchmark_family"] == "phase2ag_verifiable_candidate_sidecar" for row in built_rows)
    assert all(
        row["phase2ag_probe_audit"]["probe_prediction"]
        == row["phase2ag_probe_audit"]["expected_slot"]
        for row in built_rows
    )


def test_phase2ag_split_builder_rejects_low_scale_or_repo_overlap(
    tmp_path: Path,
) -> None:
    train = _write_jsonl(tmp_path / "train.jsonl", [_row("shared_repo", 0)])
    val = _write_jsonl(tmp_path / "val.jsonl", [_row("shared_repo", 1)])
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", [_row("holdout_repo", 0)])

    manifest = build_phase2ag_verifiable_candidate_sidecar_split(
        train_jsonl=[train],
        val_jsonl=[val],
        holdout_jsonl=[holdout],
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_train_rows=2,
        min_val_rows=1,
        min_holdout_rows=1,
    )

    assert manifest["passed"] is False
    assert manifest["checks"]["min_train_rows"] is False
    assert manifest["checks"]["repo_disjoint"] is False
    assert "do_not_train_claim_bearing_phase2ag_adapter" in manifest["blocked_actions"]


def test_phase2ag_split_builder_rejects_eval_slots_missing_from_train(
    tmp_path: Path,
) -> None:
    train = _write_jsonl(tmp_path / "train.jsonl", [_row("train_repo", 0, expected_slot=0)])
    val = _write_jsonl(tmp_path / "val.jsonl", [_row("val_repo", 1, expected_slot=1)])
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", [_row("holdout_repo", 2, expected_slot=1)])

    manifest = build_phase2ag_verifiable_candidate_sidecar_split(
        train_jsonl=[train],
        val_jsonl=[val],
        holdout_jsonl=[holdout],
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_train_rows=1,
        min_val_rows=1,
        min_holdout_rows=1,
    )

    assert manifest["passed"] is False
    assert manifest["checks"]["train_covers_val_and_holdout_slots"] is False


def test_phase2ag_split_builder_rejects_split_suffix_repo_overlap(
    tmp_path: Path,
) -> None:
    train = _write_jsonl(tmp_path / "train.jsonl", [_row("same_repo_train", 0)])
    val = _write_jsonl(tmp_path / "val.jsonl", [_row("same_repo_val", 1)])
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", [_row("other_repo_holdout", 0)])

    manifest = build_phase2ag_verifiable_candidate_sidecar_split(
        train_jsonl=[train],
        val_jsonl=[val],
        holdout_jsonl=[holdout],
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_train_rows=1,
        min_val_rows=1,
        min_holdout_rows=1,
    )

    assert manifest["passed"] is False
    assert manifest["checks"]["repo_disjoint"] is False
    assert manifest["repo_overlaps"] == {"train__val": ["same_repo"]}
