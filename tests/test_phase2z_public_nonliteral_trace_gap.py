import json
from pathlib import Path

from reflexlm.cli.audit_phase2z_public_nonliteral_trace_gap import (
    audit_phase2z_public_nonliteral_trace_gap,
    classify_patch_text,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_phase2z_public_gap_classifier_treats_literal_flip_as_not_structural() -> None:
    patch = (
        "--- a/src/mod.py\n"
        "+++ b/src/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 0\n"
    )

    stats = classify_patch_text(patch)

    assert stats["single_line_literal_like"] is True
    assert stats["structural_nonliteral_candidate"] is False


def test_phase2z_public_gap_classifier_accepts_multifile_patch() -> None:
    patch = (
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 'old'\n"
        "+VALUE = 'new'\n"
        "--- a/src/b.py\n"
        "+++ b/src/b.py\n"
        "@@ -1 +1 @@\n"
        "-def f(): return 'old'\n"
        "+def f(): return 'new'\n"
    )

    stats = classify_patch_text(patch)

    assert stats["multi_file"] is True
    assert stats["structural_nonliteral_candidate"] is True


def test_phase2z_public_nonliteral_gap_blocks_literal_only_public_split(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    patch_dir = root / "artifacts" / "r0"
    patch_dir.mkdir(parents=True)
    (patch_dir / "patch.diff").write_text(
        "--- a/src/mod.py\n"
        "+++ b/src/mod.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = True\n"
        "+VALUE = False\n",
        encoding="utf-8",
    )
    rows = [
        {
            "trace_id": "r0",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": "repo0",
            "artifact_paths": {"patch_diff": "artifacts/r0/patch.diff"},
        }
    ]
    split = _write_jsonl(tmp_path / "holdout.raw.jsonl", rows)

    report = audit_phase2z_public_nonliteral_trace_gap(
        dataset_root=root,
        split_jsonl=[split],
        min_rows=1,
        min_structural_nonliteral_rows=1,
        min_multifile_rows=1,
    )

    assert report["passed"] is False
    assert report["checks"]["multifile_minimum_met"] is False
    assert "do_not_claim_public_nonliteral_patch_generation" in report["blocked_actions"]
