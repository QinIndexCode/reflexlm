import json
from pathlib import Path

from reflexlm.cli.audit_phase2ar_diverse_symbolic_patch_benchmark import (
    audit_phase2ar_diverse_symbolic_patch_benchmark,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(root: Path, split: str, index: int, repo_id: str, kind: str) -> dict:
    artifact_dir = root / "artifacts" / split / repo_id / f"row_{index:05d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if kind == "text_membership":
        test_source = (
            "from pathlib import Path\n"
            "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
            "def test_line():\n"
            "    text = (REPO_ROOT / 'a.py').read_text(encoding='utf-8')\n"
            "    assert 'import os' in text\n"
        )
    else:
        test_source = (
            "import ast\n"
            "from pathlib import Path\n"
            "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
            "def test_attr():\n"
            "    tree = ast.parse((REPO_ROOT / 'a.py').read_text(encoding='utf-8'))\n"
            "    assert any(isinstance(node, ast.Attribute) and node.attr == 'lower' for node in ast.walk(tree))\n"
        )
    (artifact_dir / "generated_test.py").write_text(test_source, encoding="utf-8")
    (artifact_dir / "patch.diff").write_text("diff --git a/a.py b/a.py\n", encoding="utf-8")
    return {
        "trace_id": f"{split}:{repo_id}:{index}",
        "repo_id": repo_id,
        "normalization": {"sealed_feedback_absent": True},
        "artifact_paths": {
            "generated_test": f"artifacts/{split}/{repo_id}/row_{index:05d}/generated_test.py",
            "patch_diff": f"artifacts/{split}/{repo_id}/row_{index:05d}/patch.diff",
        },
    }


def _dataset(root: Path) -> Path:
    _write_json(
        root / "manifest.json",
        {"target_selection_policy": "stratified_repair_mode"},
    )
    for split, repo_prefix in (
        ("train", "repo_train"),
        ("val", "repo_val"),
        ("holdout", "repo_holdout"),
    ):
        rows = [
            _row(root, split, i, f"{repo_prefix}_{i}", "text_membership" if i % 2 == 0 else "ast_attribute_restoration")
            for i in range(8)
        ]
        _write_jsonl(root / f"{split}.raw.jsonl", rows)
    return root


def test_phase2ar_audit_accepts_diverse_repo_disjoint_benchmark(tmp_path: Path) -> None:
    report = audit_phase2ar_diverse_symbolic_patch_benchmark(
        dataset_root=_dataset(tmp_path / "dataset"),
        min_rows_per_split=8,
    )

    assert report["passed"] is True
    assert report["checks"]["repo_origin_disjoint"] is True
    assert report["checks"]["holdout_required_patch_kinds_present"] is True


def test_phase2ar_audit_rejects_single_kind_holdout(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset")
    rows = [_row(root, "holdout", i, f"repo_holdout_{i}", "text_membership") for i in range(8)]
    _write_jsonl(root / "holdout.raw.jsonl", rows)

    report = audit_phase2ar_diverse_symbolic_patch_benchmark(
        dataset_root=root,
        min_rows_per_split=8,
    )

    assert report["passed"] is False
    assert report["checks"]["holdout_required_patch_kinds_present"] is False
