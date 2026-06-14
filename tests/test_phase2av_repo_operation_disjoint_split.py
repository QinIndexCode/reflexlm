import json
from pathlib import Path

from reflexlm.cli.build_phase2av_repo_operation_disjoint_split import (
    build_phase2av_repo_operation_disjoint_split,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(repo: str, operation: str, index: int) -> dict:
    return {
        "trace_id": f"{repo}:{operation}:{index}",
        "repo_url_or_origin": repo,
        "split": "source_pool",
        "learned_patch_candidate_target": {
            "operation": operation,
            "after_fragment_template_id": {
                "replace_attribute": "call_attribute_restoration",
                "insert_import": "import_restoration",
                "replace_literal": "literal_restoration",
            }[operation],
        },
        "runtime_visible_evidence": {
            "descriptor_failure_family": {
                "replace_attribute": "attribute_missing_runtime",
                "insert_import": "missing_import_or_symbol_runtime",
                "replace_literal": "assertion_behavior_mismatch_runtime",
            }[operation],
        },
    }


def test_phase2av_repo_operation_disjoint_split_balances_operation_repo_coverage(
    tmp_path: Path,
) -> None:
    rows = []
    for repo_index in range(9):
        repo = f"repo-{repo_index}"
        for operation in ("replace_attribute", "insert_import", "replace_literal"):
            for index in range(2):
                rows.append(_row(repo, operation, index))
    source = _write_jsonl(tmp_path / "pool.jsonl", rows)

    manifest = build_phase2av_repo_operation_disjoint_split(
        input_jsonl=[source],
        output_dir=tmp_path / "split",
        manifest_json=tmp_path / "manifest.json",
        min_rows_per_split=12,
        min_examples_per_operation=4,
        min_repo_origins_per_split=3,
        min_repo_origins_per_operation=2,
    )

    assert manifest["passed"] is True
    assert manifest["smoke_training_allowed"] is True
    assert manifest["full_training_allowed"] is False
    assert manifest["package_allowed"] is False
    assert manifest["sealed_eval_allowed"] is False
    reports = manifest["data_health"]["split_reports"]
    for split in ("train", "val", "holdout"):
        assert reports[split]["operation_repo_origin_ready"] is True
        assert reports[split]["repo_origin_ready"] is True

    repo_to_split = {}
    for split in ("train", "val", "holdout"):
        for row in (tmp_path / "split" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            payload = json.loads(row)
            repo = payload["repo_url_or_origin"]
            assert repo not in repo_to_split or repo_to_split[repo] == split
            repo_to_split[repo] = split
