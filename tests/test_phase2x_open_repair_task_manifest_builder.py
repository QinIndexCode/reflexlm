import json
from pathlib import Path

from reflexlm.cli.audit_phase2x_open_repair_task_manifest import (
    audit_phase2x_open_repair_task_manifest,
)
from reflexlm.cli.build_phase2x_open_repair_task_manifest import (
    build_phase2x_open_repair_task_manifest,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _source_row(index: int) -> dict:
    families = [
        "dependency_or_import_mismatch",
        "assertion_or_literal_mismatch",
        "state_or_cache_mismatch",
        "multi_file_patch",
        "hidden_side_effect_guard",
    ]
    return {
        "trace_id": f"phase2t:trace:{index}",
        "repo_id": f"repo-{index % 4}",
        "repo_url_or_origin": f"https://github.com/example/repo-{index % 4}.git",
        "commit_hash": "a" * 40,
        "trace_construction_mode": "phase2t_dynamic_public_repo_repair_loop_trace",
        "difficulty": {
            "task_family": families[index % len(families)],
            "repair_depth": "multi_edit" if index % 5 == 3 else "one_edit",
        },
        "runtime_visible_evidence": {
            "changed_files": [f"src/module_{index}.py"],
            "failing_test_target": f"tests/test_{index}.py",
            "watched_files": [f"tests/test_{index}.py"],
        },
        "artifact_paths": {"test_output": f"artifacts/{index}/test_output.json"},
        "repair_candidates": [
            {
                "edit_scope": f"src/module_{index}.py",
                "verification_command": f"python -m pytest -q tests/test_{index}.py --maxfail=1",
            }
        ],
    }


def test_phase2x_builder_outputs_auditable_open_repair_manifest(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "source.jsonl", [_source_row(index) for index in range(64)])
    output = tmp_path / "tasks.jsonl"
    report = build_phase2x_open_repair_task_manifest(
        input_jsonl=source,
        output_jsonl=output,
        split="holdout",
    )
    assert report["passed"] is True
    assert report["claim_boundary"] == "task_manifest_only_not_open_repair_result"

    audit = audit_phase2x_open_repair_task_manifest(
        tasks_jsonl=output,
        min_rows=64,
        min_holdout_rows=64,
        min_repo_origins=4,
    )
    assert audit["passed"] is True
