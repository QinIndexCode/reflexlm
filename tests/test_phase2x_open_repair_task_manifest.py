import json
from pathlib import Path

from reflexlm.cli.audit_phase2x_open_repair_task_manifest import (
    REQUIRED_DIFFICULTY_AXES,
    audit_phase2x_open_repair_task_manifest,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _rows(count: int = 64, *, marker: str | None = None, family: str = "open_ended_repair") -> list[dict]:
    axes = sorted(REQUIRED_DIFFICULTY_AXES)
    rows = []
    for index in range(count):
        rows.append(
            {
                "task_id": f"phase2x-task-{index}",
                "split": "holdout" if index < 32 else "train",
                "task_family": family,
                "repo_origin": f"public-repo-{index % 4}",
                "repo_commit": "a" * 40,
                "task_spec_sha256": "b" * 64,
                "problem_statement": marker or f"Fix failing behavior {index}",
                "difficulty_axes": [axes[index % len(axes)]],
                "requires_patch": True,
                "evaluation_command": "python -m pytest",
                "rollback_command": "git checkout -- .",
                "allowed_write_scope": "workspace",
                "baseline_budget": {
                    "max_commands": 30,
                    "max_wall_clock_seconds": 1800,
                },
            }
        )
    return rows


def test_phase2x_open_repair_task_manifest_accepts_preregistered_open_tasks(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_task_manifest(
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", _rows())
    )
    assert report["passed"] is True
    assert report["checks"]["difficulty_axes_covered"] is True
    assert report["checks"]["task_hashes_valid"] is True
    assert report["metrics"]["repo_origin_count"] == 4


def test_phase2x_open_repair_task_manifest_rejects_candidate_markers(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_task_manifest(
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", _rows(marker="candidate_0 should win"))
    )
    assert report["passed"] is False
    assert report["checks"]["forbidden_markers_absent"] is False
    assert "candidate_0" in report["metrics"]["forbidden_markers_found"]


def test_phase2x_open_repair_task_manifest_rejects_bounded_family(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_task_manifest(
        tasks_jsonl=_write_jsonl(
            tmp_path / "tasks.jsonl",
            _rows(family="bounded_command_selection"),
        )
    )
    assert report["passed"] is False
    assert report["checks"]["task_family_open_ended"] is False


def test_phase2x_open_repair_task_manifest_rejects_invalid_hash_formats(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["task_spec_sha256"] = "not-a-hash"
    rows[1]["repo_commit"] = "not-a-commit"
    report = audit_phase2x_open_repair_task_manifest(
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", rows)
    )
    assert report["passed"] is False
    assert report["checks"]["task_hashes_valid"] is False
