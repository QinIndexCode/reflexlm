import json
from pathlib import Path

from reflexlm.cli.audit_phase2w_open_ended_repair import (
    audit_phase2w_open_ended_repair,
)


AXES = [
    "multi_file_patch",
    "ambiguous_traceback",
    "dependency_or_environment_issue",
    "stale_state_refresh",
    "hidden_side_effect_guard",
]


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _rows(count: int = 64, *, live_success: float = 0.70, family: str = "open_ended_repair") -> list[dict]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "task_id": f"task-{index}",
                "task_family": family,
                "repo_origin": f"public-repo-{index % 4}",
                "repo_commit": "a" * 40,
                "task_spec_sha256": "b" * 64,
                "difficulty_axes": [AXES[index % len(AXES)]],
                "full_task_success": True,
                "full_patch_correctness": True,
                "full_test_pass_rate": 1.0,
                "best_live_agent_task_success": live_success,
                "best_live_agent_patch_correctness": live_success,
                "rollback_success": True,
                "unauthorized_write_count": 0,
                "false_completion": False,
                "full_transcript_sha256": "c" * 64,
                "full_patch_diff_sha256": "d" * 64,
                "full_test_log_sha256": "e" * 64,
                "live_agent_transcript_sha256": "f" * 64,
                "live_agent_patch_diff_sha256": "0" * 64,
                "live_agent_test_log_sha256": "1" * 64,
            }
        )
    return rows


def test_phase2w_open_ended_repair_accepts_measured_open_tasks(tmp_path: Path) -> None:
    report = audit_phase2w_open_ended_repair(
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", _rows())
    )
    assert report["passed"] is True
    assert report["task_family"] == "open_ended_repair"
    assert report["metrics"]["full_minus_best_live_agent_task_success"] >= 0.10
    assert report["checks"]["row_provenance_hashes_present"] is True
    assert report["checks"]["row_provenance_hashes_valid"] is True
    assert report["checks"]["repo_commit_hashes_valid"] is True


def test_phase2w_open_ended_repair_rejects_bounded_family(tmp_path: Path) -> None:
    report = audit_phase2w_open_ended_repair(
        results_jsonl=_write_jsonl(
            tmp_path / "results.jsonl", _rows(family="bounded_command_selection")
        )
    )
    assert report["passed"] is False
    assert report["checks"]["task_family_open_ended"] is False


def test_phase2w_open_ended_repair_rejects_tie_with_live_agent(tmp_path: Path) -> None:
    report = audit_phase2w_open_ended_repair(
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", _rows(live_success=0.95))
    )
    assert report["passed"] is False
    assert report["checks"]["full_beats_live_agent_task_success"] is False


def test_phase2w_open_ended_repair_rejects_missing_provenance(tmp_path: Path) -> None:
    rows = _rows()
    del rows[0]["full_transcript_sha256"]
    report = audit_phase2w_open_ended_repair(
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", rows)
    )
    assert report["passed"] is False
    assert report["checks"]["row_schema_complete"] is False
    assert report["checks"]["row_provenance_hashes_present"] is False


def test_phase2w_open_ended_repair_rejects_invalid_hash_formats(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["task_spec_sha256"] = "not-a-hash"
    rows[1]["repo_commit"] = "not-a-commit"
    report = audit_phase2w_open_ended_repair(
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", rows)
    )
    assert report["passed"] is False
    assert report["checks"]["row_provenance_hashes_valid"] is False
    assert report["checks"]["repo_commit_hashes_valid"] is False
