import json
from pathlib import Path

from reflexlm.cli.audit_phase2az_model_predicted_execution_matrix import (
    audit_phase2az_model_predicted_execution_matrix,
)
from reflexlm.cli.build_phase2az_model_execution_subset import (
    build_phase2az_model_execution_subset,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _head_row(repo: str, pair: int, member: str, slot: int) -> dict:
    return {
        "episode_id": f"phase2ax:pair_{pair:05d}:{member}",
        "command_slot": slot,
        "source_task_manifest": {
            "pair_id": f"phase2ax_pair_{pair:05d}",
            "repo_origin": f"https://github.com/example/{repo}.git",
        },
    }


def _task_row(pair: int, member: str) -> dict:
    return {"task_id": f"phase2ax:pair_{pair:05d}:{member}"}


def test_phase2az_subset_builder_selects_complete_pairs_across_repos(tmp_path: Path) -> None:
    head_rows = [
        _head_row("repo-a", 0, "a", 0),
        _head_row("repo-a", 0, "b", 1),
        _head_row("repo-b", 1, "a", 0),
        _head_row("repo-b", 1, "b", 1),
        _head_row("repo-c", 2, "a", 0),
        _head_row("repo-c", 2, "b", 1),
    ]
    task_rows = [_task_row(pair, member) for pair in range(3) for member in ("a", "b")]

    report = build_phase2az_model_execution_subset(
        head_jsonl=_write_jsonl(tmp_path / "head.jsonl", head_rows),
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", task_rows),
        output_head_jsonl=tmp_path / "subset_head.jsonl",
        output_tasks_jsonl=tmp_path / "subset_tasks.jsonl",
        report_json=tmp_path / "report.json",
        max_repos=3,
        min_repos=3,
    )

    assert report["passed"] is True
    assert report["repo_count"] == 3
    assert report["head_rows"] == 6
    assert report["task_rows"] == 6
    assert report["slot_counts"] == {"0": 3, "1": 3}


def test_phase2az_subset_builder_can_require_present_public_clones(tmp_path: Path) -> None:
    head_rows = [
        _head_row("repo-a", 0, "a", 0),
        _head_row("repo-a", 0, "b", 1),
        _head_row("repo-b", 1, "a", 0),
        _head_row("repo-b", 1, "b", 1),
        _head_row("repo-c", 2, "a", 0),
        _head_row("repo-c", 2, "b", 1),
    ]
    task_rows = [_task_row(pair, member) for pair in range(3) for member in ("a", "b")]
    clone_root = tmp_path / "clones"
    (clone_root / "example_repo_a").mkdir(parents=True)
    (clone_root / "example_repo_c").mkdir(parents=True)

    report = build_phase2az_model_execution_subset(
        head_jsonl=_write_jsonl(tmp_path / "head.jsonl", head_rows),
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", task_rows),
        output_head_jsonl=tmp_path / "subset_head.jsonl",
        output_tasks_jsonl=tmp_path / "subset_tasks.jsonl",
        report_json=tmp_path / "report.json",
        max_repos=3,
        min_repos=2,
        clone_root=clone_root,
        require_clone_present=True,
    )

    assert report["passed"] is True
    assert report["repo_count"] == 2
    assert report["head_rows"] == 4
    assert report["slot_counts"] == {"0": 2, "1": 2}
    assert report["clone_filtered_repos"] == [
        "https://github.com/example/repo-b.git"
    ]


def test_phase2az_subset_builder_rejects_clone_requirement_without_clone_root(
    tmp_path: Path,
) -> None:
    head_rows = [
        _head_row("repo-a", 0, "a", 0),
        _head_row("repo-a", 0, "b", 1),
    ]
    task_rows = [_task_row(0, member) for member in ("a", "b")]

    report = build_phase2az_model_execution_subset(
        head_jsonl=_write_jsonl(tmp_path / "head.jsonl", head_rows),
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", task_rows),
        output_head_jsonl=tmp_path / "subset_head.jsonl",
        output_tasks_jsonl=tmp_path / "subset_tasks.jsonl",
        report_json=tmp_path / "report.json",
        max_repos=1,
        min_repos=1,
        require_clone_present=True,
    )

    assert report["passed"] is False
    assert report["checks"]["clone_requirement_satisfied"] is False
    assert report["repo_count"] == 0


def _subset_report() -> dict:
    return {
        "passed": True,
        "repo_count": 3,
        "head_rows": 6,
        "task_rows": 6,
        "repos": ["a", "b", "c"],
        "slot_counts": {"0": 3, "1": 3},
    }


def _model_eval(*, model_accuracy: float = 1.0, source_accuracy: float = 0.5) -> dict:
    return {
        "eval_examples": 6,
        "eval_metrics": {
            "command_slot_accuracy": model_accuracy,
            "command_slot_count": 6,
        },
        "source_overlap_command_slot_baseline": {
            "phase2az": {"accuracy": source_accuracy}
        },
    }


def _execution(*, success_rate: float = 1.0) -> dict:
    return {
        "selection_policy": "model_prediction_records",
        "rows": 6,
        "success_rate": success_rate,
        "execution_attempts": 6,
        "model_prediction_records_present_rows": 6,
        "recorded_patch_artifact_used_rows": 0,
        "recorded_patch_artifact_used_for_fault_injection_rows": 6,
        "freeform_patch_generation_rows": 0,
        "sealed_feedback_used_rows": 0,
    }


def _wrong_cache() -> dict:
    return {
        "selection_policy": "wrong_cache",
        "rows": 6,
        "success_rate": 0.0,
        "execution_attempts": 0,
    }


def _phase2ay_audit() -> dict:
    return {
        "passed": True,
        "ready_for_phase2ay_model_prediction_execution_eval": True,
    }


def test_phase2az_audit_accepts_repo_diverse_model_execution_but_blocks_epoch(
    tmp_path: Path,
) -> None:
    report = audit_phase2az_model_predicted_execution_matrix(
        subset_report_json=_write_json(tmp_path / "subset.json", _subset_report()),
        model_eval_json=_write_json(tmp_path / "eval.json", _model_eval()),
        model_execution_summary_json=_write_json(tmp_path / "execution.json", _execution()),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _wrong_cache()),
        phase2ay_model_audit_json=_write_json(tmp_path / "phase2ay.json", _phase2ay_audit()),
        eval_split="phase2az",
    )

    assert report["passed"] is True
    assert report["ready_for_phase2az_package_gate"] is True
    assert report["ready_for_phase2ax_package"] is False
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert report["metrics"]["model_minus_source_overlap"] == 0.5


def test_phase2az_audit_rejects_source_overlap_tie(tmp_path: Path) -> None:
    report = audit_phase2az_model_predicted_execution_matrix(
        subset_report_json=_write_json(tmp_path / "subset.json", _subset_report()),
        model_eval_json=_write_json(
            tmp_path / "eval.json", _model_eval(model_accuracy=0.9, source_accuracy=0.9)
        ),
        model_execution_summary_json=_write_json(tmp_path / "execution.json", _execution()),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _wrong_cache()),
        phase2ay_model_audit_json=_write_json(tmp_path / "phase2ay.json", _phase2ay_audit()),
        eval_split="phase2az",
    )

    assert report["passed"] is False
    assert report["checks"]["model_beats_source_overlap"] is False


def test_phase2az_audit_rejects_missing_repo_diversity(tmp_path: Path) -> None:
    subset = _subset_report()
    subset["repo_count"] = 1
    report = audit_phase2az_model_predicted_execution_matrix(
        subset_report_json=_write_json(tmp_path / "subset.json", subset),
        model_eval_json=_write_json(tmp_path / "eval.json", _model_eval()),
        model_execution_summary_json=_write_json(tmp_path / "execution.json", _execution()),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _wrong_cache()),
        phase2ay_model_audit_json=_write_json(tmp_path / "phase2ay.json", _phase2ay_audit()),
        eval_split="phase2az",
    )

    assert report["passed"] is False
    assert report["checks"]["min_repos_met"] is False
