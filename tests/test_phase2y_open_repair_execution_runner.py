import json
from pathlib import Path

from reflexlm.cli.run_phase2y_open_repair_execution_results import (
    run_phase2y_open_repair_execution_results,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_phase2y_runner_does_not_fake_nonliteral_success(tmp_path: Path) -> None:
    clone_root = tmp_path / "repos"
    repo = clone_root / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "mod.py").write_text("VALUE = True\n", encoding="utf-8")
    tasks = [
        {
            "task_id": "phase2y:val:00000",
            "repair_mode": "nonliteral_symbolic_patch",
            "repo_origin": "https://github.com/example/repo.git",
            "repo_commit": "a" * 40,
            "task_spec_sha256": "b" * 64,
            "source": {"source_trace_id": "trace-0"},
            "evaluation_commands": [
                "python -m pytest -q phase2s_repair_tests/test_case.py --maxfail=1"
            ],
            "allowed_write_scope": "src/mod.py",
        }
    ]
    traces = [
        {
            "trace_id": "trace-0",
            "repo_id": "repo",
            "runtime_visible_evidence": {
                "target_location": {"path": "src/mod.py", "line": 1, "col": 8},
                "changed_files": ["src/mod.py"],
                "pytest_before_patch": {
                    "stdout_excerpt": "E       assert True == False\n",
                    "exit_code": 1,
                },
            },
            "expected_repair_result": {
                "test_target": "phase2s_repair_tests/test_case.py"
            },
        }
    ]
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {"policy_label": "test-policy"},
    )

    report = run_phase2y_open_repair_execution_results(
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", tasks),
        source_traces_jsonl=_write_jsonl(tmp_path / "traces.jsonl", traces),
        package_path=package,
        clone_root=clone_root,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
    )

    assert report["rows"] == 1
    assert report["successes"] == 0
    row = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert row["repair_mode"] == "nonliteral_symbolic_patch"
    assert row["success"] is False
    assert row["unsupported_reason"] == "source_trace_is_literal_only"


def test_phase2y_runner_generates_bounded_nonliteral_membership_patch(tmp_path: Path) -> None:
    clone_root = tmp_path / "repos"
    repo = clone_root / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    tasks = [
        {
            "task_id": "phase2y:val:00000",
            "repair_mode": "nonliteral_symbolic_patch",
            "repo_origin": "https://github.com/example/repo.git",
            "repo_commit": "a" * 40,
            "task_spec_sha256": "b" * 64,
            "source": {"source_trace_id": "trace-0"},
            "evaluation_commands": [
                "python -m pytest -q phase2s_repair_tests/test_case.py --maxfail=1"
            ],
            "allowed_write_scope": "src/mod.py",
        }
    ]
    traces = [
        {
            "trace_id": "trace-0",
            "repo_id": "repo",
            "runtime_visible_evidence": {
                "target_location": {"path": "src/mod.py"},
                "changed_files": ["src/mod.py"],
                "pytest_before_patch": {
                    "stdout_excerpt": "E       assert 'REQUIRED_TOKEN = True' in text\n",
                    "exit_code": 1,
                },
            },
            "expected_repair_result": {
                "test_target": "phase2s_repair_tests/test_case.py"
            },
        }
    ]
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "manifest-open-repair-policy",
            "open_repair_capabilities": {
                "patch_proposal_head": True,
                "bounded_edit_scope_policy": True,
                "rollback_safety_head": True,
                "test_selection_head": True,
            },
        },
    )

    report = run_phase2y_open_repair_execution_results(
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", tasks),
        source_traces_jsonl=_write_jsonl(tmp_path / "traces.jsonl", traces),
        package_path=package,
        clone_root=clone_root,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
    )

    assert report["rows"] == 1
    assert report["successes"] == 1
    row = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert row["repair_mode"] == "nonliteral_symbolic_patch"
    assert row["success"] is True
    assert row["patch_generator"] == "bounded_symbolic_text_membership_patch_v1"
    assert row["patch_source"] == "package_runtime_patch_proposal"
    assert row["unsupported_reason"] is None
