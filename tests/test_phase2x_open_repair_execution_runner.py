import json
from pathlib import Path

from reflexlm.cli.audit_phase2x_open_repair_execution_results import (
    audit_phase2x_open_repair_execution_results,
)
from reflexlm.cli import run_phase2x_open_repair_execution_results as runner


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_parse_assertion_literals_from_visible_pytest_output() -> None:
    assert runner.parse_assertion_literals(">       assert 2 == 1") == (2, 1)
    assert runner.parse_assertion_literals("E       assert 'bad' == 'good'") == ("bad", "good")


def test_phase2x_execution_runner_emits_non_oracle_result_row(tmp_path: Path, monkeypatch) -> None:
    class FakePackage:
        def __init__(self, _package_path: Path) -> None:
            self.last_call = {}

        def act(self, _state) -> None:
            self.last_call = {
                "open_repair_head_outputs": {
                    "patch_proposal": 1,
                    "bounded_edit_scope": 1,
                    "rollback_safety": 1,
                    "stop_condition": 0,
                    "progress_monitor": 1,
                    "verification_state": 1,
                }
            }

    monkeypatch.setattr(runner, "NativeNervousPolicyPackage", FakePackage)
    repo = tmp_path / "clones" / "example_repo"
    repo.mkdir(parents=True)
    (repo / "pkg.py").write_text(
        "\n".join(
            [
                "def answer():",
                "    return 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    package = tmp_path / "package"
    _write_json(
        package / "native_nervous_package.json",
        {
            "policy_label": "phase2x_test_package",
            "package_family": "phase2d_native_nervous_package",
        },
    )
    trace = {
        "trace_id": "val:example_repo:abc:phase2t:0",
        "repo_id": "example_repo",
        "runtime_visible_evidence": {
            "pytest_before_patch": {
                "stdout_excerpt": ">       assert 2 == 1",
            },
            "target_location": {"path": "pkg.py", "line": 2, "col": 11},
            "changed_files": ["pkg.py"],
        },
        "expected_repair_result": {
            "test_target": "phase2s_repair_tests/test_case.py",
        },
    }
    task = {
        "task_id": "phase2x:val:00000",
        "repo_origin": "https://github.com/example/repo",
        "repo_commit": "a" * 40,
        "source_trace_id": trace["trace_id"],
        "problem_statement": "Repair literal failure",
        "allowed_write_scope": "pkg.py",
        "evaluation_command": "python -m pytest -q phase2s_repair_tests/test_case.py --maxfail=1",
    }
    output_jsonl = tmp_path / "results.jsonl"
    report = runner.run_phase2x_open_repair_execution_results(
        task_manifest_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", [task]),
        source_traces_jsonl=_write_jsonl(tmp_path / "traces.jsonl", [trace]),
        package_path=package,
        clone_root=tmp_path / "clones",
        output_jsonl=output_jsonl,
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=10,
    )

    assert report["rows"] == 1
    row = json.loads(output_jsonl.read_text(encoding="utf-8"))
    assert row["success"] is True
    assert row["oracle_trace_used"] is False
    assert row["sealed_feedback_used"] is False
    assert row["patch_source"] == "package_runtime_patch_proposal"
    assert row["patch_generator"] == "bounded_assertion_literal_patch_v1"

    audit = audit_phase2x_open_repair_execution_results(
        training_readiness_json=_write_json(tmp_path / "readiness.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime.json", {"passed": True}),
        results_jsonl=output_jsonl,
    )
    assert audit["passed"] is True
