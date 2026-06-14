import json
from pathlib import Path

import reflexlm.cli.run_phase2aw_package_loaded_descriptor_execution as runner
from reflexlm.cli.run_phase2aw_package_loaded_descriptor_execution import (
    run_phase2aw_package_loaded_descriptor_execution,
)
from reflexlm.llm.native_nervous_package import write_native_nervous_package


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    return path


def _package(tmp_path: Path) -> Path:
    package = tmp_path / "pkg"
    write_native_nervous_package(
        package,
        base_model_name="qwen",
        native_head_path="heads",
        low_level_checkpoint_path="low.pt",
        policy_label="phase2aw_test_package",
    )
    return package


def _task(expected: str = "fix_b") -> dict:
    return {
        "task_id": "task-1",
        "repo_origin": "https://github.com/example/repo.git",
        "repo_commit": "abc",
        "expected_repair_action": expected,
        "artifact_paths": {"generated_tests": ["tests/test_generated.py"]},
        "repair_candidates": [
            {"repair_action": "fix_a", "structural_probe_hash": "a"},
            {"repair_action": "fix_b", "structural_probe_hash": "b"},
        ],
        "runtime_visible_evidence": {"changed_files": ["src/mod.py"]},
    }


def _patch_selection_helpers(monkeypatch, tmp_path: Path) -> None:
    patch = tmp_path / "patch.diff"
    patch.write_text("diff --git a/src/mod.py b/src/mod.py\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_copy_public_repo", lambda row, clone_root, sandbox: sandbox.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(runner, "_materialize_generated_test", lambda row, root, sandbox: "tests/test_generated.py")
    monkeypatch.setattr(runner, "_resolve_patch", lambda row, root: (patch, patch.read_text(encoding="utf-8")))
    monkeypatch.setattr(
        runner,
        "_git_apply_reverse_patch",
        lambda sandbox, patch_path, timeout_seconds: {
            "exit_code": 0,
            "duration_seconds": 0.01,
            "stdout": "",
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        runner,
        "_run_pytest_target",
        lambda sandbox, test_rel, timeout_seconds, python_executable: {
            "exit_code": 1,
            "duration_seconds": 0.01,
            "stdout": "failed",
            "stderr": "",
        },
    )


def test_phase2aw_package_loaded_runner_uses_loaded_package_slot_and_executes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakePackage:
        def __init__(self, _package_path: Path) -> None:
            self.last_call = {}

        def reset(self) -> None:
            self.last_call = {}

        def act(self, _state) -> None:
            self.last_call = {
                "action_source": "native_head_cortex",
                "cortex_plan": {"command_slot": 1},
                "open_repair_head_outputs": {
                    "patch_proposal": 1,
                    "bounded_edit_scope": 1,
                    "rollback_safety": 1,
                },
                "qwen_called": True,
            }

    _patch_selection_helpers(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "NativeNervousPolicyPackage", FakePackage)
    monkeypatch.setattr(
        runner,
        "_execute_selected_row",
        lambda **_kwargs: {
            "summary": {"rows": 1, "successes": 1},
            "row": {
                "success": True,
                "full_patch_correctness": True,
                "full_test_pass_rate": 1.0,
                "rollback_failure_restored": True,
                "unauthorized_write_count": 0,
                "false_completion": False,
                "verification_state": "passed",
                "stop_condition": "verification_passed",
                "artifact_paths": {"patch": "patch.diff"},
            },
        },
    )

    report = run_phase2aw_package_loaded_descriptor_execution(
        runtime_tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", [_task()]),
        dataset_root=tmp_path,
        clone_root=tmp_path / "clones",
        package_path=_package(tmp_path),
        output_jsonl=tmp_path / "out.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["policy_loaded"] is True
    assert report["success_rate"] == 1.0
    assert report["patch_candidate_selection_accuracy"] == 1.0
    assert rows[0]["selected_patch_candidate_slot"] == 1
    assert rows[0]["policy_loaded"] is True
    assert rows[0]["freeform_patch_generation"] is False


def test_phase2aw_package_loaded_runner_stops_when_package_selects_wrong_slot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakePackage:
        def __init__(self, _package_path: Path) -> None:
            self.last_call = {}

        def reset(self) -> None:
            self.last_call = {}

        def act(self, _state) -> None:
            self.last_call = {
                "action_source": "native_head_cortex",
                "cortex_plan": {"command_slot": 0},
                "open_repair_head_outputs": {
                    "patch_proposal": 1,
                    "bounded_edit_scope": 1,
                    "rollback_safety": 1,
                },
                "qwen_called": True,
            }

    _patch_selection_helpers(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "NativeNervousPolicyPackage", FakePackage)
    monkeypatch.setattr(
        runner,
        "_execute_selected_row",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not execute")),
    )

    report = run_phase2aw_package_loaded_descriptor_execution(
        runtime_tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", [_task()]),
        dataset_root=tmp_path,
        clone_root=tmp_path / "clones",
        package_path=_package(tmp_path),
        output_jsonl=tmp_path / "out.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["success_rate"] == 0.0
    assert report["patch_candidate_selection_accuracy"] == 0.0
    assert rows[0]["stop_condition"] == "package_candidate_selection_failed_before_patch_application"


def test_phase2aw_package_loaded_runner_observes_stderr_receptor_before_slot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakePackage:
        def __init__(self, _package_path: Path) -> None:
            self.last_call = {}
            self.calls = 0

        def reset(self) -> None:
            self.last_call = {}
            self.calls = 0

        def act(self, _state) -> None:
            self.calls += 1
            if self.calls == 1:
                self.last_call = {"action_source": "low_level_debug_receptor"}
                return
            self.last_call = {
                "action_source": "native_head_cortex",
                "cortex_plan": {"command_slot": 1},
                "open_repair_head_outputs": {
                    "patch_proposal": 1,
                    "bounded_edit_scope": 1,
                    "rollback_safety": 1,
                },
                "qwen_called": True,
            }

    _patch_selection_helpers(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "NativeNervousPolicyPackage", FakePackage)
    monkeypatch.setattr(
        runner,
        "_execute_selected_row",
        lambda **_kwargs: {
            "summary": {"rows": 1, "successes": 1},
            "row": {
                "success": True,
                "full_patch_correctness": True,
                "full_test_pass_rate": 1.0,
                "rollback_failure_restored": True,
                "verification_state": "passed",
                "stop_condition": "verification_passed",
            },
        },
    )

    run_phase2aw_package_loaded_descriptor_execution(
        runtime_tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", [_task()]),
        dataset_root=tmp_path,
        clone_root=tmp_path / "clones",
        package_path=_package(tmp_path),
        output_jsonl=tmp_path / "out.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
    )

    row = json.loads((tmp_path / "out.jsonl").read_text(encoding="utf-8").strip())
    assert row["low_level_debug_receptor_observed"] is True
    assert row["selected_patch_candidate_slot"] == 1
