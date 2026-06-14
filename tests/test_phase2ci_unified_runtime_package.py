import json
from pathlib import Path

from reflexlm.cli.build_phase2ci_unified_runtime_package import (
    build_phase2ci_unified_runtime_package,
)
from reflexlm.cli.run_phase2bq_open_task_family_repo_runtime import (
    run_phase2bq_open_task_family_repo_runtime,
)
from reflexlm.llm.native_nervous_package import NativeNervousPolicyPackage
import reflexlm.cli.build_phase2ci_unified_runtime_package as builder
import reflexlm.cli.run_phase2bq_open_task_family_repo_runtime as phase2bq
import reflexlm.llm.native_nervous_package as native_nervous_package


class _FakeNativeHeadPolicy:
    def __init__(self, **_kwargs) -> None:
        self.stats = type("Stats", (), {"token_cost": 0, "model_calls": 0})()
        self.last_call = {}

    def metadata(self) -> dict:
        return {"policy_family": "phase2c_native_heads"}


def _base_manifest(tmp_path: Path) -> Path:
    package = tmp_path / "base_package"
    package.mkdir()
    (package / "native_nervous_package.json").write_text(
        json.dumps(
            {
                "package_family": "phase2d_native_nervous_package",
                "policy_label": "base",
                "base_model_name": "model",
                "native_head_path": "head",
                "low_level_checkpoint_path": "low",
                "quantization": "none",
                "nsi_device": "cpu",
                "device": "cpu",
                "model_load_strategy": "auto",
                "offload_state_dict": True,
                "verification_cortex_path": "verification.pt",
            }
        ),
        encoding="utf-8",
    )
    return package


def test_phase2ci_builder_extends_unified_package_manifest(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "runtime.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "environment.json").write_text(
        json.dumps({"python": {"executable": "C:\\Python313\\python.exe"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        builder,
        "load_model_checkpoint",
        lambda *_args, **_kwargs: (object(), object(), {"training_summary": {"epochs": 3}}),
    )

    report = build_phase2ci_unified_runtime_package(
        base_package_path=_base_manifest(tmp_path),
        structured_runtime_checkpoint_path=checkpoint,
        output_package_dir=tmp_path / "output_package",
        output_report_json=tmp_path / "report.json",
    )

    manifest = json.loads(
        (tmp_path / "output_package" / "native_nervous_package.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"] is True
    assert manifest["structured_runtime_cortex_checkpoint_path"] == str(checkpoint)
    assert manifest["structured_runtime_cortex_python_identity"] == (
        "C:\\Python313\\python.exe"
    )
    assert manifest["verification_cortex_path"] == "verification.pt"


def test_phase2ci_builder_packages_calibrated_runtime_with_explicit_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "promoted-calibrated-runtime.pt"
    checkpoint.write_bytes(b"checkpoint")
    calibration = {"threshold": 0.25, "schema_version": "calibration.v1"}
    monkeypatch.setattr(
        builder,
        "load_model_checkpoint",
        lambda *_args, **_kwargs: (
            object(),
            object(),
            {
                "training_summary": {
                    "epochs": 3,
                    "prediction_error_calibration": calibration,
                }
            },
        ),
    )

    report = build_phase2ci_unified_runtime_package(
        base_package_path=_base_manifest(tmp_path),
        structured_runtime_checkpoint_path=checkpoint,
        structured_runtime_python_identity="C:\\Python313\\python.exe",
        require_prediction_error_calibration=True,
        output_package_dir=tmp_path / "output_package",
        output_report_json=tmp_path / "report.json",
    )

    manifest = json.loads(
        (tmp_path / "output_package" / "native_nervous_package.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"] is True
    assert report["checks"]["structured_runtime_prediction_error_calibrated"] is True
    assert manifest["structured_runtime_cortex_python_identity"] == (
        "C:\\Python313\\python.exe"
    )
    assert manifest["structured_runtime_cortex_prediction_error_calibration"] == calibration


def test_package_creates_structured_runtime_policy_adapter(tmp_path: Path, monkeypatch) -> None:
    package = _base_manifest(tmp_path)
    manifest_path = package / "native_nervous_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["structured_runtime_cortex_checkpoint_path"] = str(tmp_path / "runtime.pt")
    manifest["verification_cortex_path"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class _FakeSequencePolicy:
        def __init__(self, *_args, **_kwargs) -> None:
            self.stats = type("Stats", (), {"token_cost": 0, "model_calls": 0})()
            self.last_call = {}

        def reset(self) -> None:
            self.last_call = {}

        def act(self, state):
            return state

        def metadata(self) -> dict:
            return {"policy_family": "sequence_model"}

        def save_homeostatic_state(self, path, *, authenticity_key=None):
            assert authenticity_key is None
            return {"path": str(path)}

        def load_homeostatic_state(self, path, *, authenticity_key=None):
            assert authenticity_key is None
            return {"path": str(path)}

    monkeypatch.setattr(
        native_nervous_package,
        "NativeHeadPolicy",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("structured runtime package view loaded native head policy")
        ),
    )
    monkeypatch.setattr(
        "reflexlm.train.load_model_checkpoint",
        lambda *_args, **_kwargs: (object(), object(), {"training_summary": {}}),
    )
    monkeypatch.setattr("reflexlm.eval.SequenceModelPolicy", _FakeSequencePolicy)

    package_view = NativeNervousPolicyPackage(
        package,
        load_native_head_policy=False,
        load_verification_cortex=False,
    )
    policy = package_view.create_structured_runtime_policy()

    metadata = policy.metadata()
    assert metadata["package_internal_expert"] is True
    assert metadata["expert_name"] == "structured_runtime_cortex"
    package_metadata = package_view.metadata()
    assert package_metadata["native_head_policy_loaded"] is False
    assert package_metadata["structured_runtime_cortex_packaged"] is True
    assert policy.save_homeostatic_state(tmp_path / "state.json")["path"].endswith(
        "state.json"
    )
    assert policy.load_homeostatic_state(tmp_path / "state.json")["path"].endswith(
        "state.json"
    )


def test_structured_runtime_policy_canonicalizes_python_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = _base_manifest(tmp_path)
    manifest_path = package / "native_nervous_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verification_cortex_path"] = None
    manifest["structured_runtime_cortex_checkpoint_path"] = str(tmp_path / "runtime.pt")
    manifest["structured_runtime_cortex_python_identity"] = "C:\\Python313\\python.exe"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class _FakeSequencePolicy:
        def __init__(self, *_args, **_kwargs) -> None:
            self.stats = type("Stats", (), {"token_cost": 0, "model_calls": 0})()
            self.last_call = {}

        def reset(self) -> None:
            pass

        def act(self, state):
            from reflexlm.schema import ActionDecision, ActionType

            assert state.goal.command_allowlist[0].startswith("C:\\Python313\\python.exe")
            self.last_call = {"seen_command": state.goal.command_allowlist[0]}
            return ActionDecision(
                type=ActionType.RUN_COMMAND,
                command=state.goal.command_allowlist[0],
                confidence=1.0,
            )

        def metadata(self) -> dict:
            return {"policy_family": "sequence_model"}

    monkeypatch.setattr(native_nervous_package, "NativeHeadPolicy", _FakeNativeHeadPolicy)
    monkeypatch.setattr(
        "reflexlm.train.load_model_checkpoint",
        lambda *_args, **_kwargs: (object(), object(), {"training_summary": {}}),
    )
    monkeypatch.setattr("reflexlm.eval.SequenceModelPolicy", _FakeSequencePolicy)
    monkeypatch.setattr(native_nervous_package.sys, "executable", "D:\\venv\\python.exe")
    package_policy = NativeNervousPolicyPackage(package).create_structured_runtime_policy()

    from reflexlm.schema import (
        FileSystemState,
        GoalSpec,
        ProcessState,
        SystemStateFrame,
        TaskType,
        TerminalState,
        TimeState,
    )

    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="canonicalize python identity",
            command_allowlist=["D:\\venv\\python.exe -c \"print('ok')\""],
        ),
        process=ProcessState(),
        terminal=TerminalState(),
        filesystem=FileSystemState(),
    )

    action = package_policy.act(state)

    assert action.command == "D:\\venv\\python.exe -c \"print('ok')\""
    assert package_policy.last_call["package_python_identity_canonicalized"] is True
    assert package_policy.last_call["package_python_identity_mapping_scope"] == (
        "command_executable_prefix"
    )


def test_structured_runtime_policy_only_canonicalizes_command_executable_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = _base_manifest(tmp_path)
    manifest_path = package / "native_nervous_package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verification_cortex_path"] = None
    manifest["structured_runtime_cortex_checkpoint_path"] = str(tmp_path / "runtime.pt")
    manifest["structured_runtime_cortex_python_identity"] = "C:\\Python313\\python.exe"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class _FakeSequencePolicy:
        def __init__(self, *_args, **_kwargs) -> None:
            self.stats = type("Stats", (), {"token_cost": 0, "model_calls": 0})()
            self.last_call = {}

        def reset(self) -> None:
            pass

        def act(self, state):
            from reflexlm.schema import ActionDecision, ActionType

            commands = state.goal.command_allowlist
            assert commands[0] == (
                "C:\\Python313\\python.exe -c "
                "\"print(r'D:\\venv\\python.exe')\""
            )
            assert commands[1] == (
                "\"C:\\Python313\\python.exe\" -c "
                "\"print(r'D:\\venv\\python.exe')\""
            )
            assert commands[2] == "echo D:\\venv\\python.exe"
            return ActionDecision(
                type=ActionType.RUN_COMMAND,
                command=commands[1],
                confidence=1.0,
            )

        def metadata(self) -> dict:
            return {"policy_family": "sequence_model"}

    monkeypatch.setattr(native_nervous_package, "NativeHeadPolicy", _FakeNativeHeadPolicy)
    monkeypatch.setattr(
        "reflexlm.train.load_model_checkpoint",
        lambda *_args, **_kwargs: (object(), object(), {"training_summary": {}}),
    )
    monkeypatch.setattr("reflexlm.eval.SequenceModelPolicy", _FakeSequencePolicy)
    monkeypatch.setattr(native_nervous_package.sys, "executable", "D:\\venv\\python.exe")
    package_policy = NativeNervousPolicyPackage(package).create_structured_runtime_policy()

    from reflexlm.schema import (
        FileSystemState,
        GoalSpec,
        ProcessState,
        SystemStateFrame,
        TaskType,
        TerminalState,
        TimeState,
    )

    runtime_path_argument = "\"print(r'D:\\venv\\python.exe')\""
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="canonicalize executable prefix only",
            command_allowlist=[
                f"D:\\venv\\python.exe -c {runtime_path_argument}",
                f'"D:\\venv\\python.exe" -c {runtime_path_argument}',
                "echo D:\\venv\\python.exe",
            ],
        ),
        process=ProcessState(),
        terminal=TerminalState(),
        filesystem=FileSystemState(),
    )

    action = package_policy.act(state)

    assert action.command == f'"D:\\venv\\python.exe" -c {runtime_path_argument}'


def test_phase2bq_package_mode_requires_package_internal_runtime_cortex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = {
        "source_repository_root": str(tmp_path),
        "training_manifest_json": "train.json",
        "seed": 7,
        "minimum_repository_count": 1,
        "recipes_per_repository": 1,
        "repetitions_per_episode": 1,
        "repositories": [
            {"repository_id": "repo_a", "workspace_root": str(tmp_path)},
        ],
    }
    (tmp_path / "suite.json").write_text(json.dumps(suite), encoding="utf-8")
    (tmp_path / "train.json").write_text(json.dumps({"episodes": []}), encoding="utf-8")

    class _FakePackage:
        def __init__(self, *_args, **kwargs) -> None:
            assert kwargs["load_native_head_policy"] is False
            assert kwargs["load_verification_cortex"] is False

        def create_structured_runtime_policy(self):
            return object()

        def metadata(self) -> dict:
            return {"structured_runtime_cortex_packaged": True}

    def _fake_run(**kwargs):
        assert kwargs["checkpoint_path"] is None
        assert kwargs["policy_instance"] is not None
        return {
            "passed": True,
            "checks": {
                "all_model_selected_actions_were_allowlisted": True,
                "all_task_completion_predicates_satisfied": True,
            },
            "metrics": {
                "episodes": 1,
                "executed_actions": 1,
                "task_completion_successes": 1,
            },
            "policy_configuration": {
                "policy_metadata": {
                    "package_internal_expert": True,
                    "expert_name": "structured_runtime_cortex",
                }
            },
        }

    monkeypatch.setattr(phase2bq, "NativeNervousPolicyPackage", _FakePackage)
    monkeypatch.setattr(phase2bq, "run_phase2bn_model_selected_sealed_runtime", _fake_run)
    monkeypatch.setattr(
        phase2bq,
        "_git_repository_provenance",
        lambda root: {"git_root": str(root), "origin": str(root), "head": "abc"},
    )
    monkeypatch.setattr(
        phase2bq,
        "_repo_identity_checks",
        lambda **_kwargs: {
            "minimum_independent_repository_count_met": True,
            "all_git_roots_are_distinct": True,
            "all_git_roots_differ_from_source_repository": True,
            "all_origins_are_distinct": True,
            "all_heads_are_recorded": True,
        },
    )

    report = run_phase2bq_open_task_family_repo_runtime(
        checkpoint_path=None,
        package_path=tmp_path / "package",
        suite_json=tmp_path / "suite.json",
        output_dir=tmp_path / "out",
        output_report_json=tmp_path / "report.json",
    )

    assert report["passed"] is True
    assert report["artifact_family"] == (
        "phase2ci_unified_package_open_task_family_repo_runtime"
    )
    assert report["ready_for_unified_package_generated_task_family_runtime_claim"] is True
    assert report["checks"]["all_repositories_used_package_internal_runtime_cortex"] is True
