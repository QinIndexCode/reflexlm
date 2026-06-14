from pathlib import Path

from reflexlm.cli.build_phase2w_reproduction_bundle import (
    DEFAULT_VERIFY_COMMANDS,
    build_phase2w_reproduction_bundle,
)


def test_phase2w_reproduction_bundle_locks_hashes_but_is_not_independent(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"passed": true}', encoding="utf-8")
    report = build_phase2w_reproduction_bundle(artifact_paths=[artifact])
    assert report["bundle_ready_for_independent_runner"] is True
    assert report["passed"] is False
    assert report["runner_independent"] is False
    assert report["one_command_reproduction"] is True
    assert report["hash_locked_splits"] is True
    assert report["artifacts"][0]["sha256"]


def test_phase2w_reproduction_bundle_reports_missing_artifacts(tmp_path: Path) -> None:
    report = build_phase2w_reproduction_bundle(
        artifact_paths=[tmp_path / "missing.json"]
    )
    assert report["bundle_ready_for_independent_runner"] is False
    assert report["missing_artifacts"] == [str(tmp_path / "missing.json")]


def test_phase2w_reproduction_bundle_default_commands_are_cross_shell() -> None:
    assert DEFAULT_VERIFY_COMMANDS
    assert all(command.startswith("python -m pytest ") for command in DEFAULT_VERIFY_COMMANDS)
    assert all("PYTHONPATH=" not in command for command in DEFAULT_VERIFY_COMMANDS)
    assert any("test_phase2x_open_repair_task_manifest.py" in command for command in DEFAULT_VERIFY_COMMANDS)
    assert any("test_phase2w_live_agent_baseline.py" in command for command in DEFAULT_VERIFY_COMMANDS)
