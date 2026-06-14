from pathlib import Path

from reflexlm.runtime.plasticity import SynapticPlasticityMemory, state_pattern_key
from reflexlm.schema import (
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    RuntimeEvidenceState,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


def _state(*, commands: list[str] | None = None, probe: str | None = "probe") -> SystemStateFrame:
    return SystemStateFrame(
        time=TimeState(tick=1),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="bounded repair",
            command_allowlist=commands or ["repair-a", "repair-b"],
        ),
        process=ProcessState(status=ProcessStatus.EXITED, exit_code=1),
        terminal=TerminalState(),
        filesystem=FileSystemState(changed_paths=["src/a.py"], watched_paths=["tests/test_a.py"]),
        runtime_evidence=RuntimeEvidenceState(
            changed_files=["src/a.py"],
            watched_files=["tests/test_a.py"],
            structural_probe_hashes=[probe] if probe else [],
            repair_modes=["literal_restoration"],
            descriptor_operation="replace_literal",
            descriptor_template="literal_restoration",
        ),
    )


def test_plasticity_learns_only_from_verified_allowlisted_feedback() -> None:
    memory = SynapticPlasticityMemory()
    state = _state()

    rejected = memory.observe_feedback(
        state,
        command="repair-a",
        verified_success=True,
        verifier="self_reward",
    )
    accepted = memory.observe_feedback(
        state,
        command="repair-a",
        verified_success=True,
        verifier="post_execution_verifier",
    )

    assert rejected["accepted"] is False
    assert accepted["accepted"] is True
    assert memory.predict(state)["selected_slot"] == 0
    assert memory.feedback_events == 1
    assert memory.rejected_feedback_events == 1


def test_plasticity_pattern_excludes_answer_probe_and_survives_candidate_reordering() -> None:
    memory = SynapticPlasticityMemory()
    source = _state(probe="answer-probe")
    erased_reordered = _state(commands=["repair-b", "repair-a"], probe=None)

    memory.observe_feedback(
        source,
        command="repair-a",
        verified_success=True,
        verifier="post_execution_verifier",
    )

    assert state_pattern_key(source) == state_pattern_key(erased_reordered)
    assert memory.predict(erased_reordered)["selected_slot"] == 1


def test_plasticity_failure_feedback_suppresses_connection_and_reports_prediction_error() -> None:
    memory = SynapticPlasticityMemory()
    state = _state()
    memory.observe_feedback(
        state,
        command="repair-a",
        verified_success=True,
        verifier="post_execution_verifier",
    )

    feedback = memory.observe_feedback(
        state,
        command="repair-a",
        verified_success=False,
        verifier="post_execution_verifier",
    )

    assert feedback["prediction_error"] == 1.0
    assert memory.predict(state)["memory_hit"] is False


def test_plasticity_snapshot_roundtrip(tmp_path: Path) -> None:
    memory = SynapticPlasticityMemory()
    state = _state()
    memory.observe_feedback(
        state,
        command="repair-b",
        verified_success=True,
        verifier="post_execution_verifier",
    )
    path = tmp_path / "memory.json"
    memory.save(path)

    loaded = SynapticPlasticityMemory.load(path)

    assert loaded.predict(state)["selected_slot"] == 1
    assert loaded.snapshot() == memory.snapshot()


def test_plasticity_wrong_control_rotates_visible_candidate_without_gold() -> None:
    memory = SynapticPlasticityMemory()
    state = _state()
    memory.observe_feedback(
        state,
        command="repair-a",
        verified_success=True,
        verifier="post_execution_verifier",
    )
    memory.control = "wrong"

    prediction = memory.predict(state)

    assert prediction["selected_slot"] == 1
    assert prediction["wrong_memory_injected"] is True
