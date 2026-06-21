from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reflexlm.models.features import MAX_CANDIDATE_SLOTS, candidate_commands, candidate_files
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    FileSystemState,
    GoalSpec,
    ProcessState,
    RuntimeEvidenceState,
    SafetyState,
    SystemStateFrame,
    TerminalState,
    TimeState,
    UserState,
    validate_command_against_goal,
)

MOTOR_ACTIONS: tuple[ActionType, ...] = tuple(ActionType)


def action_to_index(action_type: ActionType) -> int:
    return list(MOTOR_ACTIONS).index(action_type)


def action_from_index(index: int) -> ActionType:
    bounded = max(0, min(int(index), len(MOTOR_ACTIONS) - 1))
    return MOTOR_ACTIONS[bounded]


def _canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ComputerObservation(BaseModel):
    """Typed receptor snapshot consumed by ReflexCore V0.

    The fields mirror the existing bounded runtime schema and add candidate
    slots plus optional numeric/text encodings. Candidate commands remain a
    subset of the goal allowlist; there is no free-form shell surface here.
    """

    model_config = ConfigDict(extra="forbid")

    time: TimeState
    goal: GoalSpec
    process: ProcessState
    terminal: TerminalState
    filesystem: FileSystemState
    user: UserState = Field(default_factory=UserState)
    safety: SafetyState = Field(default_factory=SafetyState)
    runtime_evidence: RuntimeEvidenceState = Field(default_factory=RuntimeEvidenceState)
    candidate_commands: list[str] = Field(default_factory=list)
    candidate_files: list[str] = Field(default_factory=list)
    text_tokens: list[int] = Field(default_factory=list)
    vector: list[float] = Field(default_factory=list)
    source_frame_hash: str | None = None

    @field_validator("candidate_commands", "candidate_files")
    @classmethod
    def validate_candidate_slot_count(cls, value: list[str]) -> list[str]:
        if len(value) > MAX_CANDIDATE_SLOTS:
            raise ValueError(f"candidate slots exceed {MAX_CANDIDATE_SLOTS}")
        return value

    @field_validator("text_tokens")
    @classmethod
    def validate_text_tokens(cls, value: list[int]) -> list[int]:
        if any(token < 0 for token in value):
            raise ValueError("text token ids must be non-negative")
        return value

    @field_validator("vector")
    @classmethod
    def validate_vector_values(cls, value: list[float]) -> list[float]:
        if any(not isinstance(item, int | float) for item in value):
            raise ValueError("observation vector must contain numeric values")
        return [float(item) for item in value]

    @model_validator(mode="after")
    def validate_command_slots_are_allowlisted(self) -> "ComputerObservation":
        allowlist = set(self.goal.command_allowlist)
        extra = [command for command in self.candidate_commands if command not in allowlist]
        if extra:
            raise ValueError(f"candidate command is not allowlisted: {extra[0]!r}")
        return self

    @classmethod
    def from_state_frame(
        cls,
        state: SystemStateFrame,
        *,
        text_tokens: list[int] | None = None,
        vector: list[float] | None = None,
    ) -> "ComputerObservation":
        frame_payload = state.model_dump(mode="json")
        return cls(
            time=state.time,
            goal=state.goal,
            process=state.process,
            terminal=state.terminal,
            filesystem=state.filesystem,
            user=state.user,
            safety=state.safety,
            runtime_evidence=state.runtime_evidence,
            candidate_commands=candidate_commands(state),
            candidate_files=candidate_files(state),
            text_tokens=text_tokens or [],
            vector=vector or [],
            source_frame_hash=_canonical_hash(frame_payload),
        )

    def to_state_frame(self) -> SystemStateFrame:
        return SystemStateFrame(
            time=self.time,
            goal=self.goal,
            process=self.process,
            terminal=self.terminal,
            filesystem=self.filesystem,
            user=self.user,
            safety=self.safety,
            runtime_evidence=self.runtime_evidence,
        )


class MotorAction(BaseModel):
    """Typed bounded motor action emitted by ReflexCore V0."""

    model_config = ConfigDict(extra="forbid")

    type: ActionType
    command: str | None = None
    file_target: str | None = None
    reason: str | None = None
    confidence: float = 1.0
    notes: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return float(value)

    @model_validator(mode="after")
    def validate_payload(self) -> "MotorAction":
        if self.type == ActionType.RUN_COMMAND and not self.command:
            raise ValueError("RUN_COMMAND requires a command payload")
        if self.type == ActionType.READ_FILE and not self.file_target:
            raise ValueError("READ_FILE requires a file_target")
        return self

    @classmethod
    def from_decision(cls, decision: ActionDecision) -> "MotorAction":
        return cls(
            type=decision.type,
            command=decision.command,
            file_target=decision.file_target,
            reason=decision.reason,
            confidence=decision.confidence,
            notes=list(decision.notes),
        )

    def to_decision(self) -> ActionDecision:
        return ActionDecision(
            type=self.type,
            command=self.command,
            file_target=self.file_target,
            reason=self.reason,
            confidence=self.confidence,
            notes=list(self.notes),
        )

    def validate_against_goal(self, goal: GoalSpec) -> None:
        validate_command_against_goal(self.to_decision(), goal)


class ReflexCoreTrainingExample(BaseModel):
    """One sensory-motor transition for ReflexCore V0 training."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    t: int
    observation: ComputerObservation
    action: MotorAction
    next_observation: ComputerObservation
    reward: float = 0.0
    done: bool = False
    source: str = "unknown"

    @model_validator(mode="after")
    def validate_transition(self) -> "ReflexCoreTrainingExample":
        if self.observation.goal.task_type != self.next_observation.goal.task_type:
            raise ValueError("observation and next_observation task types must match")
        self.action.validate_against_goal(self.observation.goal)
        return self

    def canonical_hash(self) -> str:
        return _canonical_hash(self.model_dump(mode="json"))


def dataset_hash(examples: list[ReflexCoreTrainingExample]) -> str:
    payload = [example.model_dump(mode="json") for example in examples]
    return _canonical_hash(payload)
