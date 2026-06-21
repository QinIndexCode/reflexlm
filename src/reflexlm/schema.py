from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reflexlm.spec import action_space


class ActionType(str, Enum):
    WAIT = "WAIT"
    READ_STDOUT = "READ_STDOUT"
    READ_STDERR = "READ_STDERR"
    READ_FILE = "READ_FILE"
    RUN_COMMAND = "RUN_COMMAND"
    STOP_PROCESS = "STOP_PROCESS"
    ASK_USER = "ASK_USER"
    REFRESH_STATE = "REFRESH_STATE"
    BLOCK = "BLOCK"
    DONE = "DONE"


class TaskType(str, Enum):
    BLOCKING_INPUT = "blocking_input_detection"
    TEST_FAILURE = "test_failure_reflex"
    PROCESS_HANG = "process_hang_detection"
    DANGEROUS_ACTION = "dangerous_action_interception"
    FILE_CHANGE = "external_file_change_reflex"
    ROUTINE_RECOVERY = "common_error_recovery_routine"


class RouteName(str, Enum):
    TERMINAL = "terminal_cortex"
    DEBUG = "debug_cortex"
    FILE = "file_cortex"
    PLANNER = "planner_cortex"
    SAFETY = "safety_cortex"


class InternalTarget(str, Enum):
    REFLEX_MOTOR = "REFLEX_MOTOR"
    ESCALATE_TO_DEBUG_CORTEX = "ESCALATE_TO_DEBUG_CORTEX"
    ESCALATE_TO_SEMANTIC_CORTEX = "ESCALATE_TO_SEMANTIC_CORTEX"
    INHIBIT = "INHIBIT"


class SourceType(str, Enum):
    SYNTHETIC = "synthetic"
    RULE_ORACLE = "rule_oracle"
    HUMAN = "human"
    LLM_AGENT = "llm_agent"
    MODEL = "model"
    RUNTIME_OBSERVATION = "runtime_observation"


class ProcessStatus(str, Enum):
    RUNNING = "running"
    SLEEPING = "sleeping"
    BLOCKED = "blocked"
    EXITED = "exited"


class GoalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    description: str
    command_allowlist: list[str] = Field(default_factory=list)
    watched_paths: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    recovery_hint: str | None = None


class TimeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tick: int = 0
    runtime_ms: int = 0
    wall_clock_ms: int = 0
    since_last_output_ms: int = 0
    since_last_state_change_ms: int = 0


class ProcessState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pid: int | None = None
    parent_pid: int | None = None
    status: ProcessStatus = ProcessStatus.RUNNING
    exit_code: int | None = None
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    runtime_ms: int = 0
    last_output_ms: int = 0
    waiting_for_input: bool = False
    interrupted: bool = False
    has_children: bool = False
    resource_alert: bool = False


class TerminalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout_delta: str = ""
    stderr_delta: str = ""
    stdout_unread: bool = False
    stderr_unread: bool = False
    stdout_lines: int = 0
    stderr_lines: int = 0
    prompt_visible: bool = False
    input_requested: bool = False
    last_output_channel: str | None = None
    last_command: str | None = None


class FileSystemState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    watched_paths: list[str] = Field(default_factory=list)
    changed_paths: list[str] = Field(default_factory=list)
    dirty_files: list[str] = Field(default_factory=list)
    external_change_detected: bool = False
    stale_cache_detected: bool = False
    conflict_detected: bool = False


class UserState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manual_input_active: bool = False
    confirmation_required: bool = False
    user_block_requested: bool = False


class SafetyState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dangerous_command_detected: bool = False
    command_candidate: str | None = None
    risk_label: str | None = None


class RuntimeEvidenceState(BaseModel):
    """Structured receptor observations produced by prior runtime activity."""

    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    version: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    watched_files: list[str] = Field(default_factory=list)
    terminal_observations: list[str] = Field(default_factory=list)
    structural_probe_hashes: list[str] = Field(default_factory=list)
    traceback_symbols: list[str] = Field(default_factory=list)
    repair_modes: list[str] = Field(default_factory=list)
    descriptor_operation: str | None = None
    descriptor_template: str | None = None
    expected_literal_hash: str | None = None
    target_path_hash: str | None = None
    target_symbol_hash: str | None = None
    target_path: str | None = None
    target_line: int | None = None
    target_col: int | None = None
    model_prediction_error: float | None = None
    observed_prediction_error: float | None = None
    prediction_error_delta: float | None = None


class SystemStateFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: TimeState
    goal: GoalSpec
    process: ProcessState
    terminal: TerminalState
    filesystem: FileSystemState
    user: UserState = Field(default_factory=UserState)
    safety: SafetyState = Field(default_factory=SafetyState)
    runtime_evidence: RuntimeEvidenceState = Field(default_factory=RuntimeEvidenceState)


class ActionDecision(BaseModel):
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
        return value

    @model_validator(mode="after")
    def validate_payload(self) -> "ActionDecision":
        if self.type == ActionType.RUN_COMMAND and not self.command:
            raise ValueError("RUN_COMMAND requires a command payload")
        if self.type == ActionType.READ_FILE and not self.file_target:
            raise ValueError("READ_FILE requires a file_target")
        return self


class TrajectoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    t: int
    goal: GoalSpec
    state: SystemStateFrame
    action: ActionDecision | None = None
    next_state: SystemStateFrame
    reward: float
    done: bool
    source: SourceType

    @model_validator(mode="after")
    def validate_goal_consistency(self) -> "TrajectoryRecord":
        if self.goal.task_type != self.state.goal.task_type:
            raise ValueError("goal and state.goal.task_type must match")
        if self.goal.task_type != self.next_state.goal.task_type:
            raise ValueError("goal and next_state.goal.task_type must match")
        return self


def validate_command_against_goal(action: ActionDecision, goal: GoalSpec) -> None:
    if action.type != ActionType.RUN_COMMAND or action.command is None:
        return
    if action.command not in goal.command_allowlist:
        raise ValueError(
            f"command {action.command!r} is not allowlisted for goal {goal.task_type.value}"
        )


def action_name_to_enum(name: str) -> ActionType:
    for action in ActionType:
        if action.value == name:
            return action
    raise ValueError(f"Unknown action name: {name}")


def canonical_action_names() -> list[str]:
    return action_space()


def route_bias_template(primary: RouteName) -> dict[str, float]:
    base = {route.value: 0.05 for route in RouteName}
    base[primary.value] = 0.8
    return base


def dump_jsonable(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")
