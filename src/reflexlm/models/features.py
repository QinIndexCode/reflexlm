from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from reflexlm.runtime.oracle import primary_route_for_task
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    ProcessStatus,
    RouteName,
    SystemStateFrame,
    TaskType,
    TrajectoryRecord,
)

ACTION_ORDER = list(ActionType)
ROUTE_ORDER = list(RouteName)
MAX_CANDIDATE_SLOTS = 4
COMMAND_SLOT_FEATURES = 10
FILE_SLOT_FEATURES = 5
HARD_TASK_TYPES = {
    TaskType.TEST_FAILURE,
    TaskType.FILE_CHANGE,
    TaskType.ROUTINE_RECOVERY,
}
_STRUCTURED_COMMAND_IDENTITY_RE = re.compile(
    r"\b(?:phase2j_)?command_identity_tokens\s*=\s*(.*?)(?=\s+(?:edit_scope|target_symbol|risk_label|command_candidate)=|[\n;]|$)",
    re.IGNORECASE,
)
_STOCHASTIC_TELEMETRY_INDICES = frozenset({0, 1, 2, 3, 4, 5, 13, 14})


def _redact_structured_command_identity_text(text: str) -> str:
    return _STRUCTURED_COMMAND_IDENTITY_RE.sub("command_identity_tokens=<redacted>", text)


def action_to_index(action_type: ActionType) -> int:
    return ACTION_ORDER.index(action_type)


def route_to_index(route_name: RouteName) -> int:
    return ROUTE_ORDER.index(route_name)


def serialize_state_as_text(
    state: SystemStateFrame,
    *,
    include_internal_hints: bool = False,
) -> str:
    command_slots = state.goal.command_allowlist[:MAX_CANDIDATE_SLOTS]
    file_slots = []
    for path in (
        state.filesystem.dirty_files
        + state.filesystem.changed_paths
        + state.filesystem.watched_paths
        + state.goal.watched_paths
    ):
        if path not in file_slots:
            file_slots.append(path)
        if len(file_slots) >= MAX_CANDIDATE_SLOTS:
            break
    return "\n".join(
        [
            f"task={state.goal.task_type.value}",
            f"description={state.goal.description}",
            *(
                [f"recovery_hint={state.goal.recovery_hint or ''}"]
                if include_internal_hints
                else []
            ),
            f"runtime_ms={state.time.runtime_ms}",
            f"since_last_output_ms={state.time.since_last_output_ms}",
            f"process_status={state.process.status.value}",
            f"exit_code={state.process.exit_code}",
            f"cpu_percent={state.process.cpu_percent:.2f}",
            f"memory_mb={state.process.memory_mb:.2f}",
            f"waiting_for_input={state.process.waiting_for_input}",
            f"stdout={_redact_structured_command_identity_text(state.terminal.stdout_delta)}",
            f"stderr={_redact_structured_command_identity_text(state.terminal.stderr_delta)}",
            f"prompt_visible={state.terminal.prompt_visible}",
            f"input_requested={state.terminal.input_requested}",
            f"dirty_files={','.join(state.filesystem.dirty_files)}",
            f"changed_paths={','.join(state.filesystem.changed_paths)}",
            f"watched_paths={','.join(state.filesystem.watched_paths)}",
            f"external_change_detected={state.filesystem.external_change_detected}",
            f"manual_input_active={state.user.manual_input_active}",
            f"dangerous_command_detected={state.safety.dangerous_command_detected}",
            f"command_candidate={state.safety.command_candidate or ''}",
            f"allowlist={','.join(state.goal.command_allowlist)}",
            *[f"command_slot_{index}={command}" for index, command in enumerate(command_slots)],
            *[f"file_slot_{index}={path}" for index, path in enumerate(file_slots)],
        ]
    )


def prompt_token_estimate(state: SystemStateFrame) -> int:
    return len(serialize_state_as_text(state, include_internal_hints=False).split())


def candidate_commands(state: SystemStateFrame) -> list[str]:
    return state.goal.command_allowlist[:MAX_CANDIDATE_SLOTS]


def candidate_files(state: SystemStateFrame) -> list[str]:
    candidates: list[str] = []
    for path in (
        state.filesystem.dirty_files
        + state.filesystem.changed_paths
        + state.filesystem.watched_paths
    ):
        if path not in candidates:
            candidates.append(path)
        if len(candidates) >= MAX_CANDIDATE_SLOTS:
            break
    return candidates


def _visible_failure_signals(state: SystemStateFrame) -> dict[str, bool]:
    visible_text = " ".join(
        [
            state.goal.description,
            state.terminal.stdout_delta,
            state.terminal.stderr_delta,
        ]
    ).lower()
    snapshot_signal = "snapshot" in visible_text and (
        "mismatch" in visible_text or "update" in visible_text
    )
    dependency_signal = (
        "modulenotfounderror" in visible_text
        or "no module named" in visible_text
        or "missing dependency" in visible_text
        or "dependency missing" in visible_text
    )
    assertion_signal = "assertionerror" in visible_text or "assertion failure" in visible_text
    return {
        "snapshot": snapshot_signal,
        "dependency": dependency_signal,
        "assertion": assertion_signal and not snapshot_signal,
    }


_TOKEN_RE = re.compile(r"[a-z0-9_]+")

SEMANTIC_AFFORDANCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "dependency": (
        "modulenotfounderror",
        "no module named",
        "missing dependency",
        "dependency missing",
        "import failed",
        "package unavailable",
        "install package",
        "pip install",
        "requirements",
    ),
    "snapshot": (
        "snapshot mismatch",
        "snapshot update",
        "golden output differs",
        "stored reference",
        "regenerate expected output",
        "update snapshot",
    ),
    "assertion": (
        "assertionerror",
        "assertion failure",
        "test verification",
        "pytest",
        "targeted test",
    ),
    "port": (
        "address already in use",
        "port occupied",
        "tcp port",
        "release port",
        "free port",
        "kill port",
    ),
    "permission": (
        "permission denied",
        "access is denied",
        "not permitted",
        "repair permissions",
        "chmod",
        "file permissions",
    ),
    "path": (
        "filenotfounderror",
        "directory does not exist",
        "missing path",
        "create directory",
        "create missing directory",
        "ensure path",
    ),
}


def _semantic_affordances(text: str) -> set[str]:
    lowered = text.lower()
    concepts = {
        concept
        for concept, patterns in SEMANTIC_AFFORDANCE_PATTERNS.items()
        if any(pattern in lowered for pattern in patterns)
    }
    if "expected" in lowered and (
        "received" in lowered or "actual" in lowered or "but got" in lowered
    ):
        concepts.add("assertion")
    return concepts


def _state_semantic_text(state: SystemStateFrame) -> str:
    return " ".join(
        str(part)
        for part in [
            state.goal.description,
            state.terminal.stdout_delta,
            state.terminal.stderr_delta,
            state.terminal.last_command,
            " ".join(state.runtime_evidence.terminal_observations),
            " ".join(state.filesystem.dirty_files),
            " ".join(state.filesystem.changed_paths),
        ]
        if part is not None
    )


def semantic_affordance_names(state: SystemStateFrame) -> list[str]:
    """Return bounded recovery concepts visible in current receptor state."""
    return sorted(_semantic_affordances(_state_semantic_text(state)))


@lru_cache(maxsize=8192)
def _lexical_tokens(text: str) -> frozenset[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) <= 1:
            continue
        tokens.add(token)
        for part in token.split("_"):
            if len(part) > 1:
                tokens.add(part)
    return frozenset(tokens)


def _overlap_features_from_tokens(
    candidate_text: str,
    visible_tokens: set[str],
) -> tuple[float, float, float]:
    candidate_tokens = _lexical_tokens(candidate_text)
    if not candidate_tokens or not visible_tokens:
        return 0.0, 0.0, 0.0
    overlap = candidate_tokens & visible_tokens
    overlap_count = min(len(overlap), 12) / 12.0
    candidate_ratio = len(overlap) / max(len(candidate_tokens), 1)
    visible_ratio = len(overlap) / max(len(visible_tokens), 1)
    return float(overlap_count), float(candidate_ratio), float(visible_ratio)


def _overlap_features(candidate_text: str, visible_text: str) -> tuple[float, float, float]:
    return _overlap_features_from_tokens(candidate_text, _lexical_tokens(visible_text))


def _command_slot_numeric_features(
    state: SystemStateFrame,
    *,
    include_failure_matches: bool = True,
    failure_signals: dict[str, bool] | None = None,
) -> list[float]:
    features: list[float] = []
    commands = candidate_commands(state)
    signals = failure_signals or _visible_failure_signals(state)
    visible_text = " ".join(
        str(part)
        for part in [
            state.goal.description,
            state.terminal.stdout_delta,
            state.terminal.stderr_delta,
            state.terminal.last_command,
            " ".join(state.filesystem.dirty_files),
            " ".join(state.filesystem.changed_paths),
        ]
        if part is not None
    )
    visible_tokens = _lexical_tokens(visible_text)
    for index in range(MAX_CANDIDATE_SLOTS):
        command = commands[index].lower() if index < len(commands) else ""
        command_is_pytest = "pytest" in command
        command_updates_snapshot = "snapshot" in command and "update" in command
        command_installs_dependency = (
            "pip install" in command or "requirements" in command or "install" in command
        )
        overlap_count, candidate_overlap, visible_overlap = _overlap_features_from_tokens(
            command,
            visible_tokens,
        )
        features.extend(
            [
                1.0 if command else 0.0,
                1.0 if command_is_pytest else 0.0,
                1.0 if command_updates_snapshot else 0.0,
                1.0 if command_installs_dependency else 0.0,
                1.0
                if include_failure_matches and signals["snapshot"] and command_updates_snapshot
                else 0.0,
                1.0
                if include_failure_matches
                and signals["dependency"]
                and command_installs_dependency
                else 0.0,
                1.0
                if include_failure_matches
                and signals["assertion"]
                and command_is_pytest
                and not command_updates_snapshot
                else 0.0,
                overlap_count,
                candidate_overlap,
                visible_overlap,
            ]
        )
    return features


def command_failure_match_scores(state: SystemStateFrame) -> list[float]:
    """Score candidate commands by bounded visible failure semantics."""
    state_affordances = set(semantic_affordance_names(state))
    scores: list[float] = []
    for command in candidate_commands(state):
        command_affordances = _semantic_affordances(command)
        scores.append(float(len(state_affordances & command_affordances)))
    return scores


def _file_slot_numeric_features(state: SystemStateFrame) -> list[float]:
    features: list[float] = []
    files = candidate_files(state)
    dirty = set(state.filesystem.dirty_files)
    changed = set(state.filesystem.changed_paths)
    watched = set(state.filesystem.watched_paths + state.goal.watched_paths)
    visible_text = " ".join(
        str(part)
        for part in [
            state.goal.description,
            state.terminal.stdout_delta,
            state.terminal.stderr_delta,
            state.terminal.last_command,
        ]
        if part is not None
    )
    visible_tokens = _lexical_tokens(visible_text)
    for index in range(MAX_CANDIDATE_SLOTS):
        file_target = files[index] if index < len(files) else ""
        overlap_count, candidate_overlap, _visible_overlap = _overlap_features_from_tokens(
            file_target,
            visible_tokens,
        )
        features.extend(
            [
                1.0 if file_target in dirty else 0.0,
                1.0 if file_target in changed else 0.0,
                1.0 if file_target in watched else 0.0,
                overlap_count,
                candidate_overlap,
            ]
        )
    return features


def valid_action_mask(state: SystemStateFrame) -> np.ndarray:
    """Return state-affordance constraints without consulting oracle labels."""
    mask = np.ones(len(ACTION_ORDER), dtype=np.float32)
    if not candidate_commands(state):
        mask[action_to_index(ActionType.RUN_COMMAND)] = 0.0
    if not candidate_files(state):
        mask[action_to_index(ActionType.READ_FILE)] = 0.0
    if state.process.status == ProcessStatus.EXITED:
        mask[action_to_index(ActionType.STOP_PROCESS)] = 0.0
    if not state.safety.dangerous_command_detected:
        mask[action_to_index(ActionType.BLOCK)] = 0.0
    refresh_signal_visible = (
        state.filesystem.external_change_detected
        or state.filesystem.stale_cache_detected
        or state.filesystem.conflict_detected
    )
    if not refresh_signal_visible:
        mask[action_to_index(ActionType.REFRESH_STATE)] = 0.0
    if not (state.terminal.stdout_delta or state.terminal.stdout_lines):
        mask[action_to_index(ActionType.READ_STDOUT)] = 0.0
    if not (state.terminal.stderr_delta or state.terminal.stderr_lines):
        mask[action_to_index(ActionType.READ_STDERR)] = 0.0
    return mask


@dataclass(slots=True)
class StateVectorizer:
    hash_bins: int = 256
    structured: bool = True
    numeric_features: bool = True
    include_route_features: bool = True
    include_task_features: bool = True
    include_failure_signal_features: bool = True
    include_slot_semantic_features: bool = True

    @property
    def numeric_dim(self) -> int:
        return (
            31
            + len(ROUTE_ORDER)
            + len(TaskType)
            + 3
            + (MAX_CANDIDATE_SLOTS * COMMAND_SLOT_FEATURES)
            + (MAX_CANDIDATE_SLOTS * FILE_SLOT_FEATURES)
        )

    @property
    def vector_dim(self) -> int:
        return (self.numeric_dim if self.numeric_features else 0) + self.hash_bins

    def world_model_target_mask(self) -> np.ndarray:
        """Select endogenous structured state features for dynamics prediction.

        Hashed text and stochastic runtime telemetry remain available to the
        policy and full-state novelty measurement, but are not treated as
        deterministic action effects.
        """
        mask = np.zeros(self.vector_dim, dtype=np.float32)
        if self.numeric_features:
            mask[: self.numeric_dim] = 1.0
            for index in _STOCHASTIC_TELEMETRY_INDICES:
                mask[index] = 0.0
        return mask

    def vectorize_state(self, state: SystemStateFrame) -> np.ndarray:
        numeric: list[float] = []
        if self.numeric_features:
            numeric.extend(
                [
                    state.time.runtime_ms / 60000.0,
                    state.time.since_last_output_ms / 60000.0,
                    state.time.since_last_state_change_ms / 60000.0,
                    state.process.cpu_percent / 100.0,
                    state.process.memory_mb / 4096.0,
                    state.process.runtime_ms / 60000.0,
                    float(state.process.exit_code or 0) / 255.0,
                    1.0 if state.process.waiting_for_input else 0.0,
                    1.0 if state.process.interrupted else 0.0,
                    1.0 if state.process.has_children else 0.0,
                    1.0 if state.process.resource_alert else 0.0,
                    1.0 if state.terminal.prompt_visible else 0.0,
                    1.0 if state.terminal.input_requested else 0.0,
                    state.terminal.stdout_lines / 50.0,
                    state.terminal.stderr_lines / 50.0,
                    float(bool(state.terminal.stdout_delta)),
                    float(bool(state.terminal.stderr_delta)),
                    1.0 if state.filesystem.external_change_detected else 0.0,
                    1.0 if state.filesystem.stale_cache_detected else 0.0,
                    1.0 if state.filesystem.conflict_detected else 0.0,
                    len(state.filesystem.changed_paths) / 8.0,
                    len(state.filesystem.dirty_files) / 8.0,
                    1.0 if state.user.manual_input_active else 0.0,
                    1.0 if state.user.confirmation_required else 0.0,
                    1.0 if state.user.user_block_requested else 0.0,
                    1.0 if state.safety.dangerous_command_detected else 0.0,
                    1.0 if state.safety.command_candidate else 0.0,
                ]
            )
            numeric.extend(
                [
                    1.0 if state.process.status.value == label else 0.0
                    for label in ["running", "sleeping", "blocked", "exited"]
                ]
            )
            if self.include_route_features:
                numeric.extend(
                    [
                        1.0 if primary_route_for_task(state.goal.task_type) == route else 0.0
                        for route in ROUTE_ORDER
                    ]
                )
            else:
                numeric.extend([0.0 for _ in ROUTE_ORDER])
            if self.include_task_features:
                numeric.extend(
                    [
                        1.0 if state.goal.task_type == task_type else 0.0
                        for task_type in TaskType
                    ]
                )
            else:
                numeric.extend([0.0 for _ in TaskType])
            failure_signals = _visible_failure_signals(state)
            if self.include_failure_signal_features:
                numeric.extend(
                    [
                        1.0 if failure_signals["snapshot"] else 0.0,
                        1.0 if failure_signals["dependency"] else 0.0,
                        1.0 if failure_signals["assertion"] else 0.0,
                    ]
                )
            else:
                numeric.extend([0.0, 0.0, 0.0])
            if self.include_slot_semantic_features:
                numeric.extend(
                    _command_slot_numeric_features(
                        state,
                        include_failure_matches=self.include_failure_signal_features,
                        failure_signals=failure_signals,
                    )
                )
                numeric.extend(_file_slot_numeric_features(state))
            else:
                numeric.extend([0.0 for _ in range(MAX_CANDIDATE_SLOTS * COMMAND_SLOT_FEATURES)])
                numeric.extend([0.0 for _ in range(MAX_CANDIDATE_SLOTS * FILE_SLOT_FEATURES)])
            if len(numeric) != self.numeric_dim:
                raise ValueError(f"Unexpected numeric feature size: {len(numeric)}")

        hash_features = np.zeros(self.hash_bins, dtype=np.float32)
        text = serialize_state_as_text(state, include_internal_hints=False)
        if self.structured:
            tokens = text.replace("\n", " ").split()
        else:
            tokens = [text]
        if self.hash_bins > 0:
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.hash_bins
                hash_features[index] += 1.0
            if tokens:
                hash_features /= max(len(tokens), 1)
        if self.numeric_features:
            return np.concatenate([np.array(numeric, dtype=np.float32), hash_features])
        return hash_features


def salience_target(record: TrajectoryRecord) -> float:
    state = record.state
    action = record.action.type if record.action else ActionType.WAIT
    return float(
        action not in (ActionType.WAIT, ActionType.DONE)
        or state.filesystem.external_change_detected
        or state.safety.dangerous_command_detected
        or bool(state.terminal.stderr_delta.strip())
    )


def urgency_target(record: TrajectoryRecord) -> float:
    if record.action is None:
        return 0.0
    if record.action.type in (ActionType.BLOCK, ActionType.STOP_PROCESS):
        return 1.0
    if record.action.type in (ActionType.REFRESH_STATE, ActionType.READ_STDERR):
        return 0.7
    if record.action.type in (ActionType.ASK_USER, ActionType.READ_FILE):
        return 0.5
    return 0.2


def risk_target(record: TrajectoryRecord) -> float:
    state = record.state
    if state.safety.dangerous_command_detected:
        return 1.0
    visible_text = f"{state.goal.description} {state.terminal.stdout_delta} {state.terminal.stderr_delta}".lower()
    if "permission" in visible_text or "access denied" in visible_text:
        return 0.8
    if record.action and record.action.type == ActionType.STOP_PROCESS:
        return 0.7
    return 0.2


def route_target(record: TrajectoryRecord) -> int:
    return route_to_index(primary_route_for_task(record.goal.task_type))


def command_slot_target(record: TrajectoryRecord) -> int:
    if not record.action or record.action.type != ActionType.RUN_COMMAND or not record.action.command:
        return -100
    commands = candidate_commands(record.state)
    try:
        return commands.index(record.action.command)
    except ValueError:
        return -100


def file_slot_target(record: TrajectoryRecord) -> int:
    if not record.action or record.action.type != ActionType.READ_FILE or not record.action.file_target:
        return -100
    files = candidate_files(record.state)
    try:
        return files.index(record.action.file_target)
    except ValueError:
        return -100


def delta_norm_target(record: TrajectoryRecord, vectorizer: StateVectorizer) -> float:
    state_vec = vectorizer.vectorize_state(record.state)
    next_vec = vectorizer.vectorize_state(record.next_state)
    return float(np.linalg.norm(next_vec - state_vec) / math.sqrt(len(state_vec)))


def resolve_structured_action(
    action_index: int,
    command_index: int,
    file_index: int,
    state: SystemStateFrame,
    confidence: float,
) -> ActionDecision:
    action_type = ACTION_ORDER[action_index]
    command = None
    file_target = None
    if action_type == ActionType.RUN_COMMAND:
        commands = candidate_commands(state)
        if commands:
            command = commands[max(min(command_index, len(commands) - 1), 0)]
        else:
            action_type = ActionType.WAIT
    if action_type == ActionType.READ_FILE:
        files = candidate_files(state)
        if files:
            file_target = files[max(min(file_index, len(files) - 1), 0)]
        else:
            action_type = ActionType.WAIT
    return ActionDecision(
        type=action_type,
        command=command,
        file_target=file_target,
        confidence=max(min(confidence, 1.0), 0.0),
    )
