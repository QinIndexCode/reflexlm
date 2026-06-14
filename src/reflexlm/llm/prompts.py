from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from reflexlm.models.features import (
    ACTION_ORDER,
    candidate_commands,
    candidate_files,
    serialize_state_as_text,
    valid_action_mask,
)
from reflexlm.llm.candidate_features import redact_structured_command_identity_text
from reflexlm.schema import ActionDecision, SystemStateFrame


PHASE2_PROMPT_STYLES = ("prompt_only", "react", "synapse_augmented", "nsi_state_v2")
SYNAPSE_REQUIRED_PROMPT_STYLES = ("synapse_augmented", "nsi_state_v2")


@dataclass(slots=True)
class SynapseSummary:
    route_name: str
    salience: float
    risk: float
    prediction_error: float
    confidence: float
    reflex_action: str | None = None
    reflex_command: str | None = None
    reflex_file_target: str | None = None

    def to_prompt_lines(self) -> list[str]:
        lines = [
            f"route_hint={self.route_name}",
            f"salience={self.salience:.4f}",
            f"risk={self.risk:.4f}",
            f"prediction_error={self.prediction_error:.4f}",
            f"nsi_confidence={self.confidence:.4f}",
        ]
        if self.reflex_action is not None:
            lines.extend(
                [
                    f"reflex_action={self.reflex_action}",
                    f"reflex_command={self.reflex_command or ''}",
                    f"reflex_file_target={self.reflex_file_target or ''}",
                ]
            )
        return lines

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "route_name": self.route_name,
            "salience": round(self.salience, 6),
            "risk": round(self.risk, 6),
            "prediction_error": round(self.prediction_error, 6),
            "confidence": round(self.confidence, 6),
        }
        if self.reflex_action is not None:
            payload.update(
                {
                    "reflex_action": self.reflex_action,
                    "reflex_command": self.reflex_command,
                    "reflex_file_target": self.reflex_file_target,
                }
            )
        return payload


def canonical_action_json(action: ActionDecision) -> str:
    return json.dumps(
        {
            "action": action.type.value,
            "command": action.command,
            "file_target": action.file_target,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def phase2_system_prompt(*, prompt_style: str) -> str:
    if prompt_style == "react":
        return (
            "You are a bounded Phase 2 system-state policy. "
            "Reason internally, but return only one JSON object with keys action, command, file_target. "
            "Never invent commands or files outside the provided candidates."
        )
    if prompt_style == "nsi_state_v2":
        return (
            "You are the cortex path inside a single LLM-native nervous interface. "
            "Receptor state and synaptic signals are low-level sensory inputs, not optional context. "
            "Use the motor schema constraints exactly, choose one bounded action, and return only one JSON "
            "object with keys action, command, file_target. Never invent commands or files outside the "
            "provided candidates."
        )
    return (
        "You are a bounded Phase 2 system-state policy. "
        "Return only one JSON object with keys action, command, file_target. "
        "Never invent commands or files outside the provided candidates."
    )


def _serialize_receptor_state_v2(state: SystemStateFrame) -> list[str]:
    """Model-visible receptor state; excludes task ids and hidden recovery hints."""
    return [
        f"goal_description={state.goal.description}",
        f"runtime_ms={state.time.runtime_ms}",
        f"since_last_output_ms={state.time.since_last_output_ms}",
        f"since_last_state_change_ms={state.time.since_last_state_change_ms}",
        f"process_status={state.process.status.value}",
        f"exit_code={state.process.exit_code}",
        f"cpu_percent={state.process.cpu_percent:.2f}",
        f"memory_mb={state.process.memory_mb:.2f}",
        f"waiting_for_input={state.process.waiting_for_input}",
        f"resource_alert={state.process.resource_alert}",
        f"stdout_delta={redact_structured_command_identity_text(state.terminal.stdout_delta)}",
        f"stderr_delta={redact_structured_command_identity_text(state.terminal.stderr_delta)}",
        f"stdout_lines={state.terminal.stdout_lines}",
        f"stderr_lines={state.terminal.stderr_lines}",
        f"prompt_visible={state.terminal.prompt_visible}",
        f"input_requested={state.terminal.input_requested}",
        f"last_output_channel={state.terminal.last_output_channel or ''}",
        f"last_command={state.terminal.last_command or ''}",
        f"dirty_files={','.join(state.filesystem.dirty_files)}",
        f"changed_paths={','.join(state.filesystem.changed_paths)}",
        f"watched_paths={','.join(state.filesystem.watched_paths)}",
        f"external_change_detected={state.filesystem.external_change_detected}",
        f"stale_cache_detected={state.filesystem.stale_cache_detected}",
        f"conflict_detected={state.filesystem.conflict_detected}",
        f"manual_input_active={state.user.manual_input_active}",
        f"confirmation_required={state.user.confirmation_required}",
        f"user_block_requested={state.user.user_block_requested}",
        f"dangerous_command_detected={state.safety.dangerous_command_detected}",
        f"command_candidate={state.safety.command_candidate or ''}",
        f"risk_label={state.safety.risk_label or ''}",
    ]


def _legal_action_mask_lines(state: SystemStateFrame) -> list[str]:
    mask = valid_action_mask(state)
    return [
        f"{action.value}={int(mask[index] > 0.0)}"
        for index, action in enumerate(ACTION_ORDER)
    ]


def _build_nsi_state_v2_prompt(
    state: SystemStateFrame,
    *,
    synapse_summary: SynapseSummary | None,
) -> str:
    commands = candidate_commands(state)
    files = candidate_files(state)
    lines = [
        "Interface thesis: receptor -> synaptic state -> reflex layer -> salience router -> cortex path -> motor schema.",
        "The cortex path may override the reflex suggestion only when visible state, candidate slots, or risk/salience signals justify it.",
        "",
        "Motor action space:",
        "WAIT, READ_STDOUT, READ_STDERR, READ_FILE, RUN_COMMAND, STOP_PROCESS, ASK_USER, REFRESH_STATE, BLOCK, DONE.",
        "",
        "Legal action mask:",
    ]
    lines.extend(_legal_action_mask_lines(state))
    lines.extend(
        [
            "",
            "Receptor state:",
        ]
    )
    lines.extend(_serialize_receptor_state_v2(state))
    lines.extend(
        [
            "",
            "Candidate commands:",
        ]
    )
    lines.extend([f"- {command}" for command in commands] or ["- <none>"])
    lines.append("")
    lines.append("Candidate files:")
    lines.extend([f"- {file_target}" for file_target in files] or ["- <none>"])
    lines.append("")
    lines.append("Synaptic state:")
    if synapse_summary is None:
        lines.append("- <none>")
    else:
        lines.extend(synapse_summary.to_prompt_lines())
    lines.extend(
        [
            "",
            "Motor schema constraints:",
            "- RUN_COMMAND must use exactly one candidate command.",
            "- READ_FILE must use exactly one candidate file.",
            "- If the legal action mask marks an action as 0, do not choose it.",
            "- Return only JSON.",
        ]
    )
    return "\n".join(lines)


def build_phase2_user_prompt(
    state: SystemStateFrame,
    *,
    prompt_style: str = "prompt_only",
    synapse_summary: SynapseSummary | None = None,
) -> str:
    if prompt_style == "nsi_state_v2":
        return _build_nsi_state_v2_prompt(state, synapse_summary=synapse_summary)
    commands = candidate_commands(state)
    files = candidate_files(state)
    lines = [
        "Action space: WAIT, READ_STDOUT, READ_STDERR, READ_FILE, RUN_COMMAND, STOP_PROCESS, ASK_USER, REFRESH_STATE, BLOCK, DONE.",
        "",
        "Current state:",
        serialize_state_as_text(state, include_internal_hints=False),
        "",
        "Candidate commands:",
    ]
    lines.extend([f"- {command}" for command in commands] or ["- <none>"])
    lines.append("")
    lines.append("Candidate files:")
    lines.extend([f"- {file_target}" for file_target in files] or ["- <none>"])
    if synapse_summary is not None:
        lines.append("")
        lines.append("Synapse summary:")
        lines.extend(synapse_summary.to_prompt_lines())
    lines.append("")
    lines.append("Return only JSON.")
    return "\n".join(lines)
