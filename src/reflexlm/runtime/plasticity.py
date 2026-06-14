from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reflexlm.models.features import MAX_CANDIDATE_SLOTS, candidate_commands
from reflexlm.schema import SystemStateFrame


VERIFIED_FEEDBACK_SOURCES = {
    "post_execution_verifier",
    "rollback_verifier",
    "sealed_evaluator",
}
PLASTICITY_CONTROLS = {"normal", "erased", "wrong"}


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_values(values: list[str]) -> list[str]:
    return sorted({str(value).replace("\\", "/").strip().lower() for value in values if value})


def synaptic_state_pattern(state: SystemStateFrame) -> dict[str, Any]:
    """Return a non-answer environment pattern suitable for routine reuse."""

    evidence = state.runtime_evidence
    return {
        "task_type": state.goal.task_type.value,
        "process_status": state.process.status.value,
        "exit_code": state.process.exit_code,
        "waiting_for_input": state.process.waiting_for_input,
        "resource_alert": state.process.resource_alert,
        "changed_files": _normalized_values(evidence.changed_files or state.filesystem.changed_paths),
        "watched_files": _normalized_values(evidence.watched_files or state.filesystem.watched_paths),
        "repair_modes": _normalized_values(evidence.repair_modes),
        "descriptor_operation": evidence.descriptor_operation,
        "descriptor_template": evidence.descriptor_template,
        "external_change_detected": state.filesystem.external_change_detected,
        "stale_cache_detected": state.filesystem.stale_cache_detected,
        "conflict_detected": state.filesystem.conflict_detected,
    }


def state_pattern_key(state: SystemStateFrame) -> str:
    return _stable_hash(synaptic_state_pattern(state))


def candidate_identity(command: str) -> str:
    return _stable_hash({"bounded_command": command.strip()})


@dataclass(slots=True)
class PlasticConnection:
    command: str
    weight: float = 0.0
    verified_successes: int = 0
    verified_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "weight": self.weight,
            "verified_successes": self.verified_successes,
            "verified_failures": self.verified_failures,
        }


@dataclass(slots=True)
class SynapticPlasticityMemory:
    learning_rate: float = 1.0
    failure_penalty: float = 1.0
    max_abs_weight: float = 8.0
    min_recall_weight: float = 0.5
    connections: dict[str, dict[str, PlasticConnection]] = field(default_factory=dict)
    feedback_events: int = 0
    rejected_feedback_events: int = 0
    last_feedback: dict[str, Any] = field(default_factory=dict)
    control: str = "normal"

    def predict(self, state: SystemStateFrame) -> dict[str, Any]:
        if self.control not in PLASTICITY_CONTROLS:
            raise ValueError(f"unsupported plasticity control: {self.control}")
        commands = candidate_commands(state)[:MAX_CANDIDATE_SLOTS]
        pattern_key = state_pattern_key(state)
        pattern_connections = self.connections.get(pattern_key, {})
        scores = [
            float(pattern_connections.get(candidate_identity(command), PlasticConnection(command)).weight)
            for command in commands
        ]
        padded_scores = scores + [0.0] * (MAX_CANDIDATE_SLOTS - len(scores))
        best = max(scores) if scores else 0.0
        sorted_scores = sorted(scores, reverse=True)
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        unique_best = best >= self.min_recall_weight and scores.count(best) == 1
        selected_slot = scores.index(best) if unique_best else None
        wrong_memory_injected = False
        if self.control == "erased":
            scores = [0.0 for _ in scores]
            padded_scores = scores + [0.0] * (MAX_CANDIDATE_SLOTS - len(scores))
            selected_slot = None
            unique_best = False
            best = 0.0
            second = 0.0
        elif self.control == "wrong" and selected_slot is not None and len(commands) > 1:
            wrong_slot = (selected_slot + 1) % len(commands)
            wrong_score = max(best, self.min_recall_weight)
            scores = [
                wrong_score if index == wrong_slot else 0.0
                for index in range(len(commands))
            ]
            padded_scores = scores + [0.0] * (MAX_CANDIDATE_SLOTS - len(scores))
            selected_slot = wrong_slot
            best = wrong_score
            second = 0.0
            wrong_memory_injected = True
        selected_connection = (
            pattern_connections.get(candidate_identity(commands[selected_slot]))
            if selected_slot is not None
            else None
        )
        observations = 0
        expected_success = 0.5
        if selected_connection is not None:
            observations = (
                selected_connection.verified_successes
                + selected_connection.verified_failures
            )
            if observations:
                expected_success = selected_connection.verified_successes / observations
        return {
            "pattern_key": pattern_key,
            "selected_slot": selected_slot,
            "scores": padded_scores,
            "margin": float(best - second if unique_best else 0.0),
            "confidence": float(min(max(best / self.max_abs_weight, 0.0), 1.0))
            if unique_best
            else 0.0,
            "observations": observations,
            "expected_success": expected_success,
            "memory_hit": selected_slot is not None,
            "control": self.control,
            "wrong_memory_injected": wrong_memory_injected,
        }

    def reference(self, state: SystemStateFrame) -> dict[str, float]:
        prediction = self.predict(state)
        return {
            **{
                f"command_identity_slot:{index}": float(prediction["scores"][index])
                for index in range(MAX_CANDIDATE_SLOTS)
            },
            "command_identity_margin": float(prediction["margin"]),
            "command_identity_confidence": float(prediction["confidence"]),
        }

    def observe_feedback(
        self,
        state: SystemStateFrame,
        *,
        command: str,
        verified_success: bool,
        verifier: str,
    ) -> dict[str, Any]:
        commands = candidate_commands(state)
        if verifier not in VERIFIED_FEEDBACK_SOURCES or command not in commands:
            self.rejected_feedback_events += 1
            self.last_feedback = {
                "accepted": False,
                "verifier": verifier,
                "command_allowlisted": command in commands,
            }
            return dict(self.last_feedback)
        prediction = self.predict(state)
        pattern_key = str(prediction["pattern_key"])
        command_key = candidate_identity(command)
        pattern_connections = self.connections.setdefault(pattern_key, {})
        connection = pattern_connections.setdefault(
            command_key,
            PlasticConnection(command=command),
        )
        expected_success = float(prediction["expected_success"])
        observed_success = 1.0 if verified_success else 0.0
        prediction_error = abs(observed_success - expected_success)
        if verified_success:
            connection.weight = min(
                self.max_abs_weight,
                connection.weight + self.learning_rate,
            )
            connection.verified_successes += 1
        else:
            connection.weight = max(
                -self.max_abs_weight,
                connection.weight - self.failure_penalty,
            )
            connection.verified_failures += 1
        self.feedback_events += 1
        self.last_feedback = {
            "accepted": True,
            "verifier": verifier,
            "verified_success": verified_success,
            "pattern_key": pattern_key,
            "command_identity": command_key,
            "updated_weight": connection.weight,
            "prediction_error": prediction_error,
            "feedback_events": self.feedback_events,
        }
        return dict(self.last_feedback)

    def clear(self) -> None:
        self.connections.clear()
        self.feedback_events = 0
        self.rejected_feedback_events = 0
        self.last_feedback = {}

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "reflexlm.synaptic_plasticity.v1",
            "learning_rate": self.learning_rate,
            "failure_penalty": self.failure_penalty,
            "max_abs_weight": self.max_abs_weight,
            "min_recall_weight": self.min_recall_weight,
            "feedback_events": self.feedback_events,
            "rejected_feedback_events": self.rejected_feedback_events,
            "connections": {
                pattern_key: {
                    command_key: connection.to_dict()
                    for command_key, connection in sorted(pattern.items())
                }
                for pattern_key, pattern in sorted(self.connections.items())
            },
            "last_feedback": dict(self.last_feedback),
            "control": self.control,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.snapshot(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SynapticPlasticityMemory":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if payload.get("schema_version") != "reflexlm.synaptic_plasticity.v1":
            raise ValueError("unsupported synaptic plasticity memory schema")
        memory = cls(
            learning_rate=float(payload.get("learning_rate", 1.0)),
            failure_penalty=float(payload.get("failure_penalty", 1.0)),
            max_abs_weight=float(payload.get("max_abs_weight", 8.0)),
            min_recall_weight=float(payload.get("min_recall_weight", 0.5)),
        )
        memory.feedback_events = int(payload.get("feedback_events", 0))
        memory.rejected_feedback_events = int(payload.get("rejected_feedback_events", 0))
        memory.last_feedback = dict(payload.get("last_feedback") or {})
        memory.control = str(payload.get("control", "normal"))
        for pattern_key, rows in (payload.get("connections") or {}).items():
            memory.connections[str(pattern_key)] = {
                str(command_key): PlasticConnection(
                    command=str(row.get("command") or ""),
                    weight=float(row.get("weight", 0.0)),
                    verified_successes=int(row.get("verified_successes", 0)),
                    verified_failures=int(row.get("verified_failures", 0)),
                )
                for command_key, row in rows.items()
                if isinstance(row, dict)
            }
        return memory
