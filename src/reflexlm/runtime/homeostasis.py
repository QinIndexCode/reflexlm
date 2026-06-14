from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reflexlm.schema import ActionType, InternalTarget


SIDE_EFFECT_ACTIONS = {
    ActionType.RUN_COMMAND,
    ActionType.STOP_PROCESS,
}
HOMEOSTATIC_CONTROLLER_SCHEMA = "reflexlm.homeostatic_synaptic_control.v4"
HOMEOSTATIC_PERSISTENT_STATE_SCHEMA = "reflexlm.homeostatic_persistent_state.v3"
PERSISTENT_STATE_KEYS = {
    "active_surprise_wake_threshold",
    "lifetime_failure_sensitivity_adaptations",
    "lifetime_set_point_recovery_adaptations",
    "adaptive_threshold_preserved_resets",
    "adaptive_threshold_reset_decay_events",
}


def _unit(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _quantized_lower(value: float, resolution: float) -> float:
    value = _unit(value)
    resolution = _unit(resolution)
    if resolution == 0.0:
        return value
    return max(0.0, math.floor((value + 1e-12) / resolution) * resolution)


def _quantized_upper(value: float, resolution: float) -> float:
    value = _unit(value)
    resolution = _unit(resolution)
    if resolution == 0.0:
        return value
    return min(1.0, math.ceil((value - 1e-12) / resolution) * resolution)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _auth_key_bytes(key: str | bytes | None) -> bytes | None:
    if key is None:
        return None
    if isinstance(key, bytes):
        return key
    if not isinstance(key, str) or key == "":
        raise ValueError("homeostatic persistent state authenticity key is invalid")
    return key.encode("utf-8")


def _unsigned_persistent_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in artifact.items()
        if key not in {"integrity_sha256", "authenticator"}
    }


def _hmac_digest(payload: dict[str, Any], key: bytes) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _persistent_state_authenticator(
    artifact: dict[str, Any],
    *,
    authenticity_key: str | bytes | None = None,
) -> dict[str, object]:
    unsigned = _unsigned_persistent_artifact(artifact)
    key = _auth_key_bytes(authenticity_key)
    if key is None:
        return {
            "algorithm": "sha256",
            "digest": _stable_hash(unsigned),
            "key_fingerprint_sha256": None,
        }
    return {
        "algorithm": "hmac-sha256",
        "digest": _hmac_digest(unsigned, key),
        "key_fingerprint_sha256": hashlib.sha256(key).hexdigest(),
    }


@dataclass(slots=True)
class HomeostaticControlConfig:
    """Thresholds for task-label-independent continuous synaptic control."""

    ema_decay: float = 0.80
    surprise_wake_threshold: float = 0.55
    online_failure_sensitivity_enabled: bool = True
    failure_sensitivity_rate: float = 0.25
    failure_sensitivity_hysteresis: float = 0.005
    decision_signal_resolution: float = 0.005
    set_point_recovery_rate: float = 0.05
    minimum_surprise_wake_threshold: float = 0.05
    preserve_adaptive_threshold_across_reset: bool = False
    cross_episode_threshold_retention: float = 0.95
    risk_inhibition_threshold: float = 0.85
    persistent_failure_wake_threshold: float = 0.65
    low_salience_threshold: float = 0.20
    low_prediction_error_threshold: float = 0.12
    habituation_repetitions: int = 2


@dataclass(slots=True)
class HomeostaticControlDecision:
    target: InternalTarget
    reason: str
    inhibited: bool = False
    habituated: bool = False
    wake_pressure: float = 0.0
    risk_pressure: float = 0.0
    failure_pressure: float = 0.0
    repeated_action_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "target": self.target.value,
        }


class HomeostaticSynapticController:
    """Persistent cross-frame controller driven by learned and observed signals.

    The controller cannot authorize an action. It can only preserve a reflex,
    wake a bounded cortex, or inhibit/habituate a proposed motor action.
    """

    def __init__(self, config: HomeostaticControlConfig | None = None) -> None:
        self.config = config or HomeostaticControlConfig()
        self.reset()

    def _clear_episode_state(self) -> None:
        self.ema_salience = 0.0
        self.ema_risk = 0.0
        self.ema_prediction_error = 0.0
        self.ema_failure = 0.0
        self.observations = 0
        self.last_action: ActionType | None = None
        self.repeated_action_count = 0
        self.wake_events = 0
        self.inhibition_events = 0
        self.habituation_events = 0
        self.failure_sensitivity_adaptations = 0
        self.set_point_recovery_adaptations = 0
        self.last_threshold_adaptation: dict[str, object] = {}

    def reset(self) -> None:
        baseline = _unit(self.config.surprise_wake_threshold)
        had_prior_threshold = hasattr(self, "active_surprise_wake_threshold")
        preserve_threshold = (
            self.config.online_failure_sensitivity_enabled
            and self.config.preserve_adaptive_threshold_across_reset
            and had_prior_threshold
        )
        prior_threshold = getattr(self, "active_surprise_wake_threshold", baseline)
        prior_lifetime_failure_adaptations = getattr(
            self,
            "lifetime_failure_sensitivity_adaptations",
            0,
        )
        prior_lifetime_recovery_adaptations = getattr(
            self,
            "lifetime_set_point_recovery_adaptations",
            0,
        )
        prior_preserved_resets = getattr(
            self,
            "adaptive_threshold_preserved_resets",
            0,
        )
        prior_reset_decay_events = getattr(
            self,
            "adaptive_threshold_reset_decay_events",
            0,
        )
        retained_threshold = max(
            min(baseline, _unit(self.config.minimum_surprise_wake_threshold)),
            min(
                baseline
                + _unit(self.config.cross_episode_threshold_retention)
                * (prior_threshold - baseline),
                baseline,
            ),
        )
        self.active_surprise_wake_threshold = (
            retained_threshold if preserve_threshold else baseline
        )
        self._clear_episode_state()
        self.lifetime_failure_sensitivity_adaptations = (
            prior_lifetime_failure_adaptations if preserve_threshold else 0
        )
        self.lifetime_set_point_recovery_adaptations = (
            prior_lifetime_recovery_adaptations if preserve_threshold else 0
        )
        self.adaptive_threshold_preserved_resets = (
            prior_preserved_resets + 1 if preserve_threshold else 0
        )
        self.adaptive_threshold_reset_decay_events = (
            prior_reset_decay_events + int(retained_threshold > prior_threshold)
            if preserve_threshold
            else 0
        )

    def config_fingerprint(self) -> str:
        return _stable_hash(asdict(self.config))

    def persistent_state_scope_fingerprint(
        self,
        persistence_scope: str | None = None,
    ) -> str:
        return _stable_hash(
            persistence_scope
            or "reflexlm.homeostatic_persistent_state.unscoped"
        )

    def export_persistent_state(
        self,
        *,
        persistence_scope: str | None = None,
        authenticity_key: str | bytes | None = None,
    ) -> dict[str, object]:
        self._require_persistent_memory_enabled()
        artifact: dict[str, object] = {
            "schema_version": HOMEOSTATIC_PERSISTENT_STATE_SCHEMA,
            "controller_schema_version": HOMEOSTATIC_CONTROLLER_SCHEMA,
            "config": asdict(self.config),
            "config_fingerprint": self.config_fingerprint(),
            "scope_fingerprint": self.persistent_state_scope_fingerprint(
                persistence_scope
            ),
            "state": {
                "active_surprise_wake_threshold": (
                    self.active_surprise_wake_threshold
                ),
                "lifetime_failure_sensitivity_adaptations": (
                    self.lifetime_failure_sensitivity_adaptations
                ),
                "lifetime_set_point_recovery_adaptations": (
                    self.lifetime_set_point_recovery_adaptations
                ),
                "adaptive_threshold_preserved_resets": (
                    self.adaptive_threshold_preserved_resets
                ),
                "adaptive_threshold_reset_decay_events": (
                    self.adaptive_threshold_reset_decay_events
                ),
            },
        }
        artifact["integrity_sha256"] = _stable_hash(
            _unsigned_persistent_artifact(artifact)
        )
        artifact["authenticator"] = _persistent_state_authenticator(
            artifact,
            authenticity_key=authenticity_key,
        )
        return artifact

    def save_persistent_state(
        self,
        path: str | Path,
        *,
        persistence_scope: str | None = None,
        authenticity_key: str | bytes | None = None,
    ) -> dict[str, object]:
        artifact = self.export_persistent_state(
            persistence_scope=persistence_scope,
            authenticity_key=authenticity_key,
        )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        temporary.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(output)
        return artifact

    def load_persistent_state_file(
        self,
        path: str | Path,
        *,
        persistence_scope: str | None = None,
        authenticity_key: str | bytes | None = None,
    ) -> dict[str, object]:
        artifact = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        self.load_persistent_state(
            artifact,
            persistence_scope=persistence_scope,
            authenticity_key=authenticity_key,
        )
        return artifact

    def load_persistent_state(
        self,
        artifact: dict[str, Any],
        *,
        persistence_scope: str | None = None,
        authenticity_key: str | bytes | None = None,
    ) -> None:
        self._require_persistent_memory_enabled()
        if not isinstance(artifact, dict):
            raise ValueError("homeostatic persistent state must be an object")
        expected_artifact_keys = {
            "schema_version",
            "controller_schema_version",
            "config",
            "config_fingerprint",
            "scope_fingerprint",
            "state",
            "integrity_sha256",
            "authenticator",
        }
        if set(artifact) != expected_artifact_keys:
            raise ValueError("homeostatic persistent state has unexpected fields")
        if artifact.get("schema_version") != HOMEOSTATIC_PERSISTENT_STATE_SCHEMA:
            raise ValueError("unsupported homeostatic persistent state schema")
        if artifact.get("controller_schema_version") != HOMEOSTATIC_CONTROLLER_SCHEMA:
            raise ValueError("homeostatic controller schema mismatch")
        supplied_integrity = artifact.get("integrity_sha256")
        unsigned_artifact = _unsigned_persistent_artifact(artifact)
        if supplied_integrity != _stable_hash(unsigned_artifact):
            raise ValueError("homeostatic persistent state integrity mismatch")
        authenticator = artifact.get("authenticator")
        if not isinstance(authenticator, dict):
            raise ValueError("homeostatic persistent state authenticator is invalid")
        expected_authenticator = _persistent_state_authenticator(
            artifact,
            authenticity_key=authenticity_key,
        )
        algorithm = authenticator.get("algorithm")
        if algorithm == "hmac-sha256" and authenticity_key is None:
            raise ValueError(
                "homeostatic persistent state hmac authenticator requires "
                "an authenticity key"
            )
        if authenticity_key is not None and algorithm != "hmac-sha256":
            raise ValueError(
                "homeostatic persistent state keyed authenticator required"
            )
        if (
            authenticator.get("algorithm") != expected_authenticator["algorithm"]
            or authenticator.get("key_fingerprint_sha256")
            != expected_authenticator["key_fingerprint_sha256"]
            or not hmac.compare_digest(
                str(authenticator.get("digest", "")),
                str(expected_authenticator["digest"]),
            )
        ):
            raise ValueError(
                "homeostatic persistent state authenticator mismatch"
            )
        artifact_config = artifact.get("config")
        if not isinstance(artifact_config, dict):
            raise ValueError("homeostatic persistent state config must be an object")
        artifact_config_fingerprint = artifact.get("config_fingerprint")
        if artifact_config_fingerprint != _stable_hash(artifact_config):
            raise ValueError("homeostatic persistent state config fingerprint mismatch")
        if artifact_config_fingerprint != self.config_fingerprint():
            raise ValueError("homeostatic persistent state is incompatible with config")
        if artifact.get(
            "scope_fingerprint"
        ) != self.persistent_state_scope_fingerprint(persistence_scope):
            raise ValueError("homeostatic persistent state scope mismatch")
        state = artifact.get("state")
        if not isinstance(state, dict) or set(state) != PERSISTENT_STATE_KEYS:
            raise ValueError("homeostatic persistent state payload is invalid")
        active_threshold = state["active_surprise_wake_threshold"]
        if isinstance(active_threshold, bool) or not isinstance(
            active_threshold,
            (int, float),
        ):
            raise ValueError("active surprise threshold must be numeric")
        active_threshold = float(active_threshold)
        baseline = _unit(self.config.surprise_wake_threshold)
        minimum = min(
            baseline,
            _unit(self.config.minimum_surprise_wake_threshold),
        )
        if (
            not math.isfinite(active_threshold)
            or active_threshold < minimum
            or active_threshold > baseline
        ):
            raise ValueError("active surprise threshold is outside configured bounds")
        count_keys = PERSISTENT_STATE_KEYS - {"active_surprise_wake_threshold"}
        counts: dict[str, int] = {}
        for key in count_keys:
            value = state[key]
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
            counts[key] = value
        if (
            counts["adaptive_threshold_reset_decay_events"]
            > counts["adaptive_threshold_preserved_resets"]
        ):
            raise ValueError("reset decay events cannot exceed preserved resets")
        self._clear_episode_state()
        self.active_surprise_wake_threshold = active_threshold
        self.lifetime_failure_sensitivity_adaptations = counts[
            "lifetime_failure_sensitivity_adaptations"
        ]
        self.lifetime_set_point_recovery_adaptations = counts[
            "lifetime_set_point_recovery_adaptations"
        ]
        self.adaptive_threshold_preserved_resets = counts[
            "adaptive_threshold_preserved_resets"
        ]
        self.adaptive_threshold_reset_decay_events = counts[
            "adaptive_threshold_reset_decay_events"
        ]

    def _require_persistent_memory_enabled(self) -> None:
        if not (
            self.config.online_failure_sensitivity_enabled
            and self.config.preserve_adaptive_threshold_across_reset
        ):
            raise ValueError(
                "persistent homeostatic state requires online adaptation and "
                "cross-episode memory"
            )

    def _adapt_surprise_threshold(
        self,
        *,
        prediction_error: float,
        temporal_observation_available: bool,
        failure_visible: bool,
    ) -> None:
        if (
            not self.config.online_failure_sensitivity_enabled
            or not temporal_observation_available
        ):
            return
        baseline = _unit(self.config.surprise_wake_threshold)
        minimum = min(
            baseline,
            _unit(self.config.minimum_surprise_wake_threshold),
        )
        before = self.active_surprise_wake_threshold
        sensitivity_hysteresis = _unit(self.config.failure_sensitivity_hysteresis)
        resolution = _unit(self.config.decision_signal_resolution)
        error_upper_bound = _quantized_upper(prediction_error, resolution)
        threshold_lower_bound = _quantized_lower(before, resolution)
        if (
            failure_visible
            and error_upper_bound + sensitivity_hysteresis < threshold_lower_bound
        ):
            rate = _unit(self.config.failure_sensitivity_rate)
            target = max(minimum, min(baseline, prediction_error * 0.90))
            after = max(minimum, before + rate * (target - before))
            if after < before:
                self.active_surprise_wake_threshold = after
                self.failure_sensitivity_adaptations += 1
                self.lifetime_failure_sensitivity_adaptations += 1
                self.last_threshold_adaptation = {
                    "reason": "visible_failure_increased_sensitivity",
                    "before": before,
                    "after": after,
                    "target": target,
                    "prediction_error": prediction_error,
                }
            return
        if not failure_visible and before < baseline:
            rate = _unit(self.config.set_point_recovery_rate)
            after = min(baseline, before + rate * (baseline - before))
            if after > before:
                self.active_surprise_wake_threshold = after
                self.set_point_recovery_adaptations += 1
                self.lifetime_set_point_recovery_adaptations += 1
                self.last_threshold_adaptation = {
                    "reason": "stable_outcome_restored_set_point",
                    "before": before,
                    "after": after,
                    "target": baseline,
                    "prediction_error": prediction_error,
                }

    def observe(
        self,
        *,
        proposed_action: ActionType,
        salience: float,
        risk: float,
        prediction_error: float,
        temporal_observation_available: bool,
        hard_dangerous: bool = False,
        receptor_priority: bool = False,
        failure_visible: bool = False,
    ) -> HomeostaticControlDecision:
        salience = _unit(salience)
        risk = _unit(risk)
        prediction_error = _unit(prediction_error)
        failure = float(bool(failure_visible))
        decay = _unit(self.config.ema_decay)
        if self.observations == 0:
            self.ema_salience = salience
            self.ema_risk = risk
            self.ema_prediction_error = prediction_error
            self.ema_failure = failure
        else:
            keep = decay
            learn = 1.0 - keep
            self.ema_salience = keep * self.ema_salience + learn * salience
            self.ema_risk = keep * self.ema_risk + learn * risk
            self.ema_prediction_error = (
                keep * self.ema_prediction_error + learn * prediction_error
            )
            self.ema_failure = keep * self.ema_failure + learn * failure
        self.observations += 1
        self._adapt_surprise_threshold(
            prediction_error=prediction_error,
            temporal_observation_available=temporal_observation_available,
            failure_visible=failure_visible,
        )

        if proposed_action == self.last_action:
            self.repeated_action_count += 1
        else:
            self.last_action = proposed_action
            self.repeated_action_count = 1

        wake_pressure = max(prediction_error, self.ema_prediction_error)
        risk_pressure = max(risk, self.ema_risk)
        failure_pressure = max(failure, self.ema_failure)
        resolution = _unit(self.config.decision_signal_resolution)
        wake_lower_bound = _quantized_lower(wake_pressure, resolution)
        wake_upper_bound = _quantized_upper(wake_pressure, resolution)
        wake_threshold_lower_bound = _quantized_lower(
            self.active_surprise_wake_threshold,
            resolution,
        )
        wake_threshold_upper_bound = _quantized_upper(
            self.active_surprise_wake_threshold,
            resolution,
        )
        risk_upper_bound = _quantized_upper(risk_pressure, resolution)
        risk_threshold_lower_bound = _quantized_lower(
            self.config.risk_inhibition_threshold,
            resolution,
        )
        failure_upper_bound = _quantized_upper(failure_pressure, resolution)
        failure_threshold_lower_bound = _quantized_lower(
            self.config.persistent_failure_wake_threshold,
            resolution,
        )
        if hard_dangerous:
            self.inhibition_events += 1
            return HomeostaticControlDecision(
                target=InternalTarget.INHIBIT,
                reason="hard_safety_inhibition",
                inhibited=True,
                wake_pressure=wake_pressure,
                risk_pressure=risk_pressure,
                failure_pressure=failure_pressure,
                repeated_action_count=self.repeated_action_count,
            )
        if (
            proposed_action in SIDE_EFFECT_ACTIONS
            and risk_upper_bound >= risk_threshold_lower_bound
        ):
            self.inhibition_events += 1
            return HomeostaticControlDecision(
                target=InternalTarget.INHIBIT,
                reason="learned_risk_inhibition",
                inhibited=True,
                wake_pressure=wake_pressure,
                risk_pressure=risk_pressure,
                failure_pressure=failure_pressure,
                repeated_action_count=self.repeated_action_count,
            )
        if (
            temporal_observation_available
            and not receptor_priority
            and proposed_action in {ActionType.WAIT, ActionType.DONE}
            and failure_upper_bound >= failure_threshold_lower_bound
        ):
            self.wake_events += 1
            return HomeostaticControlDecision(
                target=InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX,
                reason="homeostatic_persistent_failure_wake",
                wake_pressure=wake_pressure,
                risk_pressure=risk_pressure,
                failure_pressure=failure_pressure,
                repeated_action_count=self.repeated_action_count,
            )
        if (
            temporal_observation_available
            and not receptor_priority
            and (
                wake_lower_bound > wake_threshold_upper_bound
                or (
                    failure_visible
                    and wake_upper_bound
                    + _unit(self.config.failure_sensitivity_hysteresis)
                    > wake_threshold_lower_bound
                )
            )
        ):
            self.wake_events += 1
            return HomeostaticControlDecision(
                target=InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX,
                reason="homeostatic_surprise_wake",
                wake_pressure=wake_pressure,
                risk_pressure=risk_pressure,
                failure_pressure=failure_pressure,
                repeated_action_count=self.repeated_action_count,
            )
        if (
            temporal_observation_available
            and not receptor_priority
            and proposed_action in SIDE_EFFECT_ACTIONS
            and self.repeated_action_count >= self.config.habituation_repetitions
            and _quantized_upper(salience, resolution)
            <= _quantized_lower(self.config.low_salience_threshold, resolution)
            and _quantized_upper(prediction_error, resolution)
            <= _quantized_lower(
                self.config.low_prediction_error_threshold,
                resolution,
            )
        ):
            self.habituation_events += 1
            return HomeostaticControlDecision(
                target=InternalTarget.REFLEX_MOTOR,
                reason="homeostatic_low_value_habituation",
                habituated=True,
                wake_pressure=wake_pressure,
                risk_pressure=risk_pressure,
                failure_pressure=failure_pressure,
                repeated_action_count=self.repeated_action_count,
            )
        return HomeostaticControlDecision(
            target=InternalTarget.REFLEX_MOTOR,
            reason="homeostatic_reflex_preserved",
            wake_pressure=wake_pressure,
            risk_pressure=risk_pressure,
            failure_pressure=failure_pressure,
            repeated_action_count=self.repeated_action_count,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": HOMEOSTATIC_CONTROLLER_SCHEMA,
            "config": asdict(self.config),
            "config_fingerprint": self.config_fingerprint(),
            "active_surprise_wake_threshold": self.active_surprise_wake_threshold,
            "observations": self.observations,
            "ema_salience": self.ema_salience,
            "ema_risk": self.ema_risk,
            "ema_prediction_error": self.ema_prediction_error,
            "ema_failure": self.ema_failure,
            "last_action": self.last_action.value if self.last_action else None,
            "repeated_action_count": self.repeated_action_count,
            "wake_events": self.wake_events,
            "inhibition_events": self.inhibition_events,
            "habituation_events": self.habituation_events,
            "failure_sensitivity_adaptations": self.failure_sensitivity_adaptations,
            "set_point_recovery_adaptations": self.set_point_recovery_adaptations,
            "lifetime_failure_sensitivity_adaptations": (
                self.lifetime_failure_sensitivity_adaptations
            ),
            "lifetime_set_point_recovery_adaptations": (
                self.lifetime_set_point_recovery_adaptations
            ),
            "adaptive_threshold_preserved_resets": (
                self.adaptive_threshold_preserved_resets
            ),
            "adaptive_threshold_reset_decay_events": (
                self.adaptive_threshold_reset_decay_events
            ),
            "last_threshold_adaptation": dict(self.last_threshold_adaptation),
        }
