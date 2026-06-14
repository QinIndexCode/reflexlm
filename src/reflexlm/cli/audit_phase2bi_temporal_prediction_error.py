from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from reflexlm.eval import SequenceModelPolicy
from reflexlm.llm.hybrid import HybridPolicyConfig, HybridSynapticPolicy
from reflexlm.llm.native_head_policy import NativeHeadPolicy
from reflexlm.llm.prompts import SynapseSummary
from reflexlm.models.features import ACTION_ORDER, ROUTE_ORDER, StateVectorizer
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    FileSystemState,
    GoalSpec,
    InternalTarget,
    ProcessState,
    ProcessStatus,
    SafetyState,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
    UserState,
)


class _IdentityNextStateModel(torch.nn.Module):
    """Mechanism probe: predicts that the next frame should match this frame."""

    def __init__(self, input_dim: int, *, model_error: float) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.input_dim = input_dim
        self.model_error = model_error

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, seq_len, _dim = inputs.shape
        action_logits = torch.zeros(batch, seq_len, len(ACTION_ORDER), device=inputs.device)
        action_logits[..., ACTION_ORDER.index(ActionType.WAIT)] = 4.0
        return {
            "hidden": hidden,
            "action_logits": action_logits,
            "target_logits": torch.zeros(batch, seq_len, len(INTERNAL_TARGET_ORDER), device=inputs.device),
            "route_logits": torch.zeros(batch, seq_len, len(ROUTE_ORDER), device=inputs.device),
            "command_slot_logits": torch.zeros(batch, seq_len, 4, device=inputs.device),
            "file_slot_logits": torch.zeros(batch, seq_len, 4, device=inputs.device),
            "salience": torch.zeros(batch, seq_len, device=inputs.device),
            "risk": torch.zeros(batch, seq_len, device=inputs.device),
            "prediction_error": torch.full(
                (batch, seq_len),
                self.model_error,
                device=inputs.device,
            ),
            "next_state": inputs.clone(),
        }


def _frame(*, changed: bool = False) -> SystemStateFrame:
    return SystemStateFrame(
        time=TimeState(
            tick=1 if changed else 0,
            runtime_ms=60000 if changed else 0,
            since_last_output_ms=60000 if changed else 0,
            since_last_state_change_ms=60000 if changed else 0,
        ),
        goal=GoalSpec(
            task_type=TaskType.PROCESS_HANG,
            description="maintain a bounded terminal process under runtime observation",
            command_allowlist=["pytest -q", "python -m pip install -r requirements.txt"],
        ),
        process=ProcessState(
            status=ProcessStatus.EXITED if changed else ProcessStatus.RUNNING,
            exit_code=255 if changed else None,
            cpu_percent=100.0 if changed else 0.0,
            memory_mb=4096.0 if changed else 0.0,
            runtime_ms=60000 if changed else 0,
            resource_alert=changed,
        ),
        terminal=TerminalState(
            stdout_delta="stable heartbeat" if not changed else "",
            stderr_delta="AssertionError: same test still failing" if changed else "",
            stdout_lines=1 if not changed else 0,
            stderr_lines=50 if changed else 0,
            prompt_visible=changed,
            last_command="pytest -q" if changed else None,
        ),
        filesystem=FileSystemState(
            watched_paths=["tests/test_app.py"],
            changed_paths=["src/app.py"] if changed else [],
            dirty_files=["src/app.py"] if changed else [],
            external_change_detected=changed,
            stale_cache_detected=changed,
        ),
        user=UserState(
            manual_input_active=changed,
            confirmation_required=changed,
            user_block_requested=changed,
        ),
        safety=SafetyState(
            dangerous_command_detected=changed,
            command_candidate="rm -rf /" if changed else None,
        ),
    )


def _hybrid_direct_reflex_allowed(prediction_error: float, *, threshold: float) -> bool:
    policy = HybridSynapticPolicy.__new__(HybridSynapticPolicy)
    policy.hybrid_config = HybridPolicyConfig(
        base_model_name="mechanism-probe",
        confidence_threshold=0.7,
        prediction_error_threshold=threshold,
    )
    return bool(
        policy._use_direct_reflex(
            ActionDecision(type=ActionType.WAIT, confidence=0.95),
            SynapseSummary(
                route_name="terminal_cortex",
                salience=0.1,
                risk=0.1,
                prediction_error=prediction_error,
                confidence=0.95,
                reflex_action=ActionType.WAIT.value,
            ),
        )
    )


def _native_target_for_prediction_error(
    prediction_error: float,
    *,
    temporal_observation_available: bool,
    threshold: float,
) -> tuple[str, str]:
    policy = NativeHeadPolicy.__new__(NativeHeadPolicy)
    policy.prediction_error_escalation_threshold = threshold
    policy.nsi_policy = type(
        "Nsi",
        (),
        {
            "last_call": {
                "prediction_error": prediction_error,
                "temporal_observation_available": temporal_observation_available,
            }
        },
    )()
    target, source = policy._internal_target_with_prediction_error(_frame(changed=False))
    return target.value, source


def audit_phase2bi_temporal_prediction_error(
    *,
    model_error: float = 0.05,
    prediction_error_threshold: float = 0.45,
    min_changed_error_delta: float = 0.10,
) -> dict[str, Any]:
    vectorizer = StateVectorizer(hash_bins=0)
    policy = SequenceModelPolicy(
        _IdentityNextStateModel(vectorizer.vector_dim, model_error=model_error),
        vectorizer,
        policy_label="phase2bi_temporal_prediction_error_probe",
    )

    policy.act(_frame(changed=False))
    first = dict(policy.last_call)
    policy.act(_frame(changed=False))
    stable = dict(policy.last_call)
    policy.act(_frame(changed=True))
    changed = dict(policy.last_call)
    policy.reset()
    policy.act(_frame(changed=True))
    reset_changed = dict(policy.last_call)

    stable_error = float(stable.get("observed_temporal_prediction_error") or 0.0)
    changed_error = float(changed.get("observed_temporal_prediction_error") or 0.0)
    changed_effective = float(changed.get("prediction_error") or 0.0)
    native_high_target, native_high_source = _native_target_for_prediction_error(
        changed_effective,
        temporal_observation_available=True,
        threshold=prediction_error_threshold,
    )
    native_first_target, _native_first_source = _native_target_for_prediction_error(
        1.0,
        temporal_observation_available=False,
        threshold=prediction_error_threshold,
    )
    checks = {
        "first_frame_has_no_temporal_observation": first.get("temporal_observation_available") is False,
        "stable_transition_has_observed_temporal_error": stable.get("temporal_observation_available") is True,
        "stable_transition_error_is_zero": stable_error == 0.0,
        "changed_transition_uses_observed_temporal_error": changed.get("prediction_error_source")
        == "observed_temporal_next_state",
        "changed_transition_error_exceeds_stable_by_margin": changed_error - stable_error
        >= min_changed_error_delta,
        "effective_prediction_error_matches_temporal_signal": changed_effective == changed_error,
        "reset_clears_prior_prediction": reset_changed.get("temporal_observation_available") is False,
        "high_error_blocks_direct_hybrid_reflex": _hybrid_direct_reflex_allowed(
            changed_effective,
            threshold=prediction_error_threshold,
        )
        is False,
        "low_error_allows_direct_hybrid_reflex": _hybrid_direct_reflex_allowed(
            model_error,
            threshold=prediction_error_threshold,
        )
        is True,
        "high_observed_error_escalates_native_head_target": native_high_target
        == InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX.value
        and native_high_source == "observed_temporal_prediction_error",
        "first_frame_model_error_does_not_escalate_native_head_target": native_first_target
        == InternalTarget.REFLEX_MOTOR.value,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2bi_temporal_prediction_error",
        "passed": passed,
        "ready_for_runtime_observed_prediction_error_claim": passed,
        "ready_for_trained_world_model_accuracy_claim": False,
        "ready_for_repo_disjoint_plasticity_transfer_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "model_prediction_error": model_error,
            "prediction_error_threshold": prediction_error_threshold,
            "stable_observed_temporal_prediction_error": stable_error,
            "changed_observed_temporal_prediction_error": changed_error,
            "changed_effective_prediction_error": changed_effective,
            "changed_minus_stable_error_delta": changed_error - stable_error,
            "native_high_error_target": native_high_target,
            "native_high_error_target_source": native_high_source,
            "native_first_frame_high_model_error_target": native_first_target,
        },
        "trace": {
            "first": first,
            "stable": stable,
            "changed": changed,
            "reset_changed": reset_changed,
        },
        "supported_claims": [
            "low-level NSI policy now computes runtime-observed cross-frame prediction error by comparing predicted next-state vectors with subsequent observed state vectors",
            "that effective prediction error can suppress direct hybrid reflex execution and force escalation when it exceeds the configured gate",
        ]
        if passed
        else [],
        "unsupported_claims": [
            "trained world-model accuracy",
            "repo-disjoint plasticity transfer",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bj_train_and_validate_world_model_prediction_on_real_execution_streams"
            if passed
            else "repair_phase2bi_temporal_prediction_error_mechanism"
        ),
    }


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2BI runtime-observed temporal prediction error."
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-error", type=float, default=0.05)
    parser.add_argument("--prediction-error-threshold", type=float, default=0.45)
    parser.add_argument("--min-changed-error-delta", type=float, default=0.10)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2bi_temporal_prediction_error(
        model_error=args.model_error,
        prediction_error_threshold=args.prediction_error_threshold,
        min_changed_error_delta=args.min_changed_error_delta,
    )
    _write(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
