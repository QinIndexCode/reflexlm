from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reflexlm.data.jsonl import read_jsonl
from reflexlm.eval import SequenceModelPolicy
from reflexlm.schema import ActionDecision, SourceType, TrajectoryRecord
from reflexlm.train import load_model_checkpoint


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantile(values: list[float], quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values else 0.0


def _same_motor_payload(predicted: ActionDecision, recorded: ActionDecision) -> bool:
    return (
        predicted.type == recorded.type
        and predicted.command == recorded.command
        and predicted.file_target == recorded.file_target
    )


def audit_phase2bm_continuous_runtime_closed_loop(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    min_rows: int = 100,
    min_action_accuracy: float = 0.80,
    min_recovery_action_accuracy: float = 0.80,
    max_rows: int | None = None,
) -> dict[str, Any]:
    model, vectorizer, checkpoint_payload = load_model_checkpoint(checkpoint_path, device="cpu")
    records = read_jsonl(Path(dataset_path))
    grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.episode_id].append(record)

    action_matches: list[float] = []
    motor_payload_matches: list[float] = []
    recovery_action_matches: list[float] = []
    matched_action_world_model_rmse: list[float] = []
    matched_action_identity_rmse: list[float] = []
    valid_observed_temporal_errors: list[float] = []
    raw_candidate_matches: list[float] = []
    bounded_cortex_authorized_rows = 0
    evaluated_episodes = 0
    evaluated_rows = 0
    runtime_rows = 0
    continuous_rows = 0
    feature_mask = torch.tensor(
        vectorizer.world_model_target_mask(),
        dtype=torch.float32,
    )
    feature_count = max(int(feature_mask.sum().item()), 1)

    for episode_records in grouped.values():
        episode = sorted(episode_records, key=lambda item: item.t)
        if max_rows is not None:
            remaining = max_rows - evaluated_rows
            if remaining <= 0:
                break
            episode = episode[:remaining]
        if not episode:
            continue
        policy = SequenceModelPolicy(
            model,
            vectorizer,
            policy_label="phase2bm_continuous_runtime_closed_loop",
            training_summary=checkpoint_payload.get("training_summary", {}),
            authorize_bounded_debug_cortex_recovery=True,
        )
        previous_action_matched = False
        for index, record in enumerate(episode):
            if record.action is None:
                continue
            predicted = policy.act(record.state)
            type_match = predicted.type == record.action.type
            payload_match = _same_motor_payload(predicted, record.action)
            raw_candidate_payload = policy.last_call.get("raw_motor_candidate")
            raw_candidate = (
                ActionDecision.model_validate(raw_candidate_payload)
                if isinstance(raw_candidate_payload, dict)
                else predicted
            )
            raw_candidate_matches.append(
                float(_same_motor_payload(raw_candidate, record.action))
            )
            bounded_cortex_authorized_rows += int(
                policy.last_call.get("bounded_debug_cortex_action_authorized") is True
            )
            action_matches.append(float(type_match))
            motor_payload_matches.append(float(payload_match))
            if record.state.process.exit_code not in (None, 0) or record.state.process.interrupted:
                recovery_action_matches.append(float(payload_match))
            if index > 0:
                continuous_rows += 1
                if previous_action_matched:
                    observed = policy.last_call.get("observed_temporal_prediction_error_raw")
                    if observed is not None:
                        valid_observed_temporal_errors.append(float(observed))
            if payload_match and policy.predicted_next_state is not None:
                predicted_next = policy.predicted_next_state.float()
                observed_next = torch.tensor(
                    vectorizer.vectorize_state(record.next_state),
                    dtype=torch.float32,
                )
                current = torch.tensor(
                    vectorizer.vectorize_state(record.state),
                    dtype=torch.float32,
                )
                matched_action_world_model_rmse.append(
                    float(
                        torch.linalg.vector_norm(
                            (predicted_next - observed_next) * feature_mask
                        ).item()
                        / math.sqrt(feature_count)
                    )
                )
                matched_action_identity_rmse.append(
                    float(
                        torch.linalg.vector_norm((current - observed_next) * feature_mask).item()
                        / math.sqrt(feature_count)
                    )
                )
            previous_action_matched = payload_match
            evaluated_rows += 1
            runtime_rows += int(record.source == SourceType.RUNTIME_OBSERVATION)
        evaluated_episodes += 1

    action_accuracy = _mean(action_matches)
    motor_payload_accuracy = _mean(motor_payload_matches)
    recovery_action_accuracy = _mean(recovery_action_matches)
    world_model_rmse = _mean(matched_action_world_model_rmse)
    identity_rmse = _mean(matched_action_identity_rmse)
    checks = {
        "rows_meet_minimum": evaluated_rows >= min_rows,
        "all_rows_are_runtime_observations": evaluated_rows > 0
        and runtime_rows == evaluated_rows,
        "continuous_rows_present": continuous_rows > 0,
        "action_accuracy_meets_gate": action_accuracy >= min_action_accuracy,
        "structured_motor_payload_accuracy_meets_gate": motor_payload_accuracy
        >= min_action_accuracy,
        "raw_candidate_accuracy_meets_gate": _mean(raw_candidate_matches)
        >= min_action_accuracy,
        "recovery_action_accuracy_meets_gate": recovery_action_accuracy
        >= min_recovery_action_accuracy,
        "matched_action_world_model_rows_present": len(matched_action_world_model_rmse)
        >= min_rows,
        "matched_action_world_model_beats_identity": world_model_rmse < identity_rmse,
        "valid_cross_frame_temporal_errors_present": bool(valid_observed_temporal_errors),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2bm_continuous_runtime_closed_loop",
        "passed": passed,
        "ready_for_bounded_continuous_runtime_policy_world_model_claim": passed,
        "ready_for_repo_disjoint_runtime_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "rows": evaluated_rows,
            "episodes": evaluated_episodes,
            "runtime_rows": runtime_rows,
            "continuous_rows": continuous_rows,
            "action_accuracy": action_accuracy,
            "structured_motor_payload_accuracy": motor_payload_accuracy,
            "raw_candidate_accuracy": _mean(raw_candidate_matches),
            "bounded_debug_cortex_authorized_rows": bounded_cortex_authorized_rows,
            "recovery_action_rows": len(recovery_action_matches),
            "recovery_action_accuracy": recovery_action_accuracy,
            "matched_action_world_model_rows": len(matched_action_world_model_rmse),
            "matched_action_world_model_rmse": world_model_rmse,
            "matched_action_identity_rmse": identity_rmse,
            "matched_action_world_model_relative_improvement_over_identity": (
                (identity_rmse - world_model_rmse) / max(identity_rmse, 1.0e-12)
            ),
            "valid_cross_frame_temporal_error_rows": len(valid_observed_temporal_errors),
            "valid_cross_frame_temporal_error_p50": _quantile(
                valid_observed_temporal_errors, 0.50
            ),
            "valid_cross_frame_temporal_error_p95": _quantile(
                valid_observed_temporal_errors, 0.95
            ),
            "world_model_target_features": feature_count,
        },
        "supported_claims": [
            "the bounded sequence policy selects recorded structured actions and predicts subsequent controllable state transitions during state-chained real runtime replay"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "counterfactual execution of model-selected recovery actions",
            "repo-disjoint runtime transfer",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bn_execute_model_selected_actions_in_sealed_runtime"
            if passed
            else "repair_phase2bm_continuous_runtime_policy_or_world_model"
        ),
        "inputs": {
            "checkpoint_path": str(Path(checkpoint_path)),
            "dataset_path": str(Path(dataset_path)),
            "max_rows": max_rows,
            "min_action_accuracy": min_action_accuracy,
            "min_recovery_action_accuracy": min_recovery_action_accuracy,
        },
    }


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit model-selected actions and world-model predictions on continuous runtime replay."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-action-accuracy", type=float, default=0.80)
    parser.add_argument("--min-recovery-action-accuracy", type=float, default=0.80)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2bm_continuous_runtime_closed_loop(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        min_rows=args.min_rows,
        min_action_accuracy=args.min_action_accuracy,
        min_recovery_action_accuracy=args.min_recovery_action_accuracy,
        max_rows=args.max_rows,
    )
    _write(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
