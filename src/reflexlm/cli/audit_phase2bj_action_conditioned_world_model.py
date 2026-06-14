from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reflexlm.data.jsonl import read_jsonl
from reflexlm.models.features import ACTION_ORDER, action_to_index
from reflexlm.schema import SourceType, TrajectoryRecord
from reflexlm.train import load_model_checkpoint


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantile(values: list[float], quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values else 0.0


def audit_phase2bj_action_conditioned_world_model(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    min_rows: int = 100,
    min_wrong_action_mse_delta: float = 1.0e-5,
    min_relative_wrong_action_mse_delta: float = 0.01,
    max_rows: int | None = None,
    prediction_error_threshold: float | None = None,
) -> dict[str, Any]:
    model, vectorizer, checkpoint_payload = load_model_checkpoint(checkpoint_path, device="cpu")
    records = read_jsonl(Path(dataset_path))
    all_rows_are_runtime_observations = bool(records) and all(
        record.source == SourceType.RUNTIME_OBSERVATION for record in records
    )
    grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.episode_id].append(record)
    continuous_episode_count = 0
    continuous_row_count = 0
    state_chain_pairs = 0
    state_chain_consistent_pairs = 0
    failure_transition_rows = 0
    timeout_transition_rows = 0
    recovered_episode_count = 0
    for episode_records in grouped.values():
        episode = sorted(episode_records, key=lambda item: item.t)
        if len(episode) > 1:
            continuous_episode_count += 1
            continuous_row_count += len(episode) - 1
        saw_failure = False
        recovered = False
        for index, record in enumerate(episode):
            exit_code = record.next_state.process.exit_code
            if exit_code not in (None, 0):
                failure_transition_rows += 1
                saw_failure = True
            if record.next_state.process.interrupted:
                timeout_transition_rows += 1
            if saw_failure and exit_code == 0:
                recovered = True
            if index > 0:
                state_chain_pairs += 1
                if record.state == episode[index - 1].next_state:
                    state_chain_consistent_pairs += 1
        if recovered:
            recovered_episode_count += 1

    correct_mse_rows: list[float] = []
    wrong_mse_rows: list[float] = []
    identity_mse_rows: list[float] = []
    action_sensitivity_rows: list[float] = []
    full_state_correct_mse_rows: list[float] = []
    full_state_identity_mse_rows: list[float] = []
    feature_mask = torch.tensor(
        vectorizer.world_model_target_mask(),
        dtype=torch.float32,
    )
    feature_mask_denominator = feature_mask.sum().clamp(min=1.0)
    evaluated_episodes = 0
    with torch.inference_mode():
        for episode_records in grouped.values():
            episode = [
                record
                for record in sorted(episode_records, key=lambda item: item.t)
                if record.action is not None
            ]
            if not episode:
                continue
            if max_rows is not None:
                remaining = max_rows - len(correct_mse_rows)
                if remaining <= 0:
                    break
                episode = episode[:remaining]
            inputs = torch.tensor(
                np.stack([vectorizer.vectorize_state(record.state) for record in episode]),
                dtype=torch.float32,
            ).unsqueeze(0)
            observed_next = torch.tensor(
                np.stack([vectorizer.vectorize_state(record.next_state) for record in episode]),
                dtype=torch.float32,
            ).unsqueeze(0)
            correct_actions = torch.tensor(
                [[action_to_index(record.action.type) for record in episode]],
                dtype=torch.long,
            )
            wrong_actions = (correct_actions + 1) % len(ACTION_ORDER)
            correct_prediction = model(
                inputs,
                action_indices=correct_actions,
            )["next_state"]
            wrong_prediction = model(
                inputs,
                action_indices=wrong_actions,
            )["next_state"]
            def masked_feature_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
                return (((left - right) ** 2) * feature_mask).sum(
                    dim=-1
                ) / feature_mask_denominator

            correct_mse_rows.extend(masked_feature_mse(correct_prediction, observed_next)[0].tolist())
            wrong_mse_rows.extend(masked_feature_mse(wrong_prediction, observed_next)[0].tolist())
            identity_mse_rows.extend(masked_feature_mse(inputs, observed_next)[0].tolist())
            action_sensitivity_rows.extend(
                masked_feature_mse(correct_prediction, wrong_prediction)[0].tolist()
            )
            full_state_correct_mse_rows.extend(
                torch.mean((correct_prediction - observed_next) ** 2, dim=-1)[0].tolist()
            )
            full_state_identity_mse_rows.extend(
                torch.mean((inputs - observed_next) ** 2, dim=-1)[0].tolist()
            )
            evaluated_episodes += 1

    correct_mse = _mean(correct_mse_rows)
    wrong_mse = _mean(wrong_mse_rows)
    identity_mse = _mean(identity_mse_rows)
    action_sensitivity = _mean(action_sensitivity_rows)
    correct_rmse_rows = [value**0.5 for value in correct_mse_rows]
    recommended_prediction_error_threshold = max(
        0.01,
        _quantile(correct_rmse_rows, 0.99) * 1.25,
    )
    evaluated_prediction_error_threshold = (
        recommended_prediction_error_threshold
        if prediction_error_threshold is None
        else float(prediction_error_threshold)
    )
    prediction_error_rate_above_threshold = _mean(
        [
            1.0 if value > evaluated_prediction_error_threshold else 0.0
            for value in correct_rmse_rows
        ]
    )
    wrong_delta = wrong_mse - correct_mse
    relative_wrong_delta = wrong_delta / max(wrong_mse, 1.0e-12)
    config = checkpoint_payload.get("model_config", {})
    load_info = checkpoint_payload.get("checkpoint_load", {})
    checks = {
        "checkpoint_declares_action_conditioned_world_model": config.get(
            "action_conditioned_world_model"
        )
        is True,
        "checkpoint_load_has_no_missing_or_unexpected_keys": not load_info.get("missing_keys")
        and not load_info.get("unexpected_keys"),
        "heldout_rows_meet_minimum": len(correct_mse_rows) >= min_rows,
        "world_model_is_action_sensitive": action_sensitivity > 0.0,
        "correct_action_beats_wrong_action_absolute_gate": wrong_delta
        >= min_wrong_action_mse_delta,
        "correct_action_beats_wrong_action_relative_gate": relative_wrong_delta
        >= min_relative_wrong_action_mse_delta,
        "learned_world_model_beats_identity_baseline": correct_mse < identity_mse,
    }
    passed = all(checks.values())
    ready_for_bounded_real_runtime_claim = passed and all_rows_are_runtime_observations
    ready_for_bounded_continuous_runtime_recovery_claim = (
        ready_for_bounded_real_runtime_claim
        and continuous_episode_count > 0
        and state_chain_pairs > 0
        and state_chain_consistent_pairs == state_chain_pairs
        and failure_transition_rows > 0
        and timeout_transition_rows > 0
        and recovered_episode_count > 0
    )
    supported_claims = [
        "the bounded NSI world model is explicitly action-conditioned and correct held-out actions improve next-state prediction over deterministic wrong-action interventions"
    ] if passed else []
    if ready_for_bounded_real_runtime_claim:
        supported_claims.append(
            "the bounded action-conditioned world model predicts controllable structured state transitions on workspace-confined held-out real runtime observations"
        )
    if ready_for_bounded_continuous_runtime_recovery_claim:
        supported_claims.append(
            "the bounded action-conditioned world model predicts controllable structured transitions across state-chained held-out failure, timeout, and recovery episodes"
        )
    return {
        "artifact_family": "phase2bj_action_conditioned_world_model",
        "passed": passed,
        "ready_for_bounded_heldout_action_conditioned_world_model_claim": passed,
        "ready_for_bounded_real_runtime_action_conditioned_world_model_claim": ready_for_bounded_real_runtime_claim,
        "ready_for_bounded_continuous_runtime_recovery_world_model_claim": ready_for_bounded_continuous_runtime_recovery_claim,
        "ready_for_real_external_runtime_world_model_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "rows": len(correct_mse_rows),
            "episodes": evaluated_episodes,
            "correct_action_next_state_mse": correct_mse,
            "wrong_action_next_state_mse": wrong_mse,
            "wrong_minus_correct_action_mse": wrong_delta,
            "relative_wrong_minus_correct_action_mse": relative_wrong_delta,
            "identity_next_state_mse": identity_mse,
            "action_sensitivity_mse": action_sensitivity,
            "world_model_target_features": int(feature_mask.sum().item()),
            "full_state_features": int(feature_mask.numel()),
            "full_state_correct_action_next_state_mse": _mean(full_state_correct_mse_rows),
            "full_state_identity_next_state_mse": _mean(full_state_identity_mse_rows),
            "correct_action_next_state_rmse_p50": _quantile(correct_rmse_rows, 0.50),
            "correct_action_next_state_rmse_p95": _quantile(correct_rmse_rows, 0.95),
            "correct_action_next_state_rmse_p99": _quantile(correct_rmse_rows, 0.99),
            "correct_action_next_state_rmse_max": max(correct_rmse_rows, default=0.0),
            "recommended_prediction_error_threshold": recommended_prediction_error_threshold,
            "evaluated_prediction_error_threshold": evaluated_prediction_error_threshold,
            "prediction_error_rate_above_threshold": prediction_error_rate_above_threshold,
            "residual_world_model": bool(config.get("residual_world_model", False)),
            "all_rows_are_runtime_observations": all_rows_are_runtime_observations,
            "continuous_episodes": continuous_episode_count,
            "continuous_rows": continuous_row_count,
            "state_chain_pairs": state_chain_pairs,
            "state_chain_consistent_pairs": state_chain_consistent_pairs,
            "failure_transition_rows": failure_transition_rows,
            "timeout_transition_rows": timeout_transition_rows,
            "recovered_episodes": recovered_episode_count,
        },
        "supported_claims": supported_claims,
        "unsupported_claims": [
            "real external-runtime world-model accuracy",
            "repo-disjoint world-model transfer",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bk_real_runtime_action_conditioned_world_model_streams"
            if passed
            else "repair_phase2bj_action_conditioned_world_model_learning"
        ),
        "inputs": {
            "checkpoint_path": str(Path(checkpoint_path)),
            "dataset_path": str(Path(dataset_path)),
            "max_rows": max_rows,
            "prediction_error_threshold": prediction_error_threshold,
        },
    }


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit held-out action-conditioned NSI world-model predictions."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-wrong-action-mse-delta", type=float, default=1.0e-5)
    parser.add_argument("--min-relative-wrong-action-mse-delta", type=float, default=0.01)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--prediction-error-threshold", type=float)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2bj_action_conditioned_world_model(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        min_rows=args.min_rows,
        min_wrong_action_mse_delta=args.min_wrong_action_mse_delta,
        min_relative_wrong_action_mse_delta=args.min_relative_wrong_action_mse_delta,
        max_rows=args.max_rows,
        prediction_error_threshold=args.prediction_error_threshold,
    )
    _write(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
