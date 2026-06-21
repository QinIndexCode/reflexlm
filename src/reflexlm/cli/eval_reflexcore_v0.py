from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reflexlm.core.dataset import read_reflexcore_jsonl
from reflexlm.core.evaluation import (
    acceptance_against_baselines,
    evaluate_baseline_policies,
    evaluate_reflexcore_model,
    evaluate_reflexcore_sensory_ablation,
    prediction_error_acceptance,
    world_model_acceptance,
)
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReflexCore V0 on a fixed split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json")
    parser.add_argument("--sequence-mode", action="store_true")
    parser.add_argument("--max-sequence-len", type=int)
    parser.add_argument("--compare-baselines", action="store_true")
    parser.add_argument(
        "--ablation-mode",
        action="append",
        choices=["zero_numeric", "zero_hash", "zero_all"],
        default=[],
        help="Evaluate the same checkpoint after ablating observation-vector channels.",
    )
    parser.add_argument(
        "--require-sensory-ablation-drop",
        type=float,
        help=(
            "Exit non-zero unless every requested ablation mode reduces raw action "
            "accuracy by at least this amount."
        ),
    )
    parser.add_argument(
        "--require-sensory-world-drop",
        type=float,
        help=(
            "Exit non-zero unless every requested ablation mode reduces "
            "next-state relative improvement by at least this amount."
        ),
    )
    parser.add_argument(
        "--require-beats-baseline",
        action="append",
        choices=["prompt_only_heuristic", "rule_oracle", "static_wait"],
        default=[],
        help="Exit non-zero unless model action accuracy beats this baseline.",
    )
    parser.add_argument(
        "--require-world-model-improvement",
        action="store_true",
        help="Exit non-zero unless next-state MSE beats copy-current baseline.",
    )
    parser.add_argument(
        "--min-world-model-relative-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--require-prediction-error-improvement",
        action="store_true",
        help="Exit non-zero unless prediction-error MAE beats constant-mean baseline.",
    )
    parser.add_argument(
        "--min-prediction-error-relative-improvement",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint), map_location=args.device)
    config = ReflexCoreV0Config(**checkpoint["config"])
    model = ReflexCoreV0(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()
    examples = read_reflexcore_jsonl(Path(args.dataset))
    model_summary = evaluate_reflexcore_model(
        model,
        examples,
        batch_size=args.batch_size,
        device=args.device,
        sequence_mode=args.sequence_mode,
        max_sequence_len=args.max_sequence_len,
    )
    baselines = (
        evaluate_baseline_policies(examples)
        if args.compare_baselines or args.require_beats_baseline
        else {}
    )
    acceptance = acceptance_against_baselines(
        model_summary,
        baselines,
        required_baselines=args.require_beats_baseline,
    )
    world_acceptance = world_model_acceptance(
        model_summary,
        min_relative_improvement=args.min_world_model_relative_improvement,
    )
    prediction_error_gate = prediction_error_acceptance(
        model_summary,
        min_relative_improvement=args.min_prediction_error_relative_improvement,
    )
    sensory_ablation = None
    if (
        args.ablation_mode
        or args.require_sensory_ablation_drop is not None
        or args.require_sensory_world_drop is not None
    ):
        sensory_ablation = evaluate_reflexcore_sensory_ablation(
            model,
            examples,
            modes=args.ablation_mode or ["zero_numeric"],
            batch_size=args.batch_size,
            device=args.device,
            sequence_mode=args.sequence_mode,
            max_sequence_len=args.max_sequence_len,
            min_action_accuracy_drop=args.require_sensory_ablation_drop,
            min_next_state_relative_improvement_drop=args.require_sensory_world_drop,
        )
    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "dataset": str(Path(args.dataset)),
        "model": model_summary,
        "baselines": baselines,
        "acceptance": acceptance,
        "world_model_acceptance": world_acceptance,
        "prediction_error_acceptance": prediction_error_gate,
        "sensory_ablation": sensory_ablation,
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not acceptance["passed"]:
        raise SystemExit(2)
    if args.require_world_model_improvement and not world_acceptance["passed"]:
        raise SystemExit(2)
    if args.require_prediction_error_improvement and not prediction_error_gate["passed"]:
        raise SystemExit(2)
    if (
        (
            args.require_sensory_ablation_drop is not None
            or args.require_sensory_world_drop is not None
        )
        and isinstance(sensory_ablation, dict)
        and not sensory_ablation["passed"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
