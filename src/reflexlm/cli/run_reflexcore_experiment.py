from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.experiment import (
    ReflexCoreExperimentConfig,
    run_reflexcore_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ReflexCore V0 benchmark, training, offline eval, and closed-loop eval.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/reflexcore/smoke.yaml")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--eval-profile")
    parser.add_argument("--episodes-per-task", type=int, default=6)
    parser.add_argument(
        "--split-strategy",
        choices=["scenario_holdout", "episode_random", "episode_fingerprint"],
        default="scenario_holdout",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hash-bins", type=int, default=256)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-mode", action="store_true")
    parser.add_argument("--max-sequence-len", type=int)
    parser.add_argument("--required-baseline", default="prompt_only_heuristic")
    parser.add_argument("--closed-loop-episodes-per-task", type=int, default=2)
    parser.add_argument("--min-parameters", type=int)
    parser.add_argument("--max-parameters", type=int)
    parser.add_argument(
        "--no-require-world-model-improvement",
        action="store_true",
        help="Do not include next-state improvement over copy-current in the final pass gate.",
    )
    parser.add_argument(
        "--min-world-model-relative-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--no-require-prediction-error-improvement",
        action="store_true",
        help="Do not include prediction-error improvement over constant mean in the final pass gate.",
    )
    parser.add_argument(
        "--min-prediction-error-relative-improvement",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    report = run_reflexcore_experiment(
        ReflexCoreExperimentConfig(
            output_dir=Path(args.output_dir),
            model_config_path=Path(args.config),
            profile=args.profile,
            eval_profile=args.eval_profile,
            episodes_per_task=args.episodes_per_task,
            split_strategy=args.split_strategy,
            seed=args.seed,
            vocab_size=args.vocab_size,
            hash_bins=args.hash_bins,
            max_text_tokens=args.max_text_tokens,
            train_epochs=args.epochs,
            train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            sequence_mode=True if args.sequence_mode else None,
            max_sequence_len=args.max_sequence_len,
            required_baseline=args.required_baseline,
            closed_loop_episodes_per_task=args.closed_loop_episodes_per_task,
            min_parameters=args.min_parameters,
            max_parameters=args.max_parameters,
            require_world_model_improvement=not args.no_require_world_model_improvement,
            min_world_model_relative_improvement=args.min_world_model_relative_improvement,
            require_prediction_error_improvement=(
                not args.no_require_prediction_error_improvement
            ),
            min_prediction_error_relative_improvement=(
                args.min_prediction_error_relative_improvement
            ),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
