from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.real_sandbox_adaptation import (
    ReflexCoreRealSandboxAdaptationConfig,
    run_reflexcore_real_sandbox_adaptation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train ReflexCore V0 on mixed synthetic plus real-sandbox traces "
            "and evaluate bounded sandbox transfer."
        ),
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
    parser.add_argument("--no-require-world-model-improvement", action="store_true")
    parser.add_argument("--min-world-model-relative-improvement", type=float, default=0.0)
    parser.add_argument("--no-require-prediction-error-improvement", action="store_true")
    parser.add_argument("--min-prediction-error-relative-improvement", type=float, default=0.0)
    parser.add_argument("--real-sandbox-variants", type=int, default=12)
    parser.add_argument("--real-sandbox-start-variant", type=int, default=1)
    parser.add_argument("--real-sandbox-max-steps", type=int, default=4)
    parser.add_argument(
        "--real-sandbox-required-baseline",
        default="prompt_only_heuristic",
    )
    parser.add_argument("--no-require-real-sandbox-baseline", action="store_true")
    parser.add_argument("--no-require-synthetic-gate", action="store_true")
    parser.add_argument("--synthetic-repeat", type=int, default=1)
    parser.add_argument("--real-sandbox-repeat", type=int, default=1)
    args = parser.parse_args()

    report = run_reflexcore_real_sandbox_adaptation(
        ReflexCoreRealSandboxAdaptationConfig(
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
            real_sandbox_variants=args.real_sandbox_variants,
            real_sandbox_start_variant=args.real_sandbox_start_variant,
            real_sandbox_max_steps=args.real_sandbox_max_steps,
            real_sandbox_required_baseline=(
                None
                if args.no_require_real_sandbox_baseline
                else args.real_sandbox_required_baseline
            ),
            require_synthetic_gate=not args.no_require_synthetic_gate,
            synthetic_repeat=args.synthetic_repeat,
            real_sandbox_repeat=args.real_sandbox_repeat,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
