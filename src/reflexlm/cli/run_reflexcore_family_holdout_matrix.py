from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.online_adaptation_gate import (
    ReflexCoreFamilyHoldoutMatrixConfig,
    run_family_holdout_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a ReflexCore V0 online-adaptation matrix with each selected "
            "task family held out in turn."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--retention-episodes", type=int, default=1)
    parser.add_argument("--holdout-family", action="append", default=[])
    parser.add_argument("--max-retention-loss-increase", type=float, default=0.0)
    parser.add_argument("--max-holdout-loss-increase", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-mode", action="store_true")
    parser.add_argument("--max-sequence-len", type=int, default=8)
    parser.add_argument("--max-text-tokens", type=int, default=128)
    parser.add_argument(
        "--trainable-scope",
        choices=["all", "world_model_only"],
        default="all",
    )
    parser.add_argument("--behavior-eval-variants", type=int, default=0)
    parser.add_argument("--behavior-eval-start-variant", type=int, default=0)
    parser.add_argument("--behavior-eval-max-steps", type=int, default=4)
    parser.add_argument("--require-behavior-capability", action="store_true")
    parser.add_argument("--min-behavior-success-rate", type=float, default=0.0)
    args = parser.parse_args()

    report = run_family_holdout_matrix(
        ReflexCoreFamilyHoldoutMatrixConfig(
            checkpoint_path=Path(args.checkpoint),
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output_dir),
            split_seed=args.split_seed,
            train_episode_count=args.train_episodes,
            retention_episode_count=args.retention_episodes,
            holdout_families=tuple(args.holdout_family),
            max_retention_loss_increase=args.max_retention_loss_increase,
            max_holdout_loss_increase=args.max_holdout_loss_increase,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            sequence_mode=bool(args.sequence_mode),
            max_sequence_len=args.max_sequence_len,
            max_text_tokens=args.max_text_tokens,
            trainable_scope=args.trainable_scope,
            behavior_eval_variants=args.behavior_eval_variants,
            behavior_eval_start_variant=args.behavior_eval_start_variant,
            behavior_eval_max_steps=args.behavior_eval_max_steps,
            require_behavior_capability=args.require_behavior_capability,
            min_behavior_success_rate=args.min_behavior_success_rate,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
