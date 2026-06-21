from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.online_adaptation_gate import (
    ReflexCoreOnlineAdaptationGateConfig,
    run_online_adaptation_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ReflexCore V0 online adaptation on disjoint bounded "
            "train/retention/holdout episodes."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split-strategy",
        choices=["episode_holdout", "family_holdout"],
        default="episode_holdout",
    )
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument("--train-episodes", type=int, default=4)
    parser.add_argument("--retention-episodes", type=int, default=1)
    parser.add_argument("--holdout-episodes", type=int)
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
    args = parser.parse_args()

    report = run_online_adaptation_gate(
        ReflexCoreOnlineAdaptationGateConfig(
            checkpoint_path=Path(args.checkpoint),
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output_dir),
            split_strategy=args.split_strategy,
            split_seed=args.split_seed,
            train_episode_count=args.train_episodes,
            retention_episode_count=args.retention_episodes,
            holdout_episode_count=args.holdout_episodes,
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
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
