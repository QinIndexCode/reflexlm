from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.online_adaptation import (
    ReflexCoreOnlineAdaptationConfig,
    adapt_reflexcore_from_experience,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt an existing ReflexCore V0 checkpoint on bounded model "
            "experience JSONL examples."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experience", required=True)
    parser.add_argument("--retention")
    parser.add_argument("--holdout")
    parser.add_argument("--max-retention-loss-increase", type=float, default=0.0)
    parser.add_argument("--max-holdout-loss-increase", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
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

    report = adapt_reflexcore_from_experience(
        ReflexCoreOnlineAdaptationConfig(
            checkpoint_path=Path(args.checkpoint),
            experience_path=Path(args.experience),
            output_dir=Path(args.output_dir),
            retention_path=Path(args.retention) if args.retention else None,
            holdout_path=Path(args.holdout) if args.holdout else None,
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


if __name__ == "__main__":
    main()
