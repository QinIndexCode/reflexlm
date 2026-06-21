from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.benchmark import (
    ReflexCoreBenchmarkConfig,
    build_reflexcore_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reproducible ReflexCore V0 benchmark package.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--episodes-per-task", type=int, default=6)
    parser.add_argument(
        "--split-strategy",
        choices=["scenario_holdout", "episode_random", "episode_fingerprint"],
        default="scenario_holdout",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hash-bins", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    args = parser.parse_args()

    manifest = build_reflexcore_benchmark(
        ReflexCoreBenchmarkConfig(
            output_dir=Path(args.output_dir),
            profile=args.profile,
            episodes_per_task=args.episodes_per_task,
            split_strategy=args.split_strategy,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            hash_bins=args.hash_bins,
            vocab_size=args.vocab_size,
            max_text_tokens=args.max_text_tokens,
        )
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
