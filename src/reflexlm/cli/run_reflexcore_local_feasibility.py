from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.local_feasibility import (
    ReflexCoreLocalFeasibilityConfig,
    run_reflexcore_local_feasibility,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ReflexCore V0 local 20-100M feasibility gate.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/reflexcore/local.yaml")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument(
        "--split-strategy",
        choices=["scenario_holdout", "episode_random", "episode_fingerprint"],
        default="episode_random",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hash-bins", type=int, default=256)
    parser.add_argument("--max-text-tokens", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device")
    parser.add_argument("--sequence-mode", dest="sequence_mode", action="store_true")
    parser.add_argument("--no-sequence-mode", dest="sequence_mode", action="store_false")
    parser.set_defaults(sequence_mode=True)
    parser.add_argument("--max-sequence-len", type=int, default=8)
    parser.add_argument("--min-parameters", type=int, default=20_000_000)
    parser.add_argument("--max-parameters", type=int, default=100_000_000)
    args = parser.parse_args()

    report = run_reflexcore_local_feasibility(
        ReflexCoreLocalFeasibilityConfig(
            output_dir=Path(args.output_dir),
            model_config_path=Path(args.config),
            profile=args.profile,
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
            sequence_mode=args.sequence_mode,
            max_sequence_len=args.max_sequence_len,
            min_parameters=args.min_parameters,
            max_parameters=args.max_parameters,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
