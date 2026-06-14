from __future__ import annotations

import argparse
import json

from reflexlm.data.tasks import materialize_phase1_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase 1 NSI dataset.")
    parser.add_argument("--output", required=True, help="Output directory for JSONL splits")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--profile", default="default", help="Dataset environment profile")
    parser.add_argument(
        "--split-strategy",
        choices=["episode_random", "episode_fingerprint", "scenario_holdout"],
        default="episode_random",
    )
    args = parser.parse_args()
    manifest = materialize_phase1_dataset(
        args.output,
        seed=args.seed,
        profile=args.profile,
        split_strategy=args.split_strategy,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
