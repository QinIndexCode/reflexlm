from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.training import train_reflexcore_v0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ReflexCore V0 on local JSONL data.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/reflexcore/smoke.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sequence-mode", action="store_true")
    parser.add_argument("--max-sequence-len", type=int)
    args = parser.parse_args()

    summary = train_reflexcore_v0(
        dataset_path=Path(args.dataset),
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        sequence_mode=True if args.sequence_mode else None,
        max_sequence_len=args.max_sequence_len,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
