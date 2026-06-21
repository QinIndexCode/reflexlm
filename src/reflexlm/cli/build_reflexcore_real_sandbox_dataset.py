from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.sandbox_benchmark import build_real_sandbox_oracle_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ReflexCore V0 training data from real sandbox oracle traces.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--start-variant", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-text-tokens", type=int, default=128)
    args = parser.parse_args()

    summary = build_real_sandbox_oracle_dataset(
        output_path=Path(args.output),
        work_dir=Path(args.work_dir),
        variants=args.variants,
        start_variant=args.start_variant,
        vocab_size=args.vocab_size,
        max_text_tokens=args.max_text_tokens,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
