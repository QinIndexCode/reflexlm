from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.real_sandbox_capability_matrix import (
    ReflexCoreRealSandboxCapabilityMatrixConfig,
    run_reflexcore_real_sandbox_capability_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train ReflexCore V0 across seeds and evaluate real-sandbox "
            "terminal/process/filesystem/time capability on disjoint variants."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/reflexcore/smoke.yaml")
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--train-variants", type=int, default=20)
    parser.add_argument("--train-start-variant", type=int, default=0)
    parser.add_argument("--eval-variants", type=int, default=5)
    parser.add_argument("--eval-start-variant", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--max-text-tokens", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-sequence-len", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    args = parser.parse_args()

    report = run_reflexcore_real_sandbox_capability_matrix(
        ReflexCoreRealSandboxCapabilityMatrixConfig(
            output_dir=Path(args.output_dir),
            model_config_path=Path(args.config),
            seeds=tuple(args.seed) if args.seed else (13, 17, 23),
            train_variants=args.train_variants,
            train_start_variant=args.train_start_variant,
            eval_variants=args.eval_variants,
            eval_start_variant=args.eval_start_variant,
            vocab_size=args.vocab_size,
            max_text_tokens=args.max_text_tokens,
            train_epochs=args.epochs,
            train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            sequence_mode=bool(args.sequence_mode),
            max_sequence_len=args.max_sequence_len,
            max_steps=args.max_steps,
            families=tuple(args.family),
            min_success_rate=args.min_success_rate,
            min_pass_rate=args.min_pass_rate,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
