from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.sensory_ablation_matrix import (
    ReflexCoreSensoryAblationMatrixConfig,
    run_reflexcore_sensory_ablation_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate sensory ablation gates across a ReflexCore profile matrix.",
    )
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument(
        "--mode",
        action="append",
        choices=["zero_numeric", "zero_hash", "zero_all"],
        default=[],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-mode", action="store_true")
    parser.add_argument("--max-sequence-len", type=int, default=8)
    parser.add_argument("--min-action-drop", type=float, default=0.5)
    parser.add_argument("--min-world-drop", type=float, default=1.0)
    parser.add_argument("--no-require-action-drop", action="store_true")
    parser.add_argument("--no-require-world-drop", action="store_true")
    args = parser.parse_args()

    report = run_reflexcore_sensory_ablation_matrix(
        ReflexCoreSensoryAblationMatrixConfig(
            matrix_dir=Path(args.matrix_dir),
            output_json=Path(args.output_json) if args.output_json else None,
            seeds=tuple(args.seed) if args.seed else (13, 17, 23),
            profiles=tuple(args.profile) if args.profile else (
                "default",
                "hard",
                "wide_ood",
            ),
            modes=tuple(args.mode) if args.mode else ("zero_numeric",),
            batch_size=args.batch_size,
            device=args.device,
            sequence_mode=args.sequence_mode,
            max_sequence_len=args.max_sequence_len,
            min_action_accuracy_drop=(
                None if args.no_require_action_drop else args.min_action_drop
            ),
            min_world_model_drop=(
                None if args.no_require_world_drop else args.min_world_drop
            ),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
