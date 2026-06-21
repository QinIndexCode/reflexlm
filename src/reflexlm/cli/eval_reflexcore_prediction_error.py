from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.experiment import _load_model
from reflexlm.core.prediction_error_report import (
    ReflexCorePredictionErrorReportConfig,
    build_reflexcore_prediction_error_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build action-conditioned ReflexCore V0 prediction-error diagnostics.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-mode", action="store_true")
    parser.add_argument("--max-text-tokens", type=int, default=128)
    parser.add_argument("--min-relative-improvement", type=float, default=0.0)
    parser.add_argument("--min-action-group-pass-rate", type=float, default=0.0)
    parser.add_argument("--min-evaluable-constant-mae", type=float, default=1e-4)
    args = parser.parse_args()

    model = _load_model(Path(args.checkpoint), device=args.device)
    report = build_reflexcore_prediction_error_report(
        model,
        ReflexCorePredictionErrorReportConfig(
            output_dir=Path(args.output_dir),
            dataset_path=Path(args.dataset),
            device=args.device,
            sequence_mode=args.sequence_mode,
            max_text_tokens=args.max_text_tokens,
            min_relative_improvement=args.min_relative_improvement,
            min_action_group_pass_rate=args.min_action_group_pass_rate,
            min_evaluable_constant_mae=args.min_evaluable_constant_mae,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
