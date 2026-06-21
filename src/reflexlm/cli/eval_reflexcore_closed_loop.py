from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reflexlm.core.closed_loop import (
    closed_loop_acceptance_against_baselines,
    evaluate_closed_loop_baselines,
    evaluate_reflexcore_closed_loop,
)
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ReflexCore V0 in closed-loop Phase 1 task environments.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compare-baselines", action="store_true")
    parser.add_argument(
        "--require-beats-baseline",
        action="append",
        choices=["prompt_only_heuristic", "rule_oracle", "static_wait"],
        default=[],
    )
    parser.add_argument("--output-json")
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint), map_location=args.device)
    model = ReflexCoreV0(ReflexCoreV0Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model_summary = evaluate_reflexcore_closed_loop(
        model,
        profile=args.profile,
        episodes_per_task=args.episodes_per_task,
        device=args.device,
        max_steps=args.max_steps,
    )
    baselines = (
        evaluate_closed_loop_baselines(
            profile=args.profile,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
        )
        if args.compare_baselines or args.require_beats_baseline
        else {}
    )
    acceptance = closed_loop_acceptance_against_baselines(
        model_summary,
        baselines,
        required_baselines=args.require_beats_baseline,
    )
    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "profile": args.profile,
        "episodes_per_task": args.episodes_per_task,
        "model": model_summary,
        "baselines": baselines,
        "acceptance": acceptance,
        "claim_boundary": (
            "Closed-loop success applies only to bounded terminal/process/"
            "filesystem/time task environments."
        ),
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not acceptance["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
