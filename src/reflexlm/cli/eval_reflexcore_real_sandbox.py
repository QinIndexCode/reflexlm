from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.experiment import _load_model
from reflexlm.core.sandbox_benchmark import (
    RealSandboxEvalConfig,
    evaluate_reflexcore_real_sandbox,
    evaluate_reflexcore_real_sandbox_families,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ReflexCore V0 on real temporary sandbox tasks.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-compare-baselines", action="store_true")
    parser.add_argument("--require-beats-baseline", default="prompt_only_heuristic")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--start-variant", type=int, default=0)
    parser.add_argument("--min-success-rate", type=float)
    parser.add_argument(
        "--live-observation",
        action="store_true",
        help="Re-observe terminal/process/filesystem/time receptors after each model action",
    )
    parser.add_argument("--max-text-tokens", type=int, default=128)
    args = parser.parse_args()

    model = _load_model(Path(args.checkpoint), device=args.device)
    output_dir = Path(args.output_dir)
    if args.family or args.variants != 1 or args.start_variant != 0:
        report = evaluate_reflexcore_real_sandbox_families(
            model,
            output_dir=output_dir,
            families=tuple(args.family),
            variants=args.variants,
            start_variant=args.start_variant,
            max_steps=args.max_steps,
            live_observation=args.live_observation,
            max_text_tokens=args.max_text_tokens,
        )
        _apply_min_success_gate(report, args.min_success_rate)
        (output_dir / "real_sandbox_family_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        report = evaluate_reflexcore_real_sandbox(
            model,
            config=RealSandboxEvalConfig(
                output_dir=output_dir,
                max_steps=args.max_steps,
                compare_baselines=not args.no_compare_baselines,
                require_beats_baseline=args.require_beats_baseline,
                live_observation=args.live_observation,
                max_text_tokens=args.max_text_tokens,
            ),
        )
        _apply_min_success_gate(report, args.min_success_rate)
        (output_dir / "real_sandbox_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


def _apply_min_success_gate(report: dict[str, object], min_success_rate: float | None) -> None:
    if min_success_rate is None:
        report.setdefault("passed", True)
        return
    if min_success_rate < 0.0 or min_success_rate > 1.0:
        raise ValueError("min_success_rate must be between 0 and 1")
    overall = report.get("overall")
    if not isinstance(overall, dict):
        model = report.get("model")
        overall = model if isinstance(model, dict) else {}
    success_rate = overall.get("success_rate")
    passed = isinstance(success_rate, float) and success_rate >= min_success_rate
    report["min_success_acceptance"] = {
        "min_success_rate": min_success_rate,
        "model_success_rate": success_rate,
        "passed": passed,
    }
    report["passed"] = bool(report.get("passed", True)) and passed


if __name__ == "__main__":
    main()
