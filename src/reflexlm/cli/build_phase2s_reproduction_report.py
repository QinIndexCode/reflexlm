from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _parse_run(value: str) -> tuple[str, str | None, Path]:
    parts = value.split("=", 2)
    if len(parts) == 2:
        return parts[0], None, Path(parts[1])
    if len(parts) == 3:
        return parts[0], parts[1], Path(parts[2])
    raise ValueError("Run must be model=path or model=seed=path")


def build_phase2s_reproduction_report(
    *,
    runs: list[tuple[str, str | None, str | Path]],
    min_models: int = 2,
    min_seeds_per_model: int = 1,
    min_holdout_accuracy: float = 0.85,
    min_holdout_minus_source_overlap: float = 0.15,
    min_holdout_minus_zero_nsi: float = 0.15,
    require_distinct_models: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    split_hashes: dict[str, Any] | None = None
    for model_key, seed, path in runs:
        payload = _load(path)
        metrics = payload.get("metrics", {})
        checks = payload.get("checks", {})
        current_hashes = payload.get("effective_split_hashes")
        if split_hashes is None:
            split_hashes = current_hashes
        row = {
            "model_key": model_key,
            "seed": seed,
            "passed": payload.get("passed") is True,
            "eval_json": str(Path(path)),
            "val_command_slot_accuracy": metrics.get("val_command_slot_accuracy"),
            "holdout_command_slot_accuracy": metrics.get("holdout_command_slot_accuracy"),
            "holdout_model_minus_source_overlap_accuracy": metrics.get(
                "holdout_model_minus_source_overlap_accuracy"
            ),
            "holdout_model_minus_zero_nsi_accuracy": metrics.get(
                "holdout_model_minus_zero_nsi_accuracy"
            ),
            "holdout_command_record_count": metrics.get("holdout_command_record_count"),
            "duration_seconds": metrics.get("duration_seconds"),
            "pairwise_disabled": checks.get("pairwise_disabled_for_phase2s_full") is True,
            "low_level_qwen_calls_target_zero": checks.get(
                "low_level_qwen_calls_target_zero"
            )
            is True,
            "no_json_motor_target": checks.get("no_json_motor_target") is True,
            "sealed_not_tuned": checks.get("holdout_diagnostics_not_sealed_tuned") is True,
            "split_hashes_match_first_run": current_hashes == split_hashes,
        }
        rows.append(row)
    models = sorted({row["model_key"] for row in rows})
    seeds_by_model = {
        model: sorted({str(row["seed"]) for row in rows if row["model_key"] == model})
        for model in models
    }
    eligible_rows = [
        row
        for row in rows
        if row["passed"]
        and row["split_hashes_match_first_run"]
        and row["pairwise_disabled"]
        and row["low_level_qwen_calls_target_zero"]
        and row["no_json_motor_target"]
        and row["sealed_not_tuned"]
        and isinstance(row["holdout_command_slot_accuracy"], (int, float))
        and row["holdout_command_slot_accuracy"] >= min_holdout_accuracy
        and isinstance(row["holdout_model_minus_source_overlap_accuracy"], (int, float))
        and row["holdout_model_minus_source_overlap_accuracy"]
        >= min_holdout_minus_source_overlap
        and isinstance(row["holdout_model_minus_zero_nsi_accuracy"], (int, float))
        and row["holdout_model_minus_zero_nsi_accuracy"] >= min_holdout_minus_zero_nsi
    ]
    eligible_models = sorted({row["model_key"] for row in eligible_rows})
    eligible_seeds_by_model = {
        model: sorted({str(row["seed"]) for row in eligible_rows if row["model_key"] == model})
        for model in eligible_models
    }
    all_model_seed_requirements_met = all(
        len(eligible_seeds_by_model.get(model, [])) >= min_seeds_per_model
        for model in eligible_models
    )
    checks = {
        "run_count_minimum_met": len(rows) >= min_models,
        "distinct_model_count_minimum_met": (
            len(eligible_models) >= min_models if require_distinct_models else True
        ),
        "seed_count_minimum_met": all_model_seed_requirements_met,
        "all_runs_passed": len(eligible_rows) == len(rows) and bool(rows),
        "split_hashes_consistent": all(row["split_hashes_match_first_run"] for row in rows),
        "sealed_not_used_for_training_or_tuning": all(row["sealed_not_tuned"] for row in rows),
        "pairwise_disabled": all(row["pairwise_disabled"] for row in rows),
        "no_json_motor_target": all(row["no_json_motor_target"] for row in rows),
        "low_level_qwen_calls_target_zero": all(
            row["low_level_qwen_calls_target_zero"] for row in rows
        ),
    }
    passed = all(checks.values())
    supported_claims = [
        "Phase2S cross-model mechanism evidence reproduces across the listed model sizes on the same preregistered non-sealed public repair split"
        if passed
        else "No cross-model reproduction claim is supported unless this report passes",
    ]
    unsupported_claims = [
        "cross-family reproduction is not proven unless listed models include distinct model families",
        "sealed final evaluation must remain separate from this non-sealed reproduction report",
        "production autonomy and open-ended debugging generalization are not proven",
    ]
    if passed and min_seeds_per_model > 1 and checks["seed_count_minimum_met"]:
        supported_claims.append(
            "Phase2S multi-seed robustness is supported for the listed model sizes and seeds under this preregistered non-sealed split"
        )
    else:
        unsupported_claims.insert(
            0,
            "multi-seed robustness is not proven unless min_seeds_per_model exceeds one and passes",
        )

    return {
        "artifact_family": "phase2s_cross_model_reproduction_report",
        "passed": passed,
        "claim_scope": (
            "phase2s_cross_model_seed13_reproduction_positive_bounded"
            if passed and min_seeds_per_model == 1
            else "phase2s_multiseed_reproduction_positive_bounded"
            if passed
            else "phase2s_reproduction_not_complete"
        ),
        "checks": checks,
        "thresholds": {
            "min_models": min_models,
            "min_seeds_per_model": min_seeds_per_model,
            "min_holdout_accuracy": min_holdout_accuracy,
            "min_holdout_minus_source_overlap": min_holdout_minus_source_overlap,
            "min_holdout_minus_zero_nsi": min_holdout_minus_zero_nsi,
            "require_distinct_models": require_distinct_models,
        },
        "metrics": {
            "model_count": len(models),
            "eligible_model_count": len(eligible_models),
            "run_count": len(rows),
            "eligible_run_count": len(eligible_rows),
            "models": models,
            "seeds_by_model": seeds_by_model,
            "eligible_models": eligible_models,
            "eligible_seeds_by_model": eligible_seeds_by_model,
        },
        "runs": rows,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "effective_split_hashes": split_hashes,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase2S Cross-Model Reproduction Report",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Claim scope: `{report['claim_scope']}`",
        f"- Models: `{', '.join(report['metrics']['models'])}`",
        f"- Eligible models: `{', '.join(report['metrics']['eligible_models'])}`",
        "",
        "## Checks",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in report["checks"].items())
    lines.extend(["", "## Runs"])
    lines.append(
        "| model | seed | passed | val acc | holdout acc | holdout-source | holdout-zeroNSI | records |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report["runs"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['model_key']}`",
                    f"`{row['seed']}`",
                    f"`{row['passed']}`",
                    f"`{row['val_command_slot_accuracy']}`",
                    f"`{row['holdout_command_slot_accuracy']}`",
                    f"`{row['holdout_model_minus_source_overlap_accuracy']}`",
                    f"`{row['holdout_model_minus_zero_nsi_accuracy']}`",
                    f"`{row['holdout_command_record_count']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Supported Claims"])
    lines.extend(f"- {item}" for item in report["supported_claims"])
    lines.extend(["", "## Unsupported Claims"])
    lines.extend(f"- {item}" for item in report["unsupported_claims"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2S reproduction report.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run in model=path or model=seed=path form.",
    )
    parser.add_argument("--min-models", type=int, default=2)
    parser.add_argument("--min-seeds-per-model", type=int, default=1)
    parser.add_argument("--min-holdout-accuracy", type=float, default=0.85)
    parser.add_argument("--min-holdout-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-holdout-minus-zero-nsi", type=float, default=0.15)
    parser.add_argument("--allow-same-model", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2s_reproduction_report(
        runs=[_parse_run(run) for run in args.run],
        min_models=args.min_models,
        min_seeds_per_model=args.min_seeds_per_model,
        min_holdout_accuracy=args.min_holdout_accuracy,
        min_holdout_minus_source_overlap=args.min_holdout_minus_source_overlap,
        min_holdout_minus_zero_nsi=args.min_holdout_minus_zero_nsi,
        require_distinct_models=not args.allow_same_model,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
