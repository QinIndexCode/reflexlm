from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def build_phase2am_cross_model_report(
    *,
    model_reports: list[str | Path],
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    reports = [_read_json(path) for path in model_reports]
    entries: list[dict[str, Any]] = []
    for path, report in zip(model_reports, reports):
        metrics = _metrics(report)
        entries.append(
            {
                "postflight_json": str(Path(path)),
                "passed": report.get("passed") is True,
                "full_accuracy": metrics.get("full_accuracy"),
                "source_overlap_accuracy": metrics.get("source_overlap_accuracy"),
                "sidecar_erased_accuracy": metrics.get("sidecar_erased_accuracy"),
                "wrong_sidecar_accuracy": metrics.get("wrong_sidecar_accuracy"),
                "full_minus_source_overlap": metrics.get("full_minus_source_overlap"),
                "full_minus_sidecar_erased": metrics.get("full_minus_sidecar_erased"),
                "full_minus_wrong_sidecar": metrics.get("full_minus_wrong_sidecar"),
                "row_count": metrics.get("row_count"),
                "repo_count": metrics.get("repo_count"),
            }
        )
    passed_entries = [entry for entry in entries if entry["passed"]]
    common_row_count = {entry.get("row_count") for entry in entries}
    common_repo_count = {entry.get("repo_count") for entry in entries}
    min_full_minus_source = min(
        float(entry["full_minus_source_overlap"]) for entry in passed_entries
    ) if passed_entries else None
    min_full_minus_erased = min(
        float(entry["full_minus_sidecar_erased"]) for entry in passed_entries
    ) if passed_entries else None
    min_full_minus_wrong = min(
        float(entry["full_minus_wrong_sidecar"]) for entry in passed_entries
    ) if passed_entries else None
    checks = {
        "all_model_reports_passed": len(passed_entries) == len(entries) and bool(entries),
        "at_least_two_models": len(entries) >= 2,
        "same_row_count": len(common_row_count) == 1,
        "same_repo_count": len(common_repo_count) == 1,
        "min_full_minus_source_ge_0_15": isinstance(min_full_minus_source, float)
        and min_full_minus_source >= 0.15,
        "min_full_minus_erased_ge_0_25": isinstance(min_full_minus_erased, float)
        and min_full_minus_erased >= 0.25,
        "min_full_minus_wrong_ge_0_25": isinstance(min_full_minus_wrong, float)
        and min_full_minus_wrong >= 0.25,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2am_cross_model_report",
        "passed": passed,
        "cross_model_natural_sidecar_dependency_evidence": passed,
        "claim_bearing_mechanism_evidence": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "metrics": {
            "model_count": len(entries),
            "row_count": next(iter(common_row_count)) if len(common_row_count) == 1 else None,
            "repo_count": next(iter(common_repo_count)) if len(common_repo_count) == 1 else None,
            "min_full_minus_source_overlap": min_full_minus_source,
            "min_full_minus_sidecar_erased": min_full_minus_erased,
            "min_full_minus_wrong_sidecar": min_full_minus_wrong,
        },
        "model_reports": entries,
        "blocked_actions": [
            "do_not_claim_sealed_transfer_from_phase2am",
            "do_not_claim_open_ended_debugging_generalization",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
        "allowed_next_action": (
            "add_seed_replication_or_new_public_trace_family"
            if passed
            else "freeze_phase2am_cross_model_failure"
        ),
        "claim_boundary": (
            "Phase2AM supports cross-model natural held-out sidecar-dependency evidence for "
            "bounded command-slot selection only. It does not establish sealed transfer, "
            "production autonomy, open-ended debugging, or an epoch-making architecture."
        ),
    }
    if output_json:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AM cross-model evidence report.")
    parser.add_argument("--model-report", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2am_cross_model_report(
        model_reports=args.model_report,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
