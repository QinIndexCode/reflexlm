from __future__ import annotations

import argparse
import json
import re
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


def _infer_model_label(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "qwen7b" in name:
        return "qwen7b"
    if "qwen3b" in name:
        return "qwen3b"
    if "qwen15b" in name:
        return "qwen15b"
    return "unknown"


def _infer_seed_label(path: str | Path) -> str | None:
    match = re.search(r"seed(\d+)", Path(path).name.lower())
    return match.group(1) if match else None


def _min_numeric(entries: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for entry in entries:
        value = entry.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return min(values) if values else None


def build_phase2am_reproduction_report(
    *,
    postflight_reports: list[str | Path],
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    reports = [_read_json(path) for path in postflight_reports]
    entries: list[dict[str, Any]] = []
    for path, report in zip(postflight_reports, reports):
        metrics = _metrics(report)
        entries.append(
            {
                "postflight_json": str(Path(path)),
                "model_label": _infer_model_label(path),
                "seed_label": _infer_seed_label(path),
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
    model_labels = {entry["model_label"] for entry in entries if entry["model_label"] != "unknown"}
    seed_labels = {entry["seed_label"] for entry in entries if entry["seed_label"]}
    row_counts = {entry.get("row_count") for entry in entries}
    repo_counts = {entry.get("repo_count") for entry in entries}

    observed_min_full_minus_source = _min_numeric(entries, "full_minus_source_overlap")
    observed_min_full_minus_erased = _min_numeric(entries, "full_minus_sidecar_erased")
    observed_min_full_minus_wrong = _min_numeric(entries, "full_minus_wrong_sidecar")
    passed_min_full_minus_source = _min_numeric(passed_entries, "full_minus_source_overlap")
    passed_min_full_minus_erased = _min_numeric(passed_entries, "full_minus_sidecar_erased")
    passed_min_full_minus_wrong = _min_numeric(passed_entries, "full_minus_wrong_sidecar")

    checks = {
        "all_postflight_reports_passed": len(passed_entries) == len(entries) and bool(entries),
        "at_least_three_runs": len(entries) >= 3,
        "at_least_two_model_families": len(model_labels) >= 2,
        "has_seed_replication": bool(seed_labels),
        "same_row_count": len(row_counts) == 1,
        "same_repo_count": len(repo_counts) == 1,
        "observed_min_full_minus_source_ge_0_15": isinstance(
            observed_min_full_minus_source, float
        )
        and observed_min_full_minus_source >= 0.15,
        "observed_min_full_minus_erased_ge_0_25": isinstance(
            observed_min_full_minus_erased, float
        )
        and observed_min_full_minus_erased >= 0.25,
        "observed_min_full_minus_wrong_ge_0_25": isinstance(
            observed_min_full_minus_wrong, float
        )
        and observed_min_full_minus_wrong >= 0.25,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2am_reproduction_report",
        "passed": passed,
        "natural_repo_disjoint_sidecar_dependency_reproduced": passed,
        "claim_bearing_mechanism_evidence": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "metrics": {
            "run_count": len(entries),
            "model_count": len(model_labels),
            "seed_replication_count": len(seed_labels),
            "row_count": next(iter(row_counts)) if len(row_counts) == 1 else None,
            "repo_count": next(iter(repo_counts)) if len(repo_counts) == 1 else None,
            "observed_min_full_minus_source_overlap": observed_min_full_minus_source,
            "observed_min_full_minus_sidecar_erased": observed_min_full_minus_erased,
            "observed_min_full_minus_wrong_sidecar": observed_min_full_minus_wrong,
            "passed_min_full_minus_source_overlap": passed_min_full_minus_source,
            "passed_min_full_minus_sidecar_erased": passed_min_full_minus_erased,
            "passed_min_full_minus_wrong_sidecar": passed_min_full_minus_wrong,
        },
        "postflight_reports": entries,
        "blocked_actions": [
            "do_not_claim_sealed_transfer_from_phase2am",
            "do_not_claim_open_ended_debugging_generalization",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
        "allowed_next_action": (
            "add_independent_public_trace_family_or_multi_seed_full_training"
            if passed
            else "freeze_phase2am_reproduction_failure"
        ),
        "claim_boundary": (
            "Phase2AM reproduces natural held-out sidecar-dependency evidence across "
            "multiple runs and model sizes for bounded command-slot selection. It does "
            "not establish sealed transfer, production autonomy, open-ended debugging, "
            "or an epoch-making architecture."
        ),
    }
    if output_json:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AM reproduction evidence report.")
    parser.add_argument("--postflight-report", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2am_reproduction_report(
        postflight_reports=args.postflight_report,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
