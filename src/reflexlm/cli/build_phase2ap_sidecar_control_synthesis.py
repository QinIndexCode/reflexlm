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


def _infer_label(path: str | Path) -> str:
    name = Path(path).name
    match = re.search(r"(qwen\d+b(?:_seed\d+)?)", name, re.IGNORECASE)
    return match.group(1).lower() if match else Path(path).stem


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _min_metric(entries: list[dict[str, Any]], key: str) -> float | None:
    values = [_float(entry.get(key)) for entry in entries]
    numeric = [value for value in values if isinstance(value, float)]
    return min(numeric) if numeric else None


def build_phase2ap_sidecar_control_synthesis(
    *,
    phase2an_reports: list[str | Path],
    phase2ao_reports: list[str | Path],
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    an_entries: list[dict[str, Any]] = []
    for path in phase2an_reports:
        report = _read_json(path)
        metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
        an_entries.append(
            {
                "label": _infer_label(path),
                "postflight_json": str(Path(path)),
                "passed": report.get("passed") is True,
                "pure_sidecar_dependency_supported": report.get(
                    "pure_sidecar_dependency_supported"
                )
                is True,
                "neutral_original_accuracy": metrics.get("neutral_original_accuracy"),
                "neutral_sidecar_erased_accuracy": metrics.get(
                    "neutral_sidecar_erased_accuracy"
                ),
                "neutral_wrong_sidecar_accuracy": metrics.get("neutral_wrong_sidecar_accuracy"),
                "neutral_original_minus_erased": metrics.get("neutral_original_minus_erased"),
                "neutral_original_minus_wrong": metrics.get("neutral_original_minus_wrong"),
                "erased_residual_above_source": metrics.get("erased_residual_above_source")
                is True,
            }
        )
    ao_entries: list[dict[str, Any]] = []
    for path in phase2ao_reports:
        report = _read_json(path)
        metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
        ao_entries.append(
            {
                "label": _infer_label(path),
                "postflight_json": str(Path(path)),
                "passed": report.get("passed") is True,
                "sidecar_order_robustness_supported": report.get(
                    "sidecar_order_robustness_supported"
                )
                is True,
                "strict_erased_residual_control_passed": report.get(
                    "strict_erased_residual_control_passed"
                )
                is True,
                "original_permuted_accuracy": metrics.get("original_permuted_accuracy"),
                "erased_permuted_accuracy": metrics.get("erased_permuted_accuracy"),
                "source_overlap_accuracy": metrics.get("source_overlap_accuracy"),
                "original_minus_erased_permuted": metrics.get(
                    "original_minus_erased_permuted"
                ),
                "original_minus_source": metrics.get("original_minus_source"),
            }
        )

    checks = {
        "has_phase2an_reports": bool(an_entries),
        "has_phase2ao_reports": bool(ao_entries),
        "all_phase2an_diagnostic_controls_passed": all(entry["passed"] for entry in an_entries)
        and bool(an_entries),
        "all_phase2ao_order_robustness_supported": all(
            entry["sidecar_order_robustness_supported"] for entry in ao_entries
        )
        and bool(ao_entries),
        "all_strict_erased_residual_controls_passed": all(
            entry["strict_erased_residual_control_passed"] for entry in ao_entries
        )
        and bool(ao_entries),
        "all_phase2an_pure_sidecar_controls_passed": all(
            entry["pure_sidecar_dependency_supported"] for entry in an_entries
        )
        and bool(an_entries),
    }
    stable_bounded_sidecar_control = (
        checks["all_phase2an_diagnostic_controls_passed"]
        and checks["all_phase2ao_order_robustness_supported"]
    )
    strict_pure_sidecar_claim_ready = (
        stable_bounded_sidecar_control
        and checks["all_phase2an_pure_sidecar_controls_passed"]
        and checks["all_strict_erased_residual_controls_passed"]
    )
    report = {
        "artifact_family": "phase2ap_sidecar_control_synthesis",
        "passed": stable_bounded_sidecar_control,
        "stable_bounded_sidecar_control_supported": stable_bounded_sidecar_control,
        "strict_pure_sidecar_claim_ready": strict_pure_sidecar_claim_ready,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "metrics": {
            "phase2an_run_count": len(an_entries),
            "phase2ao_run_count": len(ao_entries),
            "min_neutral_original_accuracy": _min_metric(
                an_entries, "neutral_original_accuracy"
            ),
            "min_neutral_original_minus_erased": _min_metric(
                an_entries, "neutral_original_minus_erased"
            ),
            "min_neutral_original_minus_wrong": _min_metric(
                an_entries, "neutral_original_minus_wrong"
            ),
            "min_original_permuted_accuracy": _min_metric(
                ao_entries, "original_permuted_accuracy"
            ),
            "min_original_minus_erased_permuted": _min_metric(
                ao_entries, "original_minus_erased_permuted"
            ),
            "strict_erased_residual_failures": [
                entry["label"]
                for entry in ao_entries
                if not entry["strict_erased_residual_control_passed"]
            ],
            "pure_sidecar_dependency_failures": [
                entry["label"]
                for entry in an_entries
                if not entry["pure_sidecar_dependency_supported"]
            ],
        },
        "phase2an_reports": an_entries,
        "phase2ao_reports": ao_entries,
        "interpretation": (
            "The stable claim is bounded sidecar-controlled command-slot selection under "
            "candidate-text neutralization and candidate-order permutation. Strict pure-sidecar "
            "causality is not ready unless every erased residual control is clean."
        ),
        "claim_boundary": (
            "This synthesis does not support sealed transfer, production autonomy, open-ended "
            "debugging generalization, or an epoch-making architecture."
        ),
        "blocked_actions": [
            "do_not_claim_strict_pure_sidecar_causality_if_any_residual_control_fails",
            "do_not_claim_sealed_transfer_from_phase2ap",
            "do_not_claim_open_ended_debugging_generalization",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
    }
    if output_json:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AP sidecar-control synthesis.")
    parser.add_argument("--phase2an-report", action="append", required=True)
    parser.add_argument("--phase2ao-report", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2ap_sidecar_control_synthesis(
        phase2an_reports=args.phase2an_report,
        phase2ao_reports=args.phase2ao_report,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
