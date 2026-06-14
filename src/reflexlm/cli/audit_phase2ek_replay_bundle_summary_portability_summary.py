from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2ei_replay_bundle_summary_manifest_cross_directory_replay import (
    validate_phase2ei_replay_bundle_summary_manifest_cross_directory_replay,
)


SUMMARY_COLUMNS: tuple[str, ...] = (
    "Dimension",
    "Observed evidence",
    "Boundary",
)

REQUIRED_DIMENSIONS: tuple[str, ...] = (
    "source reports",
    "bundle artifacts",
    "control artifacts",
    "reproduction steps",
    "manifest negative controls",
    "replay negative controls",
    "claim boundary",
)

OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _summary_rows(
    *,
    phase2ei: dict[str, Any],
    phase2eh: dict[str, Any],
    phase2ej: dict[str, Any],
) -> list[dict[str, str]]:
    metrics = phase2ei.get("metrics", {})
    eh_metrics = phase2eh.get("metrics", {})
    ej_metrics = phase2ej.get("metrics", {})
    return [
        {
            "Dimension": "source reports",
            "Observed evidence": (
                f"{metrics.get('replayed_source_report_count')} source reports replayed"
            ),
            "Boundary": "structured source-report replay only",
        },
        {
            "Dimension": "bundle artifacts",
            "Observed evidence": (
                f"{metrics.get('replayed_bundle_artifact_hash_match_count')}/"
                f"{metrics.get('replayed_bundle_artifact_count')} artifact hashes "
                "matched after replay"
            ),
            "Boundary": "hash portability, not runtime autonomy",
        },
        {
            "Dimension": "control artifacts",
            "Observed evidence": (
                f"{metrics.get('replayed_control_artifact_count')} copied-control "
                "artifacts preserved"
            ),
            "Boundary": "artifact integrity only",
        },
        {
            "Dimension": "reproduction steps",
            "Observed evidence": (
                f"{metrics.get('replayed_reproduction_step_count')} bounded "
                "manifest-replay steps preserved"
            ),
            "Boundary": "bounded audit commands only",
        },
        {
            "Dimension": "manifest negative controls",
            "Observed evidence": (
                f"{eh_metrics.get('negative_controls_failed')}/"
                f"{eh_metrics.get('negative_control_count')} EG manifest negative "
                "controls failed"
            ),
            "Boundary": "manifest gate strictness only",
        },
        {
            "Dimension": "replay negative controls",
            "Observed evidence": (
                f"{ej_metrics.get('negative_controls_failed')}/"
                f"{ej_metrics.get('negative_control_count')} EI replay negative "
                "controls failed"
            ),
            "Boundary": "cross-directory replay gate strictness only",
        },
        {
            "Dimension": "claim boundary",
            "Observed evidence": "all overclaim readiness flags remain false",
            "Boundary": "not free-form shell autonomy or epoch-making architecture",
        },
    ]


def build_markdown_summary(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Phase2EK Replay Bundle Summary Portability Summary",
        "",
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_markdown_escape(row.get(column, "")) for column in SUMMARY_COLUMNS)
            + " |"
        )
    lines.extend(
        [
            "",
            "Boundary: bounded package-internal structured runtime evidence only; "
            "not free-form shell autonomy, not general runtime invariance, not "
            "open-ended native perception, not production autonomy, and not an "
            "epoch-making architecture.",
            "",
        ]
    )
    return "\n".join(lines)


def _ready_boundary_ok(report: dict[str, Any]) -> bool:
    return (
        report.get("ready_for_bounded_replay_bundle_summary_portability_claim") is True
        and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
    )


def validate_phase2ek_replay_bundle_summary_portability_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    markdown_path = report.get("evidence", {}).get("portability_summary_markdown")
    markdown_readable = False
    markdown_text = ""
    if markdown_path:
        try:
            markdown_text = Path(markdown_path).read_text(encoding="utf-8")
            markdown_readable = True
        except OSError:
            markdown_readable = False
    rows = report.get("summary_table", {}).get("rows", [])
    if not isinstance(rows, list):
        rows = []
    dimensions = {str(row.get("Dimension")) for row in rows if isinstance(row, dict)}
    checks = {
        "artifact_family_matches_phase2ek": (
            report.get("artifact_family")
            == "phase2ek_replay_bundle_summary_portability_summary"
        ),
        "top_level_phase2ek_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": _ready_boundary_ok(report),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2ei_and_phase2ej_passed": (
            report.get("source_summary", {}).get("phase2ei_passed") is True
            and report.get("source_summary", {}).get("phase2ej_passed") is True
        ),
        "source_phase2ei_validation_passed": (
            report.get("source_summary", {}).get("phase2ei_validation_passed") is True
        ),
        "source_phase2eh_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2eh_negative_control_count")
            == report.get("source_summary", {}).get("phase2eh_negative_controls_failed")
        ),
        "source_phase2ej_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2ej_negative_control_count")
            == report.get("source_summary", {}).get("phase2ej_negative_controls_failed")
        ),
        "summary_rows_present": len(rows) >= len(REQUIRED_DIMENSIONS),
        "summary_columns_match": report.get("summary_table", {}).get("columns")
        == list(SUMMARY_COLUMNS),
        "summary_dimensions_complete": set(REQUIRED_DIMENSIONS).issubset(dimensions),
        "markdown_summary_readable": markdown_readable,
        "markdown_contains_all_dimensions": bool(rows)
        and all(dimension in markdown_text for dimension in dimensions),
        "markdown_contains_bounded_boundary": (
            "not free-form shell autonomy" in markdown_text
            and "not an epoch-making architecture" in markdown_text
        ),
        "metrics_match_source": (
            report.get("metrics", {}).get("replayed_source_report_count")
            == report.get("source_summary", {}).get("phase2ei_source_report_count")
            and report.get("metrics", {}).get("replayed_reproduction_step_count")
            == report.get("source_summary", {}).get("phase2ei_reproduction_step_count")
            and report.get("metrics", {}).get("replayed_bundle_artifact_count")
            == report.get("source_summary", {}).get("phase2ei_bundle_artifact_count")
            and report.get("metrics", {}).get(
                "replayed_bundle_artifact_hash_match_count"
            )
            == report.get("source_summary", {}).get(
                "phase2ei_bundle_artifact_hash_match_count"
            )
            and report.get("metrics", {}).get("replayed_control_artifact_count")
            == report.get("source_summary", {}).get("phase2ei_control_artifact_count")
            and report.get("metrics", {}).get("phase2eh_negative_controls_failed")
            == report.get("source_summary", {}).get("phase2eh_negative_controls_failed")
            and report.get("metrics", {}).get("phase2ej_negative_controls_failed")
            == report.get("source_summary", {}).get("phase2ej_negative_controls_failed")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "summary_row_count": len(rows),
            "markdown_readable": markdown_readable,
            "markdown_bytes": len(markdown_text.encode("utf-8")) if markdown_text else 0,
        },
    }


def audit_phase2ek_replay_bundle_summary_portability_summary(
    *,
    phase2ej_report_json: str | Path,
    output_report_json: str | Path,
    output_markdown: str | Path,
) -> dict[str, Any]:
    phase2ej = _read_json(phase2ej_report_json)
    phase2ei_report_json = phase2ej.get("evidence", {}).get("phase2ei_report_json")
    if not phase2ei_report_json:
        raise ValueError("Phase2EK requires Phase2EJ evidence.phase2ei_report_json")
    phase2ei = _read_json(phase2ei_report_json)
    phase2eh_report_json = phase2ei.get("evidence", {}).get("phase2eh_report_json")
    phase2eg_report_json = phase2ei.get("evidence", {}).get("phase2eg_report_json")
    if not phase2eh_report_json or not phase2eg_report_json:
        raise ValueError("Phase2EK requires Phase2EI source report evidence paths")
    phase2eh = _read_json(phase2eh_report_json)
    phase2eg = _read_json(phase2eg_report_json)
    phase2ei_validation = validate_phase2ei_replay_bundle_summary_manifest_cross_directory_replay(
        phase2ei
    )
    rows = _summary_rows(phase2ei=phase2ei, phase2eh=phase2eh, phase2ej=phase2ej)
    markdown_path = Path(output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown_summary(rows), encoding="utf-8")
    checks = {
        "source_phase2ej_passed": phase2ej.get("passed") is True,
        "source_phase2ei_passed": phase2ei.get("passed") is True,
        "source_phase2ei_validation_passed": phase2ei_validation.get("passed") is True,
        "source_phase2eh_passed": phase2eh.get("passed") is True,
        "source_phase2eg_passed": phase2eg.get("passed") is True,
        "source_phase2eh_negative_controls_complete": (
            phase2eh.get("metrics", {}).get("negative_control_count")
            == phase2eh.get("metrics", {}).get("negative_controls_failed")
        ),
        "source_phase2ej_negative_controls_complete": (
            phase2ej.get("metrics", {}).get("negative_control_count")
            == phase2ej.get("metrics", {}).get("negative_controls_failed")
        ),
        "replay_artifact_hashes_all_match": (
            phase2ei.get("metrics", {}).get("replayed_bundle_artifact_count")
            == phase2ei.get("metrics", {}).get(
                "replayed_bundle_artifact_hash_match_count"
            )
        ),
        "control_artifact_count_preserved": (
            phase2ei.get("metrics", {}).get("replayed_control_artifact_count")
            == phase2eg.get("metrics", {}).get("control_artifact_count")
        ),
        "replay_steps_present": phase2ei.get("metrics", {}).get(
            "replayed_reproduction_step_count"
        )
        == 4,
        "summary_markdown_written": markdown_path.exists(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ek_replay_bundle_summary_portability_summary",
        "passed": passed,
        "ready_for_bounded_replay_bundle_summary_portability_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "summary_row_count": len(rows),
            "replayed_source_report_count": phase2ei.get("metrics", {}).get(
                "replayed_source_report_count"
            ),
            "replayed_bundle_artifact_count": phase2ei.get("metrics", {}).get(
                "replayed_bundle_artifact_count"
            ),
            "replayed_bundle_artifact_hash_match_count": phase2ei.get(
                "metrics", {}
            ).get("replayed_bundle_artifact_hash_match_count"),
            "replayed_control_artifact_count": phase2ei.get("metrics", {}).get(
                "replayed_control_artifact_count"
            ),
            "replayed_reproduction_step_count": phase2ei.get("metrics", {}).get(
                "replayed_reproduction_step_count"
            ),
            "phase2eh_negative_control_count": phase2eh.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2eh_negative_controls_failed": phase2eh.get("metrics", {}).get(
                "negative_controls_failed"
            ),
            "phase2ej_negative_control_count": phase2ej.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2ej_negative_controls_failed": phase2ej.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "source_summary": {
            "phase2eg_passed": phase2eg.get("passed") is True,
            "phase2eh_passed": phase2eh.get("passed") is True,
            "phase2ei_passed": phase2ei.get("passed") is True,
            "phase2ej_passed": phase2ej.get("passed") is True,
            "phase2ei_validation_passed": phase2ei_validation.get("passed") is True,
            "phase2ei_source_report_count": phase2ei.get("metrics", {}).get(
                "replayed_source_report_count"
            ),
            "phase2ei_reproduction_step_count": phase2ei.get("metrics", {}).get(
                "replayed_reproduction_step_count"
            ),
            "phase2ei_bundle_artifact_count": phase2ei.get("metrics", {}).get(
                "replayed_bundle_artifact_count"
            ),
            "phase2ei_bundle_artifact_hash_match_count": phase2ei.get(
                "metrics", {}
            ).get("replayed_bundle_artifact_hash_match_count"),
            "phase2ei_control_artifact_count": phase2ei.get("metrics", {}).get(
                "replayed_control_artifact_count"
            ),
            "phase2eh_negative_control_count": phase2eh.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2eh_negative_controls_failed": phase2eh.get("metrics", {}).get(
                "negative_controls_failed"
            ),
            "phase2ej_negative_control_count": phase2ej.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2ej_negative_controls_failed": phase2ej.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "summary_table": {
            "columns": list(SUMMARY_COLUMNS),
            "rows": rows,
        },
        "supported_claims": [
            (
                "bounded replay-bundle summary portability summary for the "
                "Phase2EG/EH/EI/EJ manifest, cross-directory replay, copied-control "
                "artifact, hash preservation, reproduction-step, and negative-control evidence"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "free-form shell autonomy",
            "arbitrary shell/environment generalization",
            "general runtime invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2el_replay_bundle_summary_portability_summary_negative_controls"
            if passed
            else "repair_phase2ek_replay_bundle_summary_portability_summary"
        ),
        "evidence": {
            "phase2ej_report_json": str(phase2ej_report_json),
            "phase2ei_report_json": str(phase2ei_report_json),
            "phase2eh_report_json": str(phase2eh_report_json),
            "phase2eg_report_json": str(phase2eg_report_json),
            "portability_summary_markdown": str(markdown_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2EK replay bundle summary portability summary."
    )
    parser.add_argument("--phase2ej-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ek_replay_bundle_summary_portability_summary(
        phase2ej_report_json=args.phase2ej_report_json,
        output_report_json=args.output_report_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
