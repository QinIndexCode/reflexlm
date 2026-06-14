from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2dy_replay_bundle_cross_directory_replay import (
    validate_phase2dy_replay_bundle_cross_directory_replay,
)


SUMMARY_COLUMNS: tuple[str, ...] = (
    "Dimension",
    "Observed evidence",
    "Boundary",
)

OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)

REQUIRED_DIMENSIONS: tuple[str, ...] = (
    "source reports",
    "bundle artifacts",
    "reproduction steps",
    "manifest negative controls",
    "replay negative controls",
    "claim boundary",
)


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _negative_controls_complete(report: dict[str, Any]) -> bool:
    metrics = report.get("metrics", {})
    return metrics.get("negative_control_count") == metrics.get("negative_controls_failed")


def _summary_rows(
    *,
    phase2dy: dict[str, Any],
    phase2dx: dict[str, Any],
    phase2dz: dict[str, Any],
) -> list[dict[str, str]]:
    metrics = phase2dy.get("metrics", {})
    dx_metrics = phase2dx.get("metrics", {})
    dz_metrics = phase2dz.get("metrics", {})
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
            "Dimension": "reproduction steps",
            "Observed evidence": (
                f"{metrics.get('replayed_reproduction_step_count')} bounded "
                "replay-bundle steps preserved"
            ),
            "Boundary": "bounded audit commands only",
        },
        {
            "Dimension": "manifest negative controls",
            "Observed evidence": (
                f"{dx_metrics.get('negative_controls_failed')}/"
                f"{dx_metrics.get('negative_control_count')} DW manifest negative "
                "controls failed"
            ),
            "Boundary": "manifest gate strictness only",
        },
        {
            "Dimension": "replay negative controls",
            "Observed evidence": (
                f"{dz_metrics.get('negative_controls_failed')}/"
                f"{dz_metrics.get('negative_control_count')} DY replay negative "
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
        "# Phase2EA Replay Bundle Portability Summary",
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
        report.get("ready_for_bounded_replay_bundle_portability_summary_claim") is True
        and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
    )


def validate_phase2ea_replay_bundle_portability_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    markdown_path = report.get("evidence", {}).get("replay_bundle_portability_summary")
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
        "artifact_family_matches_phase2ea": (
            report.get("artifact_family") == "phase2ea_replay_bundle_portability_summary"
        ),
        "top_level_phase2ea_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": _ready_boundary_ok(report),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2dz_and_phase2dy_passed": (
            report.get("source_summary", {}).get("phase2dz_passed") is True
            and report.get("source_summary", {}).get("phase2dy_passed") is True
        ),
        "source_phase2dy_validation_passed": (
            report.get("source_summary", {}).get("phase2dy_validation_passed") is True
        ),
        "source_phase2dz_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2dz_negative_control_count")
            == report.get("source_summary", {}).get("phase2dz_negative_controls_failed")
        ),
        "source_phase2dx_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2dx_negative_control_count")
            == report.get("source_summary", {}).get("phase2dx_negative_controls_failed")
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
            report.get("metrics", {}).get("replayed_bundle_artifact_count")
            == report.get("source_summary", {}).get("phase2dy_bundle_artifact_count")
            and report.get("metrics", {}).get("replayed_bundle_artifact_hash_match_count")
            == report.get("source_summary", {}).get(
                "phase2dy_bundle_artifact_hash_match_count"
            )
            and report.get("metrics", {}).get("phase2dx_negative_controls_failed")
            == report.get("source_summary", {}).get("phase2dx_negative_controls_failed")
            and report.get("metrics", {}).get("phase2dz_negative_controls_failed")
            == report.get("source_summary", {}).get("phase2dz_negative_controls_failed")
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


def audit_phase2ea_replay_bundle_portability_summary(
    *,
    phase2dz_report_json: str | Path,
    output_report_json: str | Path,
    output_markdown: str | Path,
) -> dict[str, Any]:
    phase2dz = _read_json(phase2dz_report_json)
    phase2dy_report_json = phase2dz.get("evidence", {}).get("phase2dy_report_json")
    if not phase2dy_report_json:
        raise ValueError("Phase2EA requires Phase2DZ evidence.phase2dy_report_json")
    phase2dy = _read_json(phase2dy_report_json)
    phase2dx_report_json = phase2dy.get("evidence", {}).get("phase2dx_report_json")
    phase2dw_report_json = phase2dy.get("evidence", {}).get("phase2dw_report_json")
    if not phase2dx_report_json or not phase2dw_report_json:
        raise ValueError("Phase2EA requires Phase2DY source report evidence paths")
    phase2dx = _read_json(phase2dx_report_json)
    phase2dw = _read_json(phase2dw_report_json)
    phase2dy_validation = validate_phase2dy_replay_bundle_cross_directory_replay(
        phase2dy
    )
    rows = _summary_rows(phase2dy=phase2dy, phase2dx=phase2dx, phase2dz=phase2dz)
    markdown_path = Path(output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown_summary(rows), encoding="utf-8")
    checks = {
        "source_phase2dz_passed": phase2dz.get("passed") is True,
        "source_phase2dy_passed": phase2dy.get("passed") is True,
        "source_phase2dy_validation_passed": phase2dy_validation.get("passed") is True,
        "source_phase2dw_passed": phase2dw.get("passed") is True,
        "source_phase2dx_passed": phase2dx.get("passed") is True,
        "source_phase2dz_negative_controls_complete": _negative_controls_complete(
            phase2dz
        ),
        "source_phase2dx_negative_controls_complete": _negative_controls_complete(
            phase2dx
        ),
        "replay_artifact_hashes_all_match": (
            phase2dy.get("metrics", {}).get("replayed_bundle_artifact_count")
            == phase2dy.get("metrics", {}).get(
                "replayed_bundle_artifact_hash_match_count"
            )
        ),
        "replay_steps_present": phase2dy.get("metrics", {}).get(
            "replayed_reproduction_step_count"
        )
        == 4,
        "summary_markdown_written": markdown_path.exists(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ea_replay_bundle_portability_summary",
        "passed": passed,
        "ready_for_bounded_replay_bundle_portability_summary_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "summary_row_count": len(rows),
            "replayed_source_report_count": phase2dy.get("metrics", {}).get(
                "replayed_source_report_count"
            ),
            "replayed_bundle_artifact_count": phase2dy.get("metrics", {}).get(
                "replayed_bundle_artifact_count"
            ),
            "replayed_bundle_artifact_hash_match_count": phase2dy.get(
                "metrics", {}
            ).get("replayed_bundle_artifact_hash_match_count"),
            "replayed_reproduction_step_count": phase2dy.get("metrics", {}).get(
                "replayed_reproduction_step_count"
            ),
            "phase2dx_negative_control_count": phase2dx.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2dx_negative_controls_failed": phase2dx.get("metrics", {}).get(
                "negative_controls_failed"
            ),
            "phase2dz_negative_control_count": phase2dz.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2dz_negative_controls_failed": phase2dz.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "source_summary": {
            "phase2dw_passed": phase2dw.get("passed") is True,
            "phase2dx_passed": phase2dx.get("passed") is True,
            "phase2dy_passed": phase2dy.get("passed") is True,
            "phase2dz_passed": phase2dz.get("passed") is True,
            "phase2dy_validation_passed": phase2dy_validation.get("passed") is True,
            "phase2dy_bundle_artifact_count": phase2dy.get("metrics", {}).get(
                "replayed_bundle_artifact_count"
            ),
            "phase2dy_bundle_artifact_hash_match_count": phase2dy.get(
                "metrics", {}
            ).get("replayed_bundle_artifact_hash_match_count"),
            "phase2dx_negative_control_count": phase2dx.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2dx_negative_controls_failed": phase2dx.get("metrics", {}).get(
                "negative_controls_failed"
            ),
            "phase2dz_negative_control_count": phase2dz.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2dz_negative_controls_failed": phase2dz.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "summary_table": {
            "columns": list(SUMMARY_COLUMNS),
            "rows": rows,
        },
        "supported_claims": [
            (
                "bounded replay-bundle portability summary for the Phase2DW/DX/DY/DZ "
                "manifest, cross-directory replay, hash preservation, reproduction "
                "step, and negative-control evidence"
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
            "phase2eb_replay_bundle_portability_summary_negative_controls"
            if passed
            else "repair_phase2ea_replay_bundle_portability_summary"
        ),
        "evidence": {
            "phase2dz_report_json": str(phase2dz_report_json),
            "phase2dy_report_json": str(phase2dy_report_json),
            "phase2dx_report_json": str(phase2dx_report_json),
            "phase2dw_report_json": str(phase2dw_report_json),
            "replay_bundle_portability_summary": str(markdown_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2EA replay bundle portability summary."
    )
    parser.add_argument("--phase2dz-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ea_replay_bundle_portability_summary(
        phase2dz_report_json=args.phase2dz_report_json,
        output_report_json=args.output_report_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
