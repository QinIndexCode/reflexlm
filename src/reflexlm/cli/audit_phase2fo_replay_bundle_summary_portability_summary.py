from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2fm_replay_bundle_summary_manifest_cross_directory_replay import (
    validate_phase2fm_replay_bundle_summary_manifest_cross_directory_replay,
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
    phase2fm: dict[str, Any],
    phase2fl: dict[str, Any],
    phase2fn: dict[str, Any],
) -> list[dict[str, str]]:
    metrics = phase2fm.get("metrics", {})
    fl_metrics = phase2fl.get("metrics", {})
    fn_metrics = phase2fn.get("metrics", {})
    return [
        {
            "Dimension": "source reports",
            "Observed evidence": (
                f"{metrics.get('replayed_source_report_count')} source reports replayed"
            ),
            "Boundary": "structured FK manifest source-report replay only",
        },
        {
            "Dimension": "bundle artifacts",
            "Observed evidence": (
                f"{metrics.get('replayed_bundle_artifact_hash_match_count')}/"
                f"{metrics.get('replayed_bundle_artifact_count')} artifact hashes "
                "matched after manifest replay"
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
                f"{fl_metrics.get('negative_controls_failed')}/"
                f"{fl_metrics.get('negative_control_count')} FK manifest negative "
                "controls failed"
            ),
            "Boundary": "manifest gate strictness only",
        },
        {
            "Dimension": "replay negative controls",
            "Observed evidence": (
                f"{fn_metrics.get('negative_controls_failed')}/"
                f"{fn_metrics.get('negative_control_count')} FM replay negative "
                "controls failed"
            ),
            "Boundary": "cross-directory manifest replay gate strictness only",
        },
        {
            "Dimension": "claim boundary",
            "Observed evidence": "all overclaim readiness flags remain false",
            "Boundary": "not free-form shell autonomy or epoch-making architecture",
        },
    ]


def build_markdown_summary(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Phase2FO Replay Bundle Summary Portability Summary",
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
            "Boundary: bounded package-internal structured manifest replay evidence only; "
            "not free-form shell autonomy, not general runtime invariance, not "
            "open-ended native perception, not production autonomy, and not an "
            "epoch-making architecture.",
            "",
        ]
    )
    return "\n".join(lines)


def _ready_boundary_ok(report: dict[str, Any]) -> bool:
    return (
        report.get("ready_for_bounded_manifest_replay_portability_claim") is True
        and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
    )


def validate_phase2fo_replay_bundle_summary_portability_summary(
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
        "artifact_family_matches_phase2fo": (
            report.get("artifact_family")
            == "phase2fo_replay_bundle_summary_portability_summary"
        ),
        "top_level_phase2fo_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": _ready_boundary_ok(report),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2fm_and_phase2fn_passed": (
            report.get("source_summary", {}).get("phase2fm_passed") is True
            and report.get("source_summary", {}).get("phase2fn_passed") is True
        ),
        "source_phase2fm_validation_passed": (
            report.get("source_summary", {}).get("phase2fm_validation_passed") is True
        ),
        "source_phase2fl_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2fl_negative_control_count")
            == report.get("source_summary", {}).get("phase2fl_negative_controls_failed")
        ),
        "source_phase2fn_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2fn_negative_control_count")
            == report.get("source_summary", {}).get("phase2fn_negative_controls_failed")
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
            == report.get("source_summary", {}).get("phase2fm_source_report_count")
            and report.get("metrics", {}).get("replayed_reproduction_step_count")
            == report.get("source_summary", {}).get("phase2fm_reproduction_step_count")
            and report.get("metrics", {}).get("replayed_bundle_artifact_count")
            == report.get("source_summary", {}).get("phase2fm_bundle_artifact_count")
            and report.get("metrics", {}).get(
                "replayed_bundle_artifact_hash_match_count"
            )
            == report.get("source_summary", {}).get(
                "phase2fm_bundle_artifact_hash_match_count"
            )
            and report.get("metrics", {}).get("replayed_control_artifact_count")
            == report.get("source_summary", {}).get("phase2fm_control_artifact_count")
            and report.get("metrics", {}).get("phase2fl_negative_controls_failed")
            == report.get("source_summary", {}).get("phase2fl_negative_controls_failed")
            and report.get("metrics", {}).get("phase2fn_negative_controls_failed")
            == report.get("source_summary", {}).get("phase2fn_negative_controls_failed")
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


def audit_phase2fo_replay_bundle_summary_portability_summary(
    *,
    phase2fn_report_json: str | Path,
    output_report_json: str | Path,
    output_markdown: str | Path,
) -> dict[str, Any]:
    phase2fn = _read_json(phase2fn_report_json)
    phase2fm_report_json = phase2fn.get("evidence", {}).get("phase2fm_report_json")
    if not phase2fm_report_json:
        raise ValueError("Phase2FO requires Phase2FN evidence.phase2fm_report_json")
    phase2fm = _read_json(phase2fm_report_json)
    phase2fl_report_json = phase2fm.get("evidence", {}).get("phase2fl_report_json")
    phase2fk_report_json = phase2fm.get("evidence", {}).get("phase2fk_report_json")
    if not phase2fl_report_json or not phase2fk_report_json:
        raise ValueError("Phase2FO requires Phase2FM source report evidence paths")
    phase2fl = _read_json(phase2fl_report_json)
    phase2fk = _read_json(phase2fk_report_json)
    phase2fm_validation = (
        validate_phase2fm_replay_bundle_summary_manifest_cross_directory_replay(
            phase2fm
        )
    )
    rows = _summary_rows(phase2fm=phase2fm, phase2fl=phase2fl, phase2fn=phase2fn)
    markdown_path = Path(output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown_summary(rows), encoding="utf-8")
    checks = {
        "source_phase2fn_passed": phase2fn.get("passed") is True,
        "source_phase2fm_passed": phase2fm.get("passed") is True,
        "source_phase2fm_validation_passed": phase2fm_validation.get("passed") is True,
        "source_phase2fl_passed": phase2fl.get("passed") is True,
        "source_phase2fk_passed": phase2fk.get("passed") is True,
        "source_phase2fl_negative_controls_complete": (
            phase2fl.get("metrics", {}).get("negative_control_count")
            == phase2fl.get("metrics", {}).get("negative_controls_failed")
        ),
        "source_phase2fn_negative_controls_complete": (
            phase2fn.get("metrics", {}).get("negative_control_count")
            == phase2fn.get("metrics", {}).get("negative_controls_failed")
        ),
        "replay_artifact_hashes_all_match": (
            phase2fm.get("metrics", {}).get("replayed_bundle_artifact_count")
            == phase2fm.get("metrics", {}).get(
                "replayed_bundle_artifact_hash_match_count"
            )
        ),
        "control_artifact_count_preserved": (
            phase2fm.get("metrics", {}).get("replayed_control_artifact_count")
            == phase2fk.get("metrics", {}).get("control_artifact_count")
        ),
        "replay_steps_present": phase2fm.get("metrics", {}).get(
            "replayed_reproduction_step_count"
        )
        == 4,
        "summary_markdown_written": markdown_path.exists(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2fo_replay_bundle_summary_portability_summary",
        "passed": passed,
        "ready_for_bounded_manifest_replay_portability_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "summary_row_count": len(rows),
            "replayed_source_report_count": phase2fm.get("metrics", {}).get(
                "replayed_source_report_count"
            ),
            "replayed_bundle_artifact_count": phase2fm.get("metrics", {}).get(
                "replayed_bundle_artifact_count"
            ),
            "replayed_bundle_artifact_hash_match_count": phase2fm.get(
                "metrics", {}
            ).get("replayed_bundle_artifact_hash_match_count"),
            "replayed_control_artifact_count": phase2fm.get("metrics", {}).get(
                "replayed_control_artifact_count"
            ),
            "replayed_reproduction_step_count": phase2fm.get("metrics", {}).get(
                "replayed_reproduction_step_count"
            ),
            "phase2fl_negative_control_count": phase2fl.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2fl_negative_controls_failed": phase2fl.get("metrics", {}).get(
                "negative_controls_failed"
            ),
            "phase2fn_negative_control_count": phase2fn.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2fn_negative_controls_failed": phase2fn.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "source_summary": {
            "phase2fk_passed": phase2fk.get("passed") is True,
            "phase2fl_passed": phase2fl.get("passed") is True,
            "phase2fm_passed": phase2fm.get("passed") is True,
            "phase2fn_passed": phase2fn.get("passed") is True,
            "phase2fm_validation_passed": phase2fm_validation.get("passed") is True,
            "phase2fm_source_report_count": phase2fm.get("metrics", {}).get(
                "replayed_source_report_count"
            ),
            "phase2fm_reproduction_step_count": phase2fm.get("metrics", {}).get(
                "replayed_reproduction_step_count"
            ),
            "phase2fm_bundle_artifact_count": phase2fm.get("metrics", {}).get(
                "replayed_bundle_artifact_count"
            ),
            "phase2fm_bundle_artifact_hash_match_count": phase2fm.get(
                "metrics", {}
            ).get("replayed_bundle_artifact_hash_match_count"),
            "phase2fm_control_artifact_count": phase2fm.get("metrics", {}).get(
                "replayed_control_artifact_count"
            ),
            "phase2fl_negative_control_count": phase2fl.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2fl_negative_controls_failed": phase2fl.get("metrics", {}).get(
                "negative_controls_failed"
            ),
            "phase2fn_negative_control_count": phase2fn.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2fn_negative_controls_failed": phase2fn.get("metrics", {}).get(
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
                "Phase2FK/FL/FM/FN manifest, cross-directory replay, copied-control "
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
            "phase2fp_replay_bundle_summary_portability_summary_negative_controls"
            if passed
            else "repair_phase2fo_replay_bundle_summary_portability_summary"
        ),
        "evidence": {
            "phase2fn_report_json": str(phase2fn_report_json),
            "phase2fm_report_json": str(phase2fm_report_json),
            "phase2fl_report_json": str(phase2fl_report_json),
            "phase2fk_report_json": str(phase2fk_report_json),
            "portability_summary_markdown": str(markdown_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2FO replay bundle summary portability summary."
    )
    parser.add_argument("--phase2fn-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2fo_replay_bundle_summary_portability_summary(
        phase2fn_report_json=args.phase2fn_report_json,
        output_report_json=args.output_report_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
