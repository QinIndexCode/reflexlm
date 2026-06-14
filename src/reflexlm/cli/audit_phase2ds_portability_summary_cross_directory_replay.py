from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2dk_latex_candidate_publication_bundle import _sha256
from reflexlm.cli.audit_phase2dq_replay_bundle_portability_summary import (
    validate_phase2dq_replay_bundle_portability_summary,
)


OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def validate_phase2ds_portability_summary_cross_directory_replay(
    report: dict[str, Any],
) -> dict[str, Any]:
    replay_report_path = report.get("evidence", {}).get("replayed_phase2dq_report")
    replay_validation = {}
    replay_report_readable = False
    if replay_report_path:
        try:
            replay_report = _read_json(replay_report_path)
            replay_report_readable = True
            replay_validation = validate_phase2dq_replay_bundle_portability_summary(
                replay_report
            )
        except (OSError, json.JSONDecodeError):
            replay_report_readable = False
            replay_validation = {}
    replay_markdown = Path(str(report.get("evidence", {}).get("replayed_markdown", "")))
    markdown_hash_matches = (
        replay_markdown.exists()
        and report.get("replay_summary", {}).get("source_markdown_sha256")
        == report.get("replay_summary", {}).get("replayed_markdown_sha256")
        == _sha256(replay_markdown)
    )
    checks = {
        "artifact_family_matches_phase2ds": (
            report.get("artifact_family")
            == "phase2ds_portability_summary_cross_directory_replay"
        ),
        "top_level_phase2ds_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_summary_replay_claim") is True
            and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2dq_and_phase2dr_passed": (
            report.get("source_summary", {}).get("phase2dq_passed") is True
            and report.get("source_summary", {}).get("phase2dr_passed") is True
        ),
        "source_phase2dq_validation_passed": (
            report.get("source_summary", {}).get("phase2dq_validation_passed") is True
        ),
        "source_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2dr_negative_control_count")
            == report.get("source_summary", {}).get("phase2dr_negative_controls_failed")
        ),
        "replayed_phase2dq_report_readable": replay_report_readable,
        "replayed_phase2dq_validation_passed": replay_validation.get("passed") is True,
        "replayed_markdown_hash_matches_source": markdown_hash_matches,
        "summary_rows_preserved": (
            report.get("replay_summary", {}).get("source_summary_row_count")
            == report.get("replay_summary", {}).get("replayed_summary_row_count")
            and report.get("replay_summary", {}).get("replayed_summary_row_count", 0)
            >= 5
        ),
        "summary_columns_preserved": (
            report.get("replay_summary", {}).get("source_summary_columns")
            == report.get("replay_summary", {}).get("replayed_summary_columns")
        ),
        "replay_directory_is_distinct": (
            report.get("replay_summary", {}).get("source_markdown_parent")
            != report.get("replay_summary", {}).get("replay_markdown_parent")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "replayed_summary_row_count": report.get("replay_summary", {}).get(
                "replayed_summary_row_count", 0
            ),
            "replayed_markdown_hash_matches_source": markdown_hash_matches,
        },
    }


def audit_phase2ds_portability_summary_cross_directory_replay(
    *,
    phase2dr_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2dr = _read_json(phase2dr_report_json)
    phase2dq_report_json = phase2dr.get("evidence", {}).get("phase2dq_report_json")
    if not phase2dq_report_json:
        raise ValueError("Phase2DS requires Phase2DR evidence.phase2dq_report_json")
    phase2dq = _read_json(phase2dq_report_json)
    source_validation = validate_phase2dq_replay_bundle_portability_summary(phase2dq)
    source_markdown_path = Path(
        str(phase2dq.get("evidence", {}).get("portability_summary_markdown", ""))
    )
    if not source_markdown_path.exists():
        raise ValueError("Phase2DS requires a readable Phase2DQ markdown summary")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    replay_markdown_path = output_root / "phase2dq_replayed_portability_summary.md"
    shutil.copy2(source_markdown_path, replay_markdown_path)
    replay_phase2dq_report = deepcopy(phase2dq)
    replay_phase2dq_report.setdefault("evidence", {})[
        "portability_summary_markdown"
    ] = str(replay_markdown_path)
    replay_phase2dq_report_path = output_root / "phase2dq_replayed_report.json"
    _write_json(replay_phase2dq_report_path, replay_phase2dq_report)
    replay_validation = validate_phase2dq_replay_bundle_portability_summary(
        replay_phase2dq_report
    )
    replay_validation_path = output_root / "phase2dq_replayed_validation.json"
    _write_json(replay_validation_path, replay_validation)
    source_rows = phase2dq.get("summary_table", {}).get("rows", [])
    replay_rows = replay_phase2dq_report.get("summary_table", {}).get("rows", [])
    source_columns = phase2dq.get("summary_table", {}).get("columns", [])
    replay_columns = replay_phase2dq_report.get("summary_table", {}).get("columns", [])
    source_markdown_hash = _sha256(source_markdown_path)
    replay_markdown_hash = _sha256(replay_markdown_path)
    checks = {
        "source_phase2dr_passed": phase2dr.get("passed") is True,
        "source_phase2dq_passed": phase2dq.get("passed") is True,
        "source_phase2dq_validation_passed": source_validation.get("passed") is True,
        "source_phase2dr_negative_controls_complete": (
            phase2dr.get("metrics", {}).get("negative_control_count")
            == phase2dr.get("metrics", {}).get("negative_controls_failed")
        ),
        "replay_phase2dq_report_written": replay_phase2dq_report_path.exists(),
        "replay_phase2dq_validation_passed": replay_validation.get("passed") is True,
        "replay_markdown_written": replay_markdown_path.exists(),
        "replay_markdown_hash_matches_source": source_markdown_hash
        == replay_markdown_hash,
        "summary_rows_preserved": len(source_rows) == len(replay_rows) and len(replay_rows) >= 5,
        "summary_columns_preserved": source_columns == replay_columns,
        "replay_directory_is_distinct": source_markdown_path.parent.resolve()
        != replay_markdown_path.parent.resolve(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ds_portability_summary_cross_directory_replay",
        "passed": passed,
        "ready_for_bounded_summary_replay_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "replayed_summary_row_count": len(replay_rows),
            "source_markdown_bytes": source_markdown_path.stat().st_size,
            "replayed_markdown_bytes": replay_markdown_path.stat().st_size,
            "phase2dr_negative_control_count": phase2dr.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2dr_negative_controls_failed": phase2dr.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "source_summary": {
            "phase2dq_passed": phase2dq.get("passed") is True,
            "phase2dr_passed": phase2dr.get("passed") is True,
            "phase2dq_validation_passed": source_validation.get("passed") is True,
            "phase2dr_negative_control_count": phase2dr.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2dr_negative_controls_failed": phase2dr.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "replay_summary": {
            "source_summary_row_count": len(source_rows),
            "replayed_summary_row_count": len(replay_rows),
            "source_summary_columns": source_columns,
            "replayed_summary_columns": replay_columns,
            "source_markdown_parent": str(source_markdown_path.parent),
            "replay_markdown_parent": str(replay_markdown_path.parent),
            "source_markdown_sha256": source_markdown_hash,
            "replayed_markdown_sha256": replay_markdown_hash,
        },
        "supported_claims": [
            (
                "cross-directory replay of the bounded Phase2DQ portability summary "
                "with preserved markdown hash, summary rows, summary columns, "
                "validator pass, and upstream Phase2DR negative-control closure"
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
            "phase2dt_portability_summary_replay_negative_controls"
            if passed
            else "repair_phase2ds_portability_summary_cross_directory_replay"
        ),
        "evidence": {
            "phase2dr_report_json": str(phase2dr_report_json),
            "phase2dq_report_json": str(phase2dq_report_json),
            "source_markdown": str(source_markdown_path),
            "replay_dir": str(output_root),
            "replayed_markdown": str(replay_markdown_path),
            "replayed_phase2dq_report": str(replay_phase2dq_report_path),
            "replayed_phase2dq_validation": str(replay_validation_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Phase2DQ portability summary in a separate directory."
    )
    parser.add_argument("--phase2dr-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ds_portability_summary_cross_directory_replay(
        phase2dr_report_json=args.phase2dr_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
