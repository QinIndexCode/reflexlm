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
from reflexlm.cli.audit_phase2fy_replay_bundle_summary_portability_summary import (
    validate_phase2fy_replay_bundle_summary_portability_summary,
)


OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _control_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("control_results", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _copy_control_evidence(
    *,
    phase2fz: dict[str, Any],
    target_dir: Path,
) -> list[dict[str, Any]]:
    copied_rows = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(_control_rows(phase2fz)):
        control_id = str(row.get("control_id", f"control_{index:02d}"))
        case_dir = target_dir / f"c{index:02d}_{control_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        copied = deepcopy(row)
        for key in ("control_report_json", "validation_report_json"):
            source = Path(str(row.get(key, "")))
            if not source.exists():
                copied[f"{key}_copied"] = False
                copied[f"replayed_{key}"] = None
                copied[f"{key}_sha256_matches"] = False
                continue
            target = case_dir / source.name
            shutil.copy2(source, target)
            copied[f"{key}_copied"] = True
            copied[f"replayed_{key}"] = str(target)
            copied[f"{key}_sha256_matches"] = _sha256(source) == _sha256(target)
        copied_rows.append(copied)
    return copied_rows


def _copied_control_file_ok(row: dict[str, Any], key: str) -> bool:
    replayed_key = f"replayed_{key}"
    source = Path(str(row.get(key, "")))
    replayed = Path(str(row.get(replayed_key, "")))
    if not source.exists() or not replayed.exists():
        return False
    return (
        row.get(f"{key}_copied") is True
        and row.get(f"{key}_sha256_matches") is True
        and _sha256(source) == _sha256(replayed)
    )


def validate_phase2ga_replay_bundle_summary_cross_directory_replay(
    report: dict[str, Any],
) -> dict[str, Any]:
    replay_report_path = report.get("evidence", {}).get("replayed_phase2fy_report")
    replay_validation = {}
    replay_report_readable = False
    if replay_report_path:
        try:
            replay_report = _read_json(replay_report_path)
            replay_report_readable = True
            replay_validation = validate_phase2fy_replay_bundle_summary_portability_summary(
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
    copied_control_rows = report.get("replayed_control_results", [])
    if not isinstance(copied_control_rows, list):
        copied_control_rows = []
    copied_control_files_complete = bool(copied_control_rows) and all(
        _copied_control_file_ok(row, "control_report_json")
        and _copied_control_file_ok(row, "validation_report_json")
        for row in copied_control_rows
        if isinstance(row, dict)
    )
    checks = {
        "artifact_family_matches_phase2ga": (
            report.get("artifact_family")
            == "phase2ga_replay_bundle_summary_cross_directory_replay"
        ),
        "top_level_phase2ga_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_summary_replay_claim") is True
            and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2fy_and_phase2fz_passed": (
            report.get("source_summary", {}).get("phase2fy_passed") is True
            and report.get("source_summary", {}).get("phase2fz_passed") is True
        ),
        "source_phase2fy_validation_passed": (
            report.get("source_summary", {}).get("phase2fy_validation_passed") is True
        ),
        "source_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2fz_negative_control_count")
            == report.get("source_summary", {}).get("phase2fz_negative_controls_failed")
        ),
        "replayed_phase2fy_report_readable": replay_report_readable,
        "replayed_phase2fy_validation_passed": replay_validation.get("passed") is True,
        "replayed_markdown_hash_matches_source": markdown_hash_matches,
        "summary_rows_preserved": (
            report.get("replay_summary", {}).get("source_summary_row_count")
            == report.get("replay_summary", {}).get("replayed_summary_row_count")
            and report.get("replay_summary", {}).get("replayed_summary_row_count", 0)
            >= 7
        ),
        "summary_columns_preserved": (
            report.get("replay_summary", {}).get("source_summary_columns")
            == report.get("replay_summary", {}).get("replayed_summary_columns")
        ),
        "control_results_count_preserved": (
            report.get("replay_summary", {}).get("source_control_count")
            == report.get("replay_summary", {}).get("replayed_control_count")
        ),
        "negative_control_failures_preserved": (
            report.get("replay_summary", {}).get("source_negative_controls_failed")
            == report.get("replay_summary", {}).get("replayed_negative_controls_failed")
        ),
        "expected_failed_assertions_preserved": (
            report.get("replay_summary", {}).get(
                "source_expected_failed_check_assertions"
            )
            == report.get("replay_summary", {}).get(
                "replayed_expected_failed_check_assertions"
            )
        ),
        "copied_control_files_complete": copied_control_files_complete,
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
            "replayed_control_count": report.get("replay_summary", {}).get(
                "replayed_control_count", 0
            ),
            "replayed_markdown_hash_matches_source": markdown_hash_matches,
        },
    }


def audit_phase2ga_replay_bundle_summary_cross_directory_replay(
    *,
    phase2fz_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2fz = _read_json(phase2fz_report_json)
    phase2fy_report_json = phase2fz.get("evidence", {}).get("phase2fy_report_json")
    if not phase2fy_report_json:
        raise ValueError("Phase2GA requires Phase2FZ evidence.phase2fy_report_json")
    phase2fy = _read_json(phase2fy_report_json)
    source_validation = validate_phase2fy_replay_bundle_summary_portability_summary(
        phase2fy
    )
    source_markdown_path = Path(
        str(phase2fy.get("evidence", {}).get("portability_summary_markdown", ""))
    )
    if not source_markdown_path.exists():
        raise ValueError("Phase2GA requires a readable Phase2FY markdown summary")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    replay_markdown_path = output_root / "phase2fy_replayed_portability_summary.md"
    shutil.copy2(source_markdown_path, replay_markdown_path)
    replay_phase2fy_report = deepcopy(phase2fy)
    replay_phase2fy_report.setdefault("evidence", {})[
        "portability_summary_markdown"
    ] = str(replay_markdown_path)
    replay_phase2fy_report_path = output_root / "phase2fy_replayed_report.json"
    _write_json(replay_phase2fy_report_path, replay_phase2fy_report)
    replay_validation = validate_phase2fy_replay_bundle_summary_portability_summary(
        replay_phase2fy_report
    )
    replay_validation_path = output_root / "phase2fy_replayed_validation.json"
    _write_json(replay_validation_path, replay_validation)
    replayed_control_rows = _copy_control_evidence(
        phase2fz=phase2fz,
        target_dir=output_root / "control_results",
    )
    replayed_control_rows_path = output_root / "phase2fz_replayed_control_results.json"
    _write_json(replayed_control_rows_path, {"control_results": replayed_control_rows})
    source_rows = phase2fy.get("summary_table", {}).get("rows", [])
    replay_rows = replay_phase2fy_report.get("summary_table", {}).get("rows", [])
    source_columns = phase2fy.get("summary_table", {}).get("columns", [])
    replay_columns = replay_phase2fy_report.get("summary_table", {}).get("columns", [])
    source_markdown_hash = _sha256(source_markdown_path)
    replay_markdown_hash = _sha256(replay_markdown_path)
    replayed_negative_controls_failed = sum(
        row.get("observed_passed") is False
        for row in replayed_control_rows
        if row.get("control_id") != "positive_control_original_phase2fy_summary"
    )
    replayed_expected_assertions = sum(
        len(row.get("expected_failed_checks", []))
        for row in replayed_control_rows
        if isinstance(row.get("expected_failed_checks"), list)
    )
    copied_control_files_complete = bool(replayed_control_rows) and all(
        _copied_control_file_ok(row, "control_report_json")
        and _copied_control_file_ok(row, "validation_report_json")
        for row in replayed_control_rows
    )
    checks = {
        "source_phase2fz_passed": phase2fz.get("passed") is True,
        "source_phase2fy_passed": phase2fy.get("passed") is True,
        "source_phase2fy_validation_passed": source_validation.get("passed") is True,
        "source_phase2fz_negative_controls_complete": (
            phase2fz.get("metrics", {}).get("negative_control_count")
            == phase2fz.get("metrics", {}).get("negative_controls_failed")
        ),
        "replay_phase2fy_report_written": replay_phase2fy_report_path.exists(),
        "replay_phase2fy_validation_passed": replay_validation.get("passed") is True,
        "replay_markdown_written": replay_markdown_path.exists(),
        "replay_markdown_hash_matches_source": source_markdown_hash
        == replay_markdown_hash,
        "summary_rows_preserved": len(source_rows) == len(replay_rows)
        and len(replay_rows) >= 7,
        "summary_columns_preserved": source_columns == replay_columns,
        "control_result_files_copied": copied_control_files_complete,
        "control_result_count_preserved": len(replayed_control_rows)
        == phase2fz.get("metrics", {}).get("control_count"),
        "negative_control_failures_preserved": replayed_negative_controls_failed
        == phase2fz.get("metrics", {}).get("negative_controls_failed"),
        "expected_failed_assertions_preserved": replayed_expected_assertions
        == phase2fz.get("metrics", {}).get("expected_failed_check_assertions"),
        "replay_directory_is_distinct": source_markdown_path.parent.resolve()
        != replay_markdown_path.parent.resolve(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ga_replay_bundle_summary_cross_directory_replay",
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
            "replayed_control_count": len(replayed_control_rows),
            "replayed_negative_controls_failed": replayed_negative_controls_failed,
            "replayed_expected_failed_check_assertions": replayed_expected_assertions,
            "phase2fz_negative_control_count": phase2fz.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2fz_negative_controls_failed": phase2fz.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "source_summary": {
            "phase2fy_passed": phase2fy.get("passed") is True,
            "phase2fz_passed": phase2fz.get("passed") is True,
            "phase2fy_validation_passed": source_validation.get("passed") is True,
            "phase2fz_negative_control_count": phase2fz.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2fz_negative_controls_failed": phase2fz.get("metrics", {}).get(
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
            "source_control_count": phase2fz.get("metrics", {}).get("control_count"),
            "replayed_control_count": len(replayed_control_rows),
            "source_negative_controls_failed": phase2fz.get("metrics", {}).get(
                "negative_controls_failed"
            ),
            "replayed_negative_controls_failed": replayed_negative_controls_failed,
            "source_expected_failed_check_assertions": phase2fz.get("metrics", {}).get(
                "expected_failed_check_assertions"
            ),
            "replayed_expected_failed_check_assertions": replayed_expected_assertions,
        },
        "replayed_control_results": replayed_control_rows,
        "supported_claims": [
            (
                "cross-directory replay of the bounded Phase2FY replay-bundle "
                "summary portability summary with preserved markdown hash, summary "
                "rows, summary columns, validator pass, copied Phase2FZ control "
                "reports, and upstream negative-control closure"
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
            "phase2gb_replay_bundle_summary_replay_negative_controls"
            if passed
            else "repair_phase2ga_replay_bundle_summary_cross_directory_replay"
        ),
        "evidence": {
            "phase2fz_report_json": str(phase2fz_report_json),
            "phase2fy_report_json": str(phase2fy_report_json),
            "source_markdown": str(source_markdown_path),
            "replay_dir": str(output_root),
            "replayed_markdown": str(replay_markdown_path),
            "replayed_phase2fy_report": str(replay_phase2fy_report_path),
            "replayed_phase2fy_validation": str(replay_validation_path),
            "replayed_control_results_dir": str(output_root / "control_results"),
            "replayed_control_results_json": str(replayed_control_rows_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Phase2FY replay-bundle summary in a separate directory."
    )
    parser.add_argument("--phase2fz-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ga_replay_bundle_summary_cross_directory_replay(
        phase2fz_report_json=args.phase2fz_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
