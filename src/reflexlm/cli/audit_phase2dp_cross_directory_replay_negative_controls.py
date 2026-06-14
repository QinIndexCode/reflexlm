from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any, Callable

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2do_reproducibility_manifest_cross_directory_replay import (
    validate_phase2do_reproducibility_manifest_cross_directory_replay,
)


Mutation = Callable[[dict[str, Any], Path], None]


def _copy_replay_tree(
    *,
    phase2do_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, str]:
    source_replay_dir = Path(str(phase2do_report.get("evidence", {}).get("replay_dir", "")))
    if not source_replay_dir.exists():
        raise ValueError("Phase2DP requires a readable Phase2DO replay directory")
    target_replay_dir = case_dir / "replay"
    if target_replay_dir.exists():
        shutil.rmtree(target_replay_dir)
    shutil.copytree(source_replay_dir, target_replay_dir)
    path_map: dict[str, str] = {}
    for source_path in source_replay_dir.rglob("*"):
        if source_path.is_file():
            relative = source_path.relative_to(source_replay_dir)
            path_map[str(source_path)] = str(target_replay_dir / relative)
    path_map[str(source_replay_dir)] = str(target_replay_dir)
    return path_map


def _rewrite_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [_rewrite_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, path_map) for key, item in value.items()}
    return value


def _materialize_control_report(
    *,
    phase2do_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    control_report = deepcopy(phase2do_report)
    path_map = _copy_replay_tree(phase2do_report=phase2do_report, case_dir=case_dir)
    control_report["evidence"] = _rewrite_paths(
        control_report.get("evidence", {}),
        path_map,
    )
    control_report["replay_summary"] = _rewrite_paths(
        control_report.get("replay_summary", {}),
        path_map,
    )
    replay_report = _read_json(control_report["evidence"]["replayed_phase2dm_report"])
    replay_report = _rewrite_paths(replay_report, path_map)
    _write_json(control_report["evidence"]["replayed_phase2dm_report"], replay_report)
    replay_manifest_path = Path(
        str(replay_report.get("evidence", {}).get("reproducibility_manifest", ""))
    )
    replay_manifest = _read_json(replay_manifest_path)
    replay_manifest = _rewrite_paths(replay_manifest, path_map)
    _write_json(replay_manifest_path, replay_manifest)
    return control_report


def _replay_report_path(report: dict[str, Any]) -> Path:
    return Path(str(report.get("evidence", {}).get("replayed_phase2dm_report", "")))


def _replay_manifest_path(report: dict[str, Any]) -> Path:
    replay_report = _read_json(_replay_report_path(report))
    return Path(str(replay_report.get("evidence", {}).get("reproducibility_manifest", "")))


def _read_replay_manifest(report: dict[str, Any]) -> dict[str, Any]:
    return _read_json(_replay_manifest_path(report))


def _write_replay_manifest(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    _write_json(_replay_manifest_path(report), manifest)


def _bundle_entry(report: dict[str, Any], role: str) -> dict[str, Any]:
    manifest = _read_replay_manifest(report)
    for item in manifest.get("bundle_artifacts", []):
        if isinstance(item, dict) and item.get("role") == role:
            return item
    raise ValueError(f"missing replay bundle artifact role: {role}")


def _mutate_source_summary_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2dm_passed"] = False


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["replay_phase2dm_validation_passed"] = False


def _mutate_source_negative_controls_incomplete(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2dn_negative_controls_failed"] = 0


def _mutate_missing_replayed_report(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    replay_report = _replay_report_path(report)
    if replay_report.exists():
        replay_report.unlink()


def _mutate_replayed_report_bad_json(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    _replay_report_path(report).write_text("{bad json", encoding="utf-8")


def _mutate_replayed_manifest_missing(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _replay_manifest_path(report)
    if manifest.exists():
        manifest.unlink()


def _mutate_replayed_bundle_hash(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    readme = Path(str(_bundle_entry(report, "bundle_readme")["path"]))
    readme.write_text("tampered\n", encoding="utf-8")


def _mutate_missing_replayed_bundle_file(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    latex = Path(str(_bundle_entry(report, "latex_candidate")["path"]))
    if latex.exists():
        latex.unlink()


def _mutate_missing_source_role(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("replay_summary", {})["source_report_roles"] = [
        role
        for role in report.get("replay_summary", {}).get("source_report_roles", [])
        if role != "phase2di_report"
    ]


def _mutate_missing_reproduction_step(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("replay_summary", {})["reproduction_step_ids"] = (
        report.get("replay_summary", {}).get("reproduction_step_ids", [])[:-1]
    )


def _mutate_replay_directory_not_distinct(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("replay_summary", {})["replay_manifest_parent"] = report.get(
        "replay_summary", {}
    ).get("source_manifest_parent")


def _mutate_bundle_hash_count_collapsed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("replay_summary", {})["bundle_artifact_hash_match_count"] = 0


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2do_replay",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_summary_failed",
        "mutation": _mutate_source_summary_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2dm_and_phase2dn_passed"],
    },
    {
        "control_id": "negative_recorded_check_false",
        "mutation": _mutate_recorded_check_false,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_source_negative_controls_incomplete",
        "mutation": _mutate_source_negative_controls_incomplete,
        "expected_passed": False,
        "expected_failed_checks": ["source_negative_controls_complete"],
    },
    {
        "control_id": "negative_missing_replayed_report",
        "mutation": _mutate_missing_replayed_report,
        "expected_passed": False,
        "expected_failed_checks": [
            "replayed_phase2dm_report_readable",
            "replayed_phase2dm_validation_passed",
        ],
    },
    {
        "control_id": "negative_replayed_report_bad_json",
        "mutation": _mutate_replayed_report_bad_json,
        "expected_passed": False,
        "expected_failed_checks": [
            "replayed_phase2dm_report_readable",
            "replayed_phase2dm_validation_passed",
        ],
    },
    {
        "control_id": "negative_replayed_manifest_missing",
        "mutation": _mutate_replayed_manifest_missing,
        "expected_passed": False,
        "expected_failed_checks": ["replayed_phase2dm_validation_passed"],
    },
    {
        "control_id": "negative_replayed_bundle_hash_tampered",
        "mutation": _mutate_replayed_bundle_hash,
        "expected_passed": False,
        "expected_failed_checks": ["replayed_phase2dm_validation_passed"],
    },
    {
        "control_id": "negative_missing_replayed_bundle_file",
        "mutation": _mutate_missing_replayed_bundle_file,
        "expected_passed": False,
        "expected_failed_checks": ["replayed_phase2dm_validation_passed"],
    },
    {
        "control_id": "negative_missing_source_role",
        "mutation": _mutate_missing_source_role,
        "expected_passed": False,
        "expected_failed_checks": ["replayed_source_report_roles_complete"],
    },
    {
        "control_id": "negative_missing_reproduction_step",
        "mutation": _mutate_missing_reproduction_step,
        "expected_passed": False,
        "expected_failed_checks": ["replayed_reproduction_steps_complete"],
    },
    {
        "control_id": "negative_replay_directory_not_distinct",
        "mutation": _mutate_replay_directory_not_distinct,
        "expected_passed": False,
        "expected_failed_checks": ["replay_directory_is_distinct"],
    },
    {
        "control_id": "negative_bundle_hash_count_collapsed",
        "mutation": _mutate_bundle_hash_count_collapsed,
        "expected_passed": False,
        "expected_failed_checks": ["all_replayed_bundle_artifact_hashes_match"],
    },
    {
        "control_id": "negative_overstated_epoch_claim",
        "mutation": _mutate_overstated_epoch_claim,
        "expected_passed": False,
        "expected_failed_checks": ["top_level_ready_claim_is_bounded"],
    },
)


def _run_control(
    *,
    control_spec: dict[str, Any],
    control_index: int,
    phase2do_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2do_report=phase2do_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2do_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2do_reproducibility_manifest_cross_directory_replay(
        control_report
    )
    validation_report_json = case_dir / "phase2do_validation.json"
    _write_json(validation_report_json, validation)
    expected_passed = bool(control_spec["expected_passed"])
    expected_failed_checks = list(control_spec["expected_failed_checks"])
    failed_checks = [
        name
        for name, passed in validation.get("checks", {}).items()
        if passed is False
    ]
    return {
        "control_id": control_id,
        "expected_passed": expected_passed,
        "observed_passed": validation.get("passed") is True,
        "pass_expectation_met": (validation.get("passed") is True) == expected_passed,
        "expected_failed_checks": expected_failed_checks,
        "failed_checks": failed_checks,
        "expected_failed_checks_observed": all(
            check in failed_checks for check in expected_failed_checks
        ),
        "control_report_json": str(control_report_json),
        "validation_report_json": str(validation_report_json),
    }


def audit_phase2dp_cross_directory_replay_negative_controls(
    *,
    phase2do_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2do_report = _read_json(phase2do_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2do_report=phase2do_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2do_replay"
    ]
    checks = {
        "source_phase2do_passed": phase2do_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2do_replay"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 13,
        "all_negative_controls_failed": bool(negative_rows)
        and all(row["observed_passed"] is False for row in negative_rows),
        "all_pass_expectations_met": all(
            row["pass_expectation_met"] for row in control_rows
        ),
        "all_expected_failed_checks_observed": all(
            row["expected_failed_checks_observed"] for row in control_rows
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2dp_cross_directory_replay_negative_controls",
        "passed": passed,
        "ready_for_phase2do_gate_strictness_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "control_count": len(control_rows),
            "negative_control_count": len(negative_rows),
            "negative_controls_failed": sum(
                row["observed_passed"] is False for row in negative_rows
            ),
            "expected_failed_check_assertions": sum(
                len(row["expected_failed_checks"]) for row in control_rows
            ),
        },
        "control_results": control_rows,
        "supported_claims": [
            (
                "the Phase2DO cross-directory replay gate rejects source tampering, "
                "recorded check tampering, incomplete upstream negative controls, "
                "missing or malformed replay reports, replay manifest loss, replay "
                "artifact tampering, missing roles or steps, non-distinct replay "
                "directories, collapsed hash counts, and overstated epoch claims"
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
            "phase2dq_replay_bundle_portability_summary"
            if passed
            else "repair_phase2dp_cross_directory_replay_negative_controls"
        ),
        "evidence": {
            "phase2do_report_json": str(phase2do_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2DP cross-directory replay negative controls."
    )
    parser.add_argument("--phase2do-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dp_cross_directory_replay_negative_controls(
        phase2do_report_json=args.phase2do_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
