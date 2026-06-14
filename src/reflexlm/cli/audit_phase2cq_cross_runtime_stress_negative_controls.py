from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

from reflexlm.cli.audit_phase2cp_cross_runtime_environment_stress_recovery_matrix import (
    audit_phase2cp_cross_runtime_environment_stress_recovery_matrix,
)


Mutation = Callable[
    [list[dict[str, Any]], list[list[dict[str, Any]]], list[list[dict[str, Any]]]],
    None,
]


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _repo_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("repository_reports", [])
    return rows if isinstance(rows, list) else []


def _episodes_by_stress(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(episode.get("generator", {}).get("stress_id", "")): episode
        for episode in manifest.get("episodes", [])
    }


def _run_steps(episode: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for collection_name in ("permissions", "completion_requirements"):
        for step in episode.get(collection_name, []):
            if step.get("action_type") == "RUN_COMMAND":
                steps.append(step)
    return steps


def _materialize_case_inputs(
    *,
    source_runtime_report_jsons: list[str | Path],
    case_dir: str | Path,
) -> tuple[list[Path], list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    case_root = Path(case_dir)
    copied_runtime_report_jsons: list[Path] = []
    manifest_groups: list[list[dict[str, Any]]] = []
    subreport_groups: list[list[dict[str, Any]]] = []
    for runtime_index, source_report_path in enumerate(source_runtime_report_jsons):
        report = deepcopy(_read_json(source_report_path))
        copied_manifests: list[dict[str, Any]] = []
        copied_subreports: list[dict[str, Any]] = []
        for repo_index, repo in enumerate(_repo_reports(report)):
            manifest = deepcopy(_read_json(repo["generated_manifest_json"]))
            manifest_path = (
                case_root
                / "manifests"
                / f"runtime_{runtime_index}"
                / f"{repo_index}_{repo['repository_id']}.json"
            )
            repo["generated_manifest_json"] = str(manifest_path)
            copied_manifests.append(manifest)

            subreport = deepcopy(_read_json(repo["report_json"]))
            subreport_path = (
                case_root
                / "subreports"
                / f"runtime_{runtime_index}"
                / f"{repo_index}_{repo['repository_id']}.json"
            )
            repo["report_json"] = str(subreport_path)
            copied_subreports.append(subreport)
        copied_report_path = case_root / f"runtime_{runtime_index}.json"
        copied_runtime_report_jsons.append(copied_report_path)
        manifest_groups.append(copied_manifests)
        subreport_groups.append(copied_subreports)
        _write_runtime_manifests_and_subreports(
            report_path=copied_report_path,
            report=report,
            manifests=copied_manifests,
            subreports=copied_subreports,
        )
    return copied_runtime_report_jsons, manifest_groups, subreport_groups


def _write_runtime_manifests_and_subreports(
    *,
    report_path: Path,
    report: dict[str, Any],
    manifests: list[dict[str, Any]],
    subreports: list[dict[str, Any]],
) -> None:
    for repo, manifest, subreport in zip(
        _repo_reports(report),
        manifests,
        subreports,
        strict=True,
    ):
        _write_json(repo["generated_manifest_json"], manifest)
        _write_json(repo["report_json"], subreport)
    _write_json(report_path, report)


def _rewrite_case_inputs(
    *,
    runtime_report_jsons: list[Path],
    reports: list[dict[str, Any]],
    manifest_groups: list[list[dict[str, Any]]],
    subreport_groups: list[list[dict[str, Any]]],
) -> None:
    for path, report, manifests, subreports in zip(
        runtime_report_jsons,
        reports,
        manifest_groups,
        subreport_groups,
        strict=True,
    ):
        _write_runtime_manifests_and_subreports(
            report_path=path,
            report=report,
            manifests=manifests,
            subreports=subreports,
        )


def _first_episode(
    manifest_groups: list[list[dict[str, Any]]],
    stress_id: str,
) -> dict[str, Any]:
    for manifests in manifest_groups:
        for manifest in manifests:
            episode = _episodes_by_stress(manifest).get(stress_id)
            if episode is not None:
                return episode
    raise ValueError(f"control fixture is missing stress_id: {stress_id}")


def _mutate_missing_observed_recovery(
    reports: list[dict[str, Any]],
    manifest_groups: list[list[dict[str, Any]]],
    subreport_groups: list[list[dict[str, Any]]],
) -> None:
    del reports, manifest_groups
    subreport_groups[1][0]["episode_reports"][0][
        "observed_recovery_after_failure"
    ] = False
    subreport_groups[1][0]["episode_reports"][0]["recovery_success"] = False
    subreport_groups[1][0]["episode_reports"][0]["task_completion_success"] = False


def _mutate_failure_metrics_erased(
    reports: list[dict[str, Any]],
    manifest_groups: list[list[dict[str, Any]]],
    subreport_groups: list[list[dict[str, Any]]],
) -> None:
    del manifest_groups, subreport_groups
    reports[1]["metrics"]["observed_recoveries_after_failure"] = 8
    reports[1]["metrics"]["failure_recovery_success_rate"] = 8 / 9
    repo = _repo_reports(reports[1])[0]
    repo["failure_recovery_metrics"]["observed_recoveries_after_failure"] = 2
    repo["metrics"]["failure_recovery_success_rate"] = 2 / 3


def _mutate_runtime_version_diversity_collapsed(
    reports: list[dict[str, Any]],
    manifest_groups: list[list[dict[str, Any]]],
    subreport_groups: list[list[dict[str, Any]]],
) -> None:
    del manifest_groups, subreport_groups
    for report in reports:
        report.setdefault("runtime_environment", {})["version"] = "3.13.2"


def _mutate_manifest_matrix_drift(
    reports: list[dict[str, Any]],
    manifest_groups: list[list[dict[str, Any]]],
    subreport_groups: list[list[dict[str, Any]]],
) -> None:
    del reports, subreport_groups
    episode = _first_episode(manifest_groups[1:], "missing_env_then_overlay_recover")
    for step in _run_steps(episode):
        step["timeout_seconds"] = 4.0


def _mutate_claim_overstated(
    reports: list[dict[str, Any]],
    manifest_groups: list[list[dict[str, Any]]],
    subreport_groups: list[list[dict[str, Any]]],
) -> None:
    del manifest_groups, subreport_groups
    reports[0]["ready_for_general_runtime_invariance_claim"] = True


def _mutate_runtime_executable_mismatch(
    reports: list[dict[str, Any]],
    manifest_groups: list[list[dict[str, Any]]],
    subreport_groups: list[list[dict[str, Any]]],
) -> None:
    del manifest_groups, subreport_groups
    reports[0].setdefault("runtime_environment", {})["executable"] = (
        "D:\\wrong-runtime\\python.exe"
    )


def _mutate_mapping_scope_missing(
    reports: list[dict[str, Any]],
    manifest_groups: list[list[dict[str, Any]]],
    subreport_groups: list[list[dict[str, Any]]],
) -> None:
    del manifest_groups, subreport_groups
    for repo in _repo_reports(reports[0]):
        repo.get("policy_configuration", {}).get("policy_metadata", {})[
            "python_identity_mapping_scope"
        ] = "full_command_text"


def _mutate_non_cpython_runtime(
    reports: list[dict[str, Any]],
    manifest_groups: list[list[dict[str, Any]]],
    subreport_groups: list[list[dict[str, Any]]],
) -> None:
    del manifest_groups, subreport_groups
    reports[2].setdefault("runtime_environment", {})["implementation"] = "PyPy"


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2cp_inputs",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_missing_observed_recovery",
        "mutation": _mutate_missing_observed_recovery,
        "expected_passed": False,
        "expected_failed_checks": ["all_report_manifests_pass_phase2co_shape"],
    },
    {
        "control_id": "negative_failure_metrics_erased",
        "mutation": _mutate_failure_metrics_erased,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_runtime_reports_share_core_failure_recovery_metrics",
            "all_report_manifests_pass_phase2co_shape",
        ],
    },
    {
        "control_id": "negative_runtime_version_diversity_collapsed",
        "mutation": _mutate_runtime_version_diversity_collapsed,
        "expected_passed": False,
        "expected_failed_checks": ["minimum_two_distinct_python_versions_met"],
    },
    {
        "control_id": "negative_manifest_matrix_drift",
        "mutation": _mutate_manifest_matrix_drift,
        "expected_passed": False,
        "expected_failed_checks": ["all_runtime_reports_share_phase2co_manifest_matrix"],
    },
    {
        "control_id": "negative_general_runtime_claim_overstated",
        "mutation": _mutate_claim_overstated,
        "expected_passed": False,
        "expected_failed_checks": ["bounded_claim_true_only_for_all_reports"],
    },
    {
        "control_id": "negative_runtime_executable_mismatch",
        "mutation": _mutate_runtime_executable_mismatch,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_runtime_environment_executables_match_reports"
        ],
    },
    {
        "control_id": "negative_command_mapping_scope_missing",
        "mutation": _mutate_mapping_scope_missing,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_repositories_used_command_prefix_identity_mapping"
        ],
    },
    {
        "control_id": "negative_non_cpython_runtime",
        "mutation": _mutate_non_cpython_runtime,
        "expected_passed": False,
        "expected_failed_checks": ["all_runtime_environments_are_cpython"],
    },
)


def _run_control(
    *,
    control_spec: dict[str, Any],
    source_runtime_report_jsons: list[str | Path],
    package_build_report_json: str | Path,
    control_root: str | Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = Path(control_root) / control_id
    runtime_report_jsons, manifest_groups, subreport_groups = _materialize_case_inputs(
        source_runtime_report_jsons=source_runtime_report_jsons,
        case_dir=case_dir,
    )
    reports = [_read_json(path) for path in runtime_report_jsons]
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(reports, manifest_groups, subreport_groups)
        _rewrite_case_inputs(
            runtime_report_jsons=runtime_report_jsons,
            reports=reports,
            manifest_groups=manifest_groups,
            subreport_groups=subreport_groups,
        )
    audit_report_json = case_dir / "phase2cp_audit.json"
    audit = audit_phase2cp_cross_runtime_environment_stress_recovery_matrix(
        runtime_report_jsons=runtime_report_jsons,
        package_build_report_json=package_build_report_json,
        output_report_json=audit_report_json,
    )
    expected_passed = bool(control_spec["expected_passed"])
    expected_failed_checks = list(control_spec["expected_failed_checks"])
    failed_checks = [
        check_name
        for check_name, passed in audit.get("checks", {}).items()
        if passed is False
    ]
    return {
        "control_id": control_id,
        "expected_passed": expected_passed,
        "observed_passed": audit.get("passed") is True,
        "pass_expectation_met": (audit.get("passed") is True) == expected_passed,
        "expected_failed_checks": expected_failed_checks,
        "failed_checks": failed_checks,
        "expected_failed_checks_observed": all(
            check_name in failed_checks for check_name in expected_failed_checks
        ),
        "audit_report_json": str(audit_report_json),
        "runtime_report_jsons": [str(path) for path in runtime_report_jsons],
    }


def audit_phase2cq_cross_runtime_stress_negative_controls(
    *,
    phase2cp_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2cp = _read_json(phase2cp_report_json)
    evidence = phase2cp.get("evidence", {})
    source_runtime_report_jsons = evidence.get("runtime_report_jsons", [])
    package_build_report_json = evidence.get("package_build_report_json")
    if not isinstance(source_runtime_report_jsons, list) or not source_runtime_report_jsons:
        raise ValueError("Phase2CQ requires Phase2CP evidence.runtime_report_jsons")
    if not package_build_report_json:
        raise ValueError("Phase2CQ requires Phase2CP evidence.package_build_report_json")

    control_rows = [
        _run_control(
            control_spec=control_spec,
            source_runtime_report_jsons=source_runtime_report_jsons,
            package_build_report_json=package_build_report_json,
            control_root=output_dir,
        )
        for control_spec in CONTROL_SPECS
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2cp_inputs"
    ]
    checks = {
        "source_phase2cp_passed": phase2cp.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2cp_inputs"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 8,
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
        "artifact_family": "phase2cq_cross_runtime_stress_negative_controls",
        "passed": passed,
        "ready_for_phase2cp_gate_strictness_claim": passed,
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
                "the Phase2CP gate rejects local counterfactual overclaims for missing "
                "observed recovery, erased failure metrics, collapsed Python-version "
                "diversity, manifest-matrix drift, overstated runtime-invariance flags, "
                "runtime executable mismatch, missing command-prefix identity mapping, "
                "and non-CPython runtime labeling"
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
            "phase2cr_stress_recovery_repetition_stability"
            if passed
            else "repair_phase2cq_cross_runtime_stress_negative_controls"
        ),
        "evidence": {
            "phase2cp_report_json": str(phase2cp_report_json),
            "source_runtime_report_jsons": [
                str(path) for path in source_runtime_report_jsons
            ],
            "package_build_report_json": str(package_build_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2CQ cross-runtime stress negative controls."
    )
    parser.add_argument("--phase2cp-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cq_cross_runtime_stress_negative_controls(
        phase2cp_report_json=args.phase2cp_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
