from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2homeostasis_publication_reproducibility_manifest import (
    REQUIRED_SOURCE_REPORT_ROLES,
    REQUIRED_STATE_ARTIFACT_ROLES,
    _read_json,
    _write_json,
    validate_phase2homeostasis_publication_reproducibility_manifest,
)
from reflexlm.cli.audit_phase2homeostasis_publication_bundle import (
    validate_phase2homeostasis_publication_bundle,
)
from reflexlm.cli.audit_phase2homeostasis_reproducibility_manifest_replay import (
    validate_phase2homeostasis_reproducibility_manifest_replay,
)


OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_unbounded_long_term_memory_claim",
    "ready_for_general_runtime_interpreter_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _role_path_map(entries: Any, *, path_key: str = "path") -> dict[str, str]:
    if not isinstance(entries, list):
        return {}
    rows: dict[str, str] = {}
    for item in entries:
        if isinstance(item, dict) and item.get("role") and item.get(path_key):
            rows[str(item["role"])] = str(item[path_key])
    return rows


def _bundle_report_paths(bundle: dict[str, Any]) -> dict[str, str]:
    paths = _role_path_map(bundle.get("positive_evidence", []), path_key="report_json")
    for row in bundle.get("negative_controls", []):
        if isinstance(row, dict) and row.get("role") and row.get("report_json"):
            paths[str(row["role"])] = str(row["report_json"])
    paths["phase2homeostasis_publication_bundle"] = str(
        bundle.get("evidence", {}).get("phase2homeostasis_publication_bundle_report_json")
        or bundle.get("evidence", {}).get("bundle_report_json")
        or ""
    )
    return {key: value for key, value in paths.items() if value}


def _bundle_state_paths(bundle: dict[str, Any]) -> dict[str, str]:
    rows = {}
    for role, item in zip(
        REQUIRED_STATE_ARTIFACT_ROLES,
        bundle.get("state_artifacts", []),
        strict=False,
    ):
        if isinstance(item, dict) and item.get("state_json"):
            rows[role] = str(item["state_json"])
    return rows


def _paths_distinct(source_paths: dict[str, str], fresh_paths: dict[str, str]) -> bool:
    shared_roles = set(source_paths).intersection(fresh_paths)
    return bool(shared_roles) and all(
        Path(source_paths[role]) != Path(fresh_paths[role]) for role in shared_roles
    )


def validate_phase2homeostasis_fresh_rerun_from_manifest(
    report: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "artifact_family_matches": (
            report.get("artifact_family")
            == "phase2homeostasis_fresh_rerun_from_manifest"
        ),
        "top_level_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_homeostasis_fresh_rerun_claim") is True
            and all(report.get(flag) is not True for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_and_fresh_reports_passed": (
            report.get("source_summary", {}).get("source_manifest_report_passed")
            is True
            and report.get("fresh_summary", {}).get("bundle_passed") is True
            and report.get("fresh_summary", {}).get("reproducibility_manifest_passed")
            is True
            and report.get("fresh_summary", {}).get("manifest_replay_passed") is True
        ),
        "fresh_source_report_roles_complete": set(REQUIRED_SOURCE_REPORT_ROLES).issubset(
            set(report.get("fresh_summary", {}).get("source_report_roles", []))
        ),
        "fresh_state_artifact_roles_complete": set(
            REQUIRED_STATE_ARTIFACT_ROLES
        ).issubset(set(report.get("fresh_summary", {}).get("state_artifact_roles", []))),
        "fresh_source_paths_are_distinct_from_source_manifest": (
            report.get("fresh_summary", {}).get(
                "source_report_paths_distinct_from_source_manifest"
            )
            is True
        ),
        "fresh_state_paths_are_distinct_from_source_manifest": (
            report.get("fresh_summary", {}).get(
                "state_artifact_paths_distinct_from_source_manifest"
            )
            is True
        ),
        "fresh_replay_directory_distinct": (
            report.get("fresh_summary", {}).get("manifest_replay_directory_distinct")
            is True
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "fresh_source_report_count": report.get("fresh_summary", {}).get(
                "source_report_count", 0
            ),
            "fresh_state_artifact_count": report.get("fresh_summary", {}).get(
                "state_artifact_count", 0
            ),
            "fresh_reproduction_step_count": report.get("fresh_summary", {}).get(
                "reproduction_step_count", 0
            ),
        },
    }


def audit_phase2homeostasis_fresh_rerun_from_manifest(
    *,
    source_reproducibility_report_json: str | Path,
    fresh_bundle_report_json: str | Path,
    fresh_reproducibility_report_json: str | Path,
    fresh_manifest_replay_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    source_repro = _read_json(source_reproducibility_report_json)
    source_manifest_path = Path(
        str(source_repro.get("evidence", {}).get("reproducibility_manifest", ""))
    )
    if not source_manifest_path.exists():
        raise ValueError("fresh rerun audit requires readable source manifest")
    source_manifest = _read_json(source_manifest_path)
    fresh_bundle = _read_json(fresh_bundle_report_json)
    fresh_repro = _read_json(fresh_reproducibility_report_json)
    fresh_replay = _read_json(fresh_manifest_replay_report_json)
    source_report_paths = _role_path_map(source_manifest.get("source_reports", []))
    source_state_paths = _role_path_map(source_manifest.get("state_artifacts", []))
    fresh_report_paths = _bundle_report_paths(fresh_bundle)
    fresh_report_paths["phase2homeostasis_publication_bundle"] = str(
        fresh_bundle_report_json
    )
    fresh_state_paths = _bundle_state_paths(fresh_bundle)
    fresh_bundle_validation = validate_phase2homeostasis_publication_bundle(
        fresh_bundle
    )
    fresh_repro_validation = (
        validate_phase2homeostasis_publication_reproducibility_manifest(fresh_repro)
    )
    fresh_replay_validation = validate_phase2homeostasis_reproducibility_manifest_replay(
        fresh_replay
    )
    checks = {
        "source_reproducibility_manifest_passed": source_repro.get("passed") is True,
        "fresh_bundle_passed_and_validates": fresh_bundle.get("passed") is True
        and fresh_bundle_validation.get("passed") is True,
        "fresh_reproducibility_manifest_passed_and_validates": (
            fresh_repro.get("passed") is True
            and fresh_repro_validation.get("passed") is True
        ),
        "fresh_manifest_replay_passed_and_validates": (
            fresh_replay.get("passed") is True
            and fresh_replay_validation.get("passed") is True
        ),
        "fresh_source_report_roles_complete": set(
            REQUIRED_SOURCE_REPORT_ROLES
        ).issubset(set(fresh_report_paths)),
        "fresh_state_artifact_roles_complete": set(
            REQUIRED_STATE_ARTIFACT_ROLES
        ).issubset(set(fresh_state_paths)),
        "fresh_source_report_paths_distinct": _paths_distinct(
            source_report_paths,
            fresh_report_paths,
        ),
        "fresh_state_artifact_paths_distinct": _paths_distinct(
            source_state_paths,
            fresh_state_paths,
        ),
        "fresh_replay_directory_distinct": (
            fresh_replay.get("checks", {}).get("replay_directory_is_distinct") is True
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_fresh_rerun_from_manifest",
        "passed": passed,
        "ready_for_bounded_homeostasis_fresh_rerun_claim": passed,
        "ready_for_unbounded_long_term_memory_claim": False,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "source_report_roles_compared": len(set(source_report_paths).intersection(fresh_report_paths)),
            "state_artifact_roles_compared": len(set(source_state_paths).intersection(fresh_state_paths)),
            "fresh_source_report_count": len(fresh_report_paths),
            "fresh_state_artifact_count": len(fresh_state_paths),
        },
        "source_summary": {
            "source_manifest_report_passed": source_repro.get("passed") is True,
            "source_reproducibility_report_json": str(source_reproducibility_report_json),
            "source_manifest_json": str(source_manifest_path),
            "source_report_roles": sorted(source_report_paths),
            "state_artifact_roles": sorted(source_state_paths),
        },
        "fresh_summary": {
            "bundle_passed": fresh_bundle.get("passed") is True,
            "reproducibility_manifest_passed": fresh_repro.get("passed") is True,
            "manifest_replay_passed": fresh_replay.get("passed") is True,
            "source_report_count": len(fresh_report_paths),
            "state_artifact_count": len(fresh_state_paths),
            "reproduction_step_count": fresh_repro.get("metrics", {}).get(
                "reproduction_step_count"
            ),
            "source_report_roles": sorted(fresh_report_paths),
            "state_artifact_roles": sorted(fresh_state_paths),
            "source_report_paths_distinct_from_source_manifest": _paths_distinct(
                source_report_paths,
                fresh_report_paths,
            ),
            "state_artifact_paths_distinct_from_source_manifest": _paths_distinct(
                source_state_paths,
                fresh_state_paths,
            ),
            "manifest_replay_directory_distinct": (
                fresh_replay.get("checks", {}).get("replay_directory_is_distinct")
                is True
            ),
        },
        "claim_boundary": (
            "This audit verifies a fresh rerun output set derived from the "
            "Phase2Homeostasis reproducibility manifest. It supports only bounded "
            "HMAC-authenticated homeostatic persistent-state mechanism evidence, "
            "not unbounded memory, open-ended native perception, production "
            "autonomy, or epoch-making architecture claims."
        ),
        "supported_claims": [
            "bounded fresh rerun of the homeostatic HMAC publication evidence bundle"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "unbounded or semantic long-term memory",
            "free-form shell autonomy",
            "general runtime interpreter invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "external_machine_replay_or_submission_validator"
            if passed
            else "repair_homeostatic_fresh_rerun_from_manifest"
        ),
        "evidence": {
            "source_reproducibility_report_json": str(source_reproducibility_report_json),
            "fresh_bundle_report_json": str(fresh_bundle_report_json),
            "fresh_reproducibility_report_json": str(fresh_reproducibility_report_json),
            "fresh_manifest_replay_report_json": str(fresh_manifest_replay_report_json),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a fresh Phase2Homeostasis rerun against the source manifest."
    )
    parser.add_argument("--source-reproducibility-report-json", required=True)
    parser.add_argument("--fresh-bundle-report-json", required=True)
    parser.add_argument("--fresh-reproducibility-report-json", required=True)
    parser.add_argument("--fresh-manifest-replay-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_fresh_rerun_from_manifest(
        source_reproducibility_report_json=args.source_reproducibility_report_json,
        fresh_bundle_report_json=args.fresh_bundle_report_json,
        fresh_reproducibility_report_json=args.fresh_reproducibility_report_json,
        fresh_manifest_replay_report_json=args.fresh_manifest_replay_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
