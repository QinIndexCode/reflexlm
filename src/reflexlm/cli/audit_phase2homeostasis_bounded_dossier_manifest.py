from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2homeostasis_bounded_publication_dossier import (
    CORE_REPORT_SPECS,
    validate_phase2homeostasis_bounded_publication_dossier,
)
from reflexlm.cli.audit_phase2homeostasis_publication_bundle import (
    _read_json,
    _write_json,
)


REQUIRED_SOURCE_REPORT_ROLES: tuple[str, ...] = tuple(
    spec["role"] for spec in CORE_REPORT_SPECS
) + (
    "exact_cross_runtime_homeostatic_dynamics_limit",
    "hmac_missing_key_negative_control",
    "phase2homeostasis_bounded_publication_dossier",
)

REQUIRED_STATE_ARTIFACT_ROLES: tuple[str, ...] = (
    "state_generation1",
    "state_generation2_py313",
    "state_generation2_py312",
)

REQUIRED_REPRODUCTION_STEPS: tuple[str, ...] = (
    "run_hmac_generation1_py313",
    "run_hmac_generation2_py313",
    "run_hmac_generation2_py312",
    "audit_hmac_chain_py313",
    "audit_hmac_chain_py313_to_py312",
    "audit_hmac_cross_runtime_dynamics_expected_limitation",
    "audit_hmac_phase2cj_invariance",
    "audit_hmac_fresh_rerun_limitations",
    "audit_hmac_no_key_negative_control",
    "audit_hmac_bounded_publication_dossier",
)

OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_exact_cross_runtime_homeostatic_dynamics_claim",
    "ready_for_unbounded_long_term_memory_claim",
    "ready_for_general_runtime_interpreter_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(*, role: str, path: str | Path, content_type: str) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "role": role,
        "path": str(file_path),
        "content_type": content_type,
        "bytes": file_path.stat().st_size,
        "sha256": _sha256(file_path),
    }


def _entry_roles(entries: Any) -> set[str]:
    if not isinstance(entries, list):
        return set()
    return {str(item.get("role")) for item in entries if isinstance(item, dict)}


def _entries_exist(entries: list[dict[str, Any]]) -> bool:
    return bool(entries) and all(
        Path(str(item.get("path", ""))).exists() for item in entries
    )


def _entry_hashes_match(entries: list[dict[str, Any]]) -> bool:
    if not entries:
        return False
    matches = []
    for item in entries:
        path = Path(str(item.get("path", "")))
        matches.append(path.exists() and _sha256(path) == item.get("sha256"))
    return all(matches)


def _source_report_entries(
    *,
    dossier_report_json: str | Path,
    dossier: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = [
        _file_entry(
            role="phase2homeostasis_bounded_publication_dossier",
            path=dossier_report_json,
            content_type="application/json",
        )
    ]
    for row in dossier.get("core_positive_evidence", []):
        if not isinstance(row, dict) or not row.get("role") or not row.get("report_json"):
            continue
        entries.append(
            _file_entry(
                role=str(row["role"]),
                path=str(row["report_json"]),
                content_type="application/json",
            )
        )
    for row in dossier.get("limitation_evidence", []):
        if not isinstance(row, dict) or not row.get("role") or not row.get("report_json"):
            continue
        entries.append(
            _file_entry(
                role=str(row["role"]),
                path=str(row["report_json"]),
                content_type="application/json",
            )
        )
    for row in dossier.get("negative_controls", []):
        if not isinstance(row, dict) or not row.get("role") or not row.get("report_json"):
            continue
        entries.append(
            _file_entry(
                role=str(row["role"]),
                path=str(row["report_json"]),
                content_type="application/json",
            )
        )
    return entries


def _state_entries(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for role, row in zip(
        REQUIRED_STATE_ARTIFACT_ROLES,
        dossier.get("state_artifacts", []),
        strict=False,
    ):
        if not isinstance(row, dict) or not row.get("state_json"):
            continue
        entries.append(
            _file_entry(
                role=role,
                path=str(row["state_json"]),
                content_type="application/json",
            )
        )
    return entries


def _supporting_entries(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    markdown = dossier.get("evidence", {}).get("output_markdown")
    if not markdown:
        return []
    return [
        _file_entry(
            role="bounded_publication_dossier_markdown",
            path=str(markdown),
            content_type="text/markdown",
        )
    ]


def _role_map(rows: list[dict[str, Any]], *, path_key: str) -> dict[str, str]:
    return {
        str(row.get("role")): str(row.get(path_key))
        for row in rows
        if isinstance(row, dict) and row.get("role") and row.get(path_key)
    }


def _reproduction_steps(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    core = _role_map(dossier.get("core_positive_evidence", []), path_key="report_json")
    limits = _role_map(dossier.get("limitation_evidence", []), path_key="report_json")
    negative = _role_map(dossier.get("negative_controls", []), path_key="report_json")
    states = [
        row for row in dossier.get("state_artifacts", []) if isinstance(row, dict)
    ]
    return [
        {
            "step_id": "run_hmac_generation1_py313",
            "module": "reflexlm.cli.run_phase2bq_open_task_family_repo_runtime",
            "auth_key_env_required": True,
            "expected_passed": True,
            "outputs": {
                "output_report_json": core.get("runtime_generation1_py313"),
                "package_homeostatic_state_output": (
                    states[0].get("state_json") if len(states) > 0 else None
                ),
            },
        },
        {
            "step_id": "run_hmac_generation2_py313",
            "module": "reflexlm.cli.run_phase2bq_open_task_family_repo_runtime",
            "auth_key_env_required": True,
            "expected_passed": True,
            "inputs": {
                "package_homeostatic_state_input": (
                    states[0].get("state_json") if len(states) > 0 else None
                )
            },
            "outputs": {
                "output_report_json": core.get("runtime_generation2_py313"),
                "package_homeostatic_state_output": (
                    states[1].get("state_json") if len(states) > 1 else None
                ),
            },
        },
        {
            "step_id": "run_hmac_generation2_py312",
            "module": "reflexlm.cli.run_phase2bq_open_task_family_repo_runtime",
            "auth_key_env_required": True,
            "expected_passed": True,
            "inputs": {
                "package_homeostatic_state_input": (
                    states[0].get("state_json") if len(states) > 0 else None
                )
            },
            "outputs": {
                "output_report_json": core.get("runtime_generation2_py312"),
                "package_homeostatic_state_output": (
                    states[2].get("state_json") if len(states) > 2 else None
                ),
            },
        },
        {
            "step_id": "audit_hmac_chain_py313",
            "module": "reflexlm.cli.audit_phase2homeostasis_persistent_state_chain",
            "auth_key_env_required": True,
            "expected_passed": True,
            "outputs": {
                "output_report_json": core.get("persistent_chain_py313"),
            },
        },
        {
            "step_id": "audit_hmac_chain_py313_to_py312",
            "module": "reflexlm.cli.audit_phase2homeostasis_persistent_state_chain",
            "auth_key_env_required": True,
            "expected_passed": True,
            "outputs": {
                "output_report_json": core.get("persistent_chain_py313_to_py312"),
            },
        },
        {
            "step_id": "audit_hmac_cross_runtime_dynamics_expected_limitation",
            "module": "reflexlm.cli.audit_phase2homeostasis_cross_runtime_dynamics",
            "auth_key_env_required": False,
            "expected_passed": False,
            "outputs": {
                "output_report_json": limits.get(
                    "exact_cross_runtime_homeostatic_dynamics_limit"
                ),
            },
        },
        {
            "step_id": "audit_hmac_phase2cj_invariance",
            "module": "reflexlm.cli.audit_phase2cj_runtime_interpreter_invariance",
            "auth_key_env_required": False,
            "expected_passed": True,
            "outputs": {
                "output_report_json": core.get("runtime_interpreter_invariance"),
            },
        },
        {
            "step_id": "audit_hmac_fresh_rerun_limitations",
            "module": "reflexlm.cli.audit_phase2homeostasis_fresh_rerun_limitations",
            "auth_key_env_required": False,
            "expected_passed": True,
            "outputs": {
                "output_report_json": core.get("fresh_rerun_limitations"),
            },
        },
        {
            "step_id": "audit_hmac_no_key_negative_control",
            "module": "reflexlm.cli.audit_phase2homeostasis_persistent_state_chain",
            "auth_key_env_required": False,
            "expected_passed": False,
            "outputs": {
                "output_report_json": negative.get(
                    "hmac_missing_key_negative_control"
                ),
            },
        },
        {
            "step_id": "audit_hmac_bounded_publication_dossier",
            "module": "reflexlm.cli.audit_phase2homeostasis_bounded_publication_dossier",
            "auth_key_env_required": True,
            "expected_passed": True,
            "outputs": {
                "output_report_json": dossier.get("evidence", {}).get(
                    "phase2homeostasis_bounded_publication_dossier_report_json"
                )
            },
        },
    ]


def validate_phase2homeostasis_bounded_dossier_manifest(
    report: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = report.get("evidence", {}).get("reproducibility_manifest")
    manifest: dict[str, Any] = {}
    manifest_readable = False
    if manifest_path:
        try:
            manifest = _read_json(manifest_path)
            manifest_readable = True
        except (OSError, json.JSONDecodeError):
            manifest_readable = False
    source_reports = manifest.get("source_reports", [])
    state_artifacts = manifest.get("state_artifacts", [])
    supporting_artifacts = manifest.get("supporting_artifacts", [])
    reproduction_steps = manifest.get("reproduction_steps", [])
    if not isinstance(source_reports, list):
        source_reports = []
    if not isinstance(state_artifacts, list):
        state_artifacts = []
    if not isinstance(supporting_artifacts, list):
        supporting_artifacts = []
    if not isinstance(reproduction_steps, list):
        reproduction_steps = []
    checks = {
        "artifact_family_matches": (
            report.get("artifact_family")
            == "phase2homeostasis_bounded_dossier_manifest"
        ),
        "top_level_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_homeostasis_dossier_manifest_claim")
            is True
            and all(report.get(flag) is not True for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_dossier_passed": (
            report.get("source_summary", {}).get("dossier_passed") is True
        ),
        "manifest_readable": manifest_readable,
        "source_report_roles_complete": set(REQUIRED_SOURCE_REPORT_ROLES).issubset(
            _entry_roles(source_reports)
        ),
        "source_reports_exist": _entries_exist(source_reports),
        "source_report_hashes_match": _entry_hashes_match(source_reports),
        "state_artifact_roles_complete": set(REQUIRED_STATE_ARTIFACT_ROLES).issubset(
            _entry_roles(state_artifacts)
        ),
        "state_artifacts_exist": _entries_exist(state_artifacts),
        "state_artifact_hashes_match": _entry_hashes_match(state_artifacts),
        "supporting_artifacts_exist": _entries_exist(supporting_artifacts),
        "supporting_artifact_hashes_match": _entry_hashes_match(
            supporting_artifacts
        ),
        "reproduction_steps_complete": set(REQUIRED_REPRODUCTION_STEPS).issubset(
            {
                str(item.get("step_id"))
                for item in reproduction_steps
                if isinstance(item, dict)
            }
        ),
        "reproduction_steps_have_modules_outputs_and_expectations": bool(
            reproduction_steps
        )
        and all(
            item.get("module")
            and isinstance(item.get("outputs"), dict)
            and item.get("expected_passed") in {True, False}
            for item in reproduction_steps
            if isinstance(item, dict)
        ),
        "manifest_contains_bounded_boundary_and_limitation": (
            "bounded" in manifest.get("claim_boundary", "")
            and "does not support exact cross-runtime internal homeostatic microdynamics"
            in manifest.get("claim_boundary", "")
            and "epoch-making architecture" in manifest.get("claim_boundary", "")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "source_report_count": len(source_reports),
            "state_artifact_count": len(state_artifacts),
            "supporting_artifact_count": len(supporting_artifacts),
            "reproduction_step_count": len(reproduction_steps),
        },
    }


def audit_phase2homeostasis_bounded_dossier_manifest(
    *,
    dossier_report_json: str | Path,
    output_manifest_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    dossier = _read_json(dossier_report_json)
    dossier.setdefault("evidence", {})[
        "phase2homeostasis_bounded_publication_dossier_report_json"
    ] = str(dossier_report_json)
    dossier_validation = validate_phase2homeostasis_bounded_publication_dossier(
        dossier
    )
    if dossier.get("passed") is not True or dossier_validation.get("passed") is not True:
        raise ValueError("bounded dossier manifest requires a passed bounded dossier")
    manifest = {
        "manifest_family": "phase2homeostasis_bounded_dossier_manifest",
        "source_reports": _source_report_entries(
            dossier_report_json=dossier_report_json,
            dossier=dossier,
        ),
        "state_artifacts": _state_entries(dossier),
        "supporting_artifacts": _supporting_entries(dossier),
        "reproduction_steps": _reproduction_steps(dossier),
        "claim_boundary": (
            "This manifest supports bounded HMAC-authenticated homeostatic "
            "persistent-state mechanism replay with an explicit limitation: it "
            "does not support exact cross-runtime internal homeostatic "
            "microdynamics, unbounded memory, free-form shell autonomy, open-ended "
            "native perception, production autonomy, or epoch-making architecture."
        ),
    }
    _write_json(output_manifest_json, manifest)
    checks = {
        "source_dossier_passed_and_validates": dossier.get("passed") is True
        and dossier_validation.get("passed") is True,
        "source_report_roles_complete": set(REQUIRED_SOURCE_REPORT_ROLES).issubset(
            _entry_roles(manifest["source_reports"])
        ),
        "source_report_hashes_match": _entry_hashes_match(
            manifest["source_reports"]
        ),
        "state_artifact_roles_complete": set(REQUIRED_STATE_ARTIFACT_ROLES).issubset(
            _entry_roles(manifest["state_artifacts"])
        ),
        "state_artifact_hashes_match": _entry_hashes_match(
            manifest["state_artifacts"]
        ),
        "supporting_artifact_hashes_match": _entry_hashes_match(
            manifest["supporting_artifacts"]
        ),
        "reproduction_steps_complete": set(REQUIRED_REPRODUCTION_STEPS).issubset(
            {
                str(item.get("step_id"))
                for item in manifest["reproduction_steps"]
                if isinstance(item, dict)
            }
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_bounded_dossier_manifest",
        "passed": passed,
        "ready_for_bounded_homeostasis_dossier_manifest_claim": passed,
        "ready_for_exact_cross_runtime_homeostatic_dynamics_claim": False,
        "ready_for_unbounded_long_term_memory_claim": False,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "source_report_count": len(manifest["source_reports"]),
            "state_artifact_count": len(manifest["state_artifacts"]),
            "supporting_artifact_count": len(manifest["supporting_artifacts"]),
            "reproduction_step_count": len(manifest["reproduction_steps"]),
        },
        "source_summary": {
            "dossier_passed": dossier.get("passed") is True,
            "dossier_validation_passed": dossier_validation.get("passed") is True,
            "dossier_report_json": str(dossier_report_json),
            "core_positive_report_count": dossier.get("metrics", {}).get(
                "core_positive_report_count"
            ),
            "limitation_evidence_count": dossier.get("metrics", {}).get(
                "limitation_evidence_count"
            ),
            "negative_control_count": dossier.get("metrics", {}).get(
                "negative_control_count"
            ),
        },
        "claim_boundary": manifest["claim_boundary"],
        "supported_claims": [
            (
                "bounded reproducibility manifest for the HMAC-authenticated "
                "homeostatic persistent-state dossier with explicit limitation"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "exact cross-runtime homeostatic microdynamics",
            "unbounded or semantic long-term memory",
            "free-form shell autonomy",
            "general runtime interpreter invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "cross_directory_replay_of_bounded_dossier_manifest"
            if passed
            else "repair_phase2homeostasis_bounded_dossier_manifest"
        ),
        "evidence": {
            "dossier_report_json": str(dossier_report_json),
            "reproducibility_manifest": str(output_manifest_json),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reproducibility manifest for the bounded dossier."
    )
    parser.add_argument("--dossier-report-json", required=True)
    parser.add_argument("--output-manifest-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_bounded_dossier_manifest(
        dossier_report_json=args.dossier_report_json,
        output_manifest_json=args.output_manifest_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
