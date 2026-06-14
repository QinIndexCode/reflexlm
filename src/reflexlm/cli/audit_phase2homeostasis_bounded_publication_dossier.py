from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2homeostasis_publication_bundle import (
    _forbidden_secret_scan,
    _negative_row,
    _read_json,
    _state_row,
    _write_json,
)


OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_exact_cross_runtime_homeostatic_dynamics_claim",
    "ready_for_unbounded_long_term_memory_claim",
    "ready_for_general_runtime_interpreter_invariance_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)

CORE_REPORT_SPECS: tuple[dict[str, str], ...] = (
    {
        "role": "runtime_generation1_py313",
        "family": "phase2ci_unified_package_open_task_family_repo_runtime",
    },
    {
        "role": "runtime_generation2_py313",
        "family": "phase2ci_unified_package_open_task_family_repo_runtime",
    },
    {
        "role": "runtime_generation2_py312",
        "family": "phase2ci_unified_package_open_task_family_repo_runtime",
    },
    {
        "role": "persistent_chain_py313",
        "family": "phase2homeostasis_persistent_state_chain",
    },
    {
        "role": "persistent_chain_py313_to_py312",
        "family": "phase2homeostasis_persistent_state_chain",
    },
    {
        "role": "runtime_interpreter_invariance",
        "family": "phase2cj_runtime_interpreter_invariance_audit",
    },
    {
        "role": "fresh_rerun_limitations",
        "family": "phase2homeostasis_fresh_rerun_limitations",
    },
)


def _bounded_ready_flags(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key.startswith("ready_for_") and key not in OVERCLAIM_READY_FLAGS
    }


def _bounded_claim_ok(report: dict[str, Any]) -> bool:
    bounded_flags = _bounded_ready_flags(report)
    return (
        bool(bounded_flags)
        and any(value is True for value in bounded_flags.values())
        and all(report.get(flag) is not True for flag in OVERCLAIM_READY_FLAGS)
    )


def _compact_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {
        key: metrics[key]
        for key in (
            "repositories",
            "episodes",
            "executed_actions",
            "task_completion_success_rate",
            "changed_episode_count",
            "maximum_active_threshold_delta",
            "threshold_drift_limit",
            "side_effect_trace_rows",
        )
        if key in metrics
    }


def _core_report_row(path: str | Path, spec: dict[str, str]) -> dict[str, Any]:
    try:
        report = _read_json(path)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        return {
            "role": spec["role"],
            "report_json": str(path),
            "readable": False,
            "read_error": type(exc).__name__,
            "passed": False,
            "artifact_family": None,
            "expected_family": spec["family"],
            "family_matches_expected": False,
            "bounded_claim_ok": False,
            "compact_metrics": {},
        }
    return {
        "role": spec["role"],
        "report_json": str(path),
        "readable": True,
        "passed": report.get("passed") is True,
        "artifact_family": report.get("artifact_family"),
        "expected_family": spec["family"],
        "family_matches_expected": report.get("artifact_family") == spec["family"],
        "bounded_claim_ok": _bounded_claim_ok(report),
        "compact_metrics": _compact_metrics(report),
        "next_required_experiment": report.get("next_required_experiment"),
    }


def _cross_runtime_limitation_row(
    path: str | Path,
    *,
    threshold_drift_limit: float,
) -> dict[str, Any]:
    try:
        report = _read_json(path)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        return {
            "role": "exact_cross_runtime_homeostatic_dynamics_limit",
            "report_json": str(path),
            "readable": False,
            "read_error": type(exc).__name__,
            "expected_failure_observed": False,
            "bounded_to_internal_dynamics": False,
            "maximum_active_threshold_delta": None,
        }
    checks = report.get("checks", {})
    metrics = report.get("metrics", {})
    max_delta = metrics.get("maximum_active_threshold_delta")
    max_delta_bounded = (
        isinstance(max_delta, (int, float))
        and not isinstance(max_delta, bool)
        and math.isfinite(float(max_delta))
        and float(max_delta) <= threshold_drift_limit
    )
    exact_failure_observed = (
        report.get("artifact_family") == "phase2homeostasis_cross_runtime_dynamics"
        and report.get("passed") is False
        and report.get("ready_for_bounded_cross_runtime_homeostatic_dynamics_claim")
        is False
        and any(
            checks.get(name) is False
            for name in (
                "discrete_homeostatic_dynamics_match",
                "active_threshold_deltas_within_tolerance",
                "wake_reason_counts_match",
                "runtime_normalized_executable_action_traces_match",
            )
        )
    )
    bounded_to_internal_dynamics = (
        checks.get("canonical_runtime_passed") is True
        and checks.get("alternate_runtime_passed") is True
        and checks.get("same_seed_and_task_matrix") is True
        and checks.get("same_core_completion_metrics") is True
        and max_delta_bounded
    )
    return {
        "role": "exact_cross_runtime_homeostatic_dynamics_limit",
        "report_json": str(path),
        "readable": True,
        "passed": report.get("passed") is True,
        "artifact_family": report.get("artifact_family"),
        "expected_failure_observed": exact_failure_observed,
        "bounded_to_internal_dynamics": bounded_to_internal_dynamics,
        "maximum_active_threshold_delta": max_delta,
        "threshold_drift_limit": threshold_drift_limit,
        "failed_exact_checks": [
            name for name, value in checks.items() if value is False
        ],
    }


def _markdown_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ("role", "passed", "bounded_claim_ok", "artifact_family", "key_metrics")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        metrics = row.get("compact_metrics", {})
        metric_text = "; ".join(f"{key}={value}" for key, value in metrics.items())
        lines.append(
            "| "
            + " | ".join(
                _markdown_escape(
                    metric_text if column == "key_metrics" else row.get(column)
                )
                for column in columns
            )
            + " |"
        )
    return "\n".join(lines)


def validate_phase2homeostasis_bounded_publication_dossier(
    report: dict[str, Any],
) -> dict[str, Any]:
    core_rows = report.get("core_positive_evidence", [])
    state_rows = report.get("state_artifacts", [])
    negative_rows = report.get("negative_controls", [])
    limitation_rows = report.get("limitation_evidence", [])
    if not isinstance(core_rows, list):
        core_rows = []
    if not isinstance(state_rows, list):
        state_rows = []
    if not isinstance(negative_rows, list):
        negative_rows = []
    if not isinstance(limitation_rows, list):
        limitation_rows = []
    forbidden_scan = report.get("forbidden_secret_scan", {})
    checks = {
        "artifact_family_matches": (
            report.get("artifact_family")
            == "phase2homeostasis_bounded_publication_dossier"
        ),
        "top_level_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_homeostasis_publication_dossier_claim")
            is True
            and all(report.get(flag) is not True for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "core_positive_evidence_count_met": len(core_rows) == len(CORE_REPORT_SPECS),
        "core_positive_evidence_passed_and_bounded": bool(core_rows)
        and all(
            row.get("readable") is True
            and row.get("passed") is True
            and row.get("family_matches_expected") is True
            and row.get("bounded_claim_ok") is True
            for row in core_rows
        ),
        "state_artifact_count_met": len(state_rows) >= 3,
        "state_artifacts_hmac_bounded_and_valid": bool(state_rows)
        and all(
            row.get("readable") is True
            and row.get("schema_valid") is True
            and row.get("bounded_state_keys_only") is True
            and row.get("authenticator_algorithm") == "hmac-sha256"
            and isinstance(row.get("key_fingerprint_sha256"), str)
            and row.get("integrity_valid") is True
            for row in state_rows
        ),
        "missing_key_negative_control_failed_closed": bool(negative_rows)
        and all(row.get("expected_failure_observed") is True for row in negative_rows),
        "limitation_evidence_records_exact_dynamics_failure": bool(limitation_rows)
        and all(
            row.get("expected_failure_observed") is True
            and row.get("bounded_to_internal_dynamics") is True
            for row in limitation_rows
        ),
        "claim_boundary_records_exact_dynamics_limitation": (
            "does not support exact cross-runtime internal homeostatic microdynamics"
            in report.get("claim_boundary", "")
        ),
        "forbidden_secret_scan_clean": (
            not forbidden_scan.get("enabled")
            or forbidden_scan.get("matches") == []
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "core_positive_report_count": len(core_rows),
            "state_artifact_count": len(state_rows),
            "negative_control_count": len(negative_rows),
            "limitation_evidence_count": len(limitation_rows),
            "forbidden_secret_match_count": len(forbidden_scan.get("matches", []))
            if isinstance(forbidden_scan, dict)
            else 0,
        },
    }


def audit_phase2homeostasis_bounded_publication_dossier(
    *,
    generation1_report_json: str | Path,
    generation2_py313_report_json: str | Path,
    generation2_py312_report_json: str | Path,
    chain_py313_report_json: str | Path,
    chain_py313_to_py312_report_json: str | Path,
    phase2cj_report_json: str | Path,
    fresh_rerun_limitations_report_json: str | Path,
    cross_runtime_dynamics_report_json: str | Path,
    no_key_negative_report_json: str | Path,
    state_jsons: list[str | Path],
    output_report_json: str | Path,
    output_markdown: str | Path | None = None,
    authenticity_key: str | bytes | None = None,
    forbidden_strings: list[str] | None = None,
    threshold_drift_limit: float = 0.01,
) -> dict[str, Any]:
    core_paths = [
        generation1_report_json,
        generation2_py313_report_json,
        generation2_py312_report_json,
        chain_py313_report_json,
        chain_py313_to_py312_report_json,
        phase2cj_report_json,
        fresh_rerun_limitations_report_json,
    ]
    core_rows = [
        _core_report_row(path, spec)
        for path, spec in zip(core_paths, CORE_REPORT_SPECS, strict=True)
    ]
    state_rows = [
        _state_row(path, authenticity_key=authenticity_key) for path in state_jsons
    ]
    negative_rows = [_negative_row(no_key_negative_report_json)]
    limitation_rows = [
        _cross_runtime_limitation_row(
            cross_runtime_dynamics_report_json,
            threshold_drift_limit=threshold_drift_limit,
        )
    ]
    scanned_paths = [
        *core_paths,
        cross_runtime_dynamics_report_json,
        no_key_negative_report_json,
        *state_jsons,
    ]
    forbidden_scan = _forbidden_secret_scan(
        scanned_paths,
        list(forbidden_strings or []),
    )
    checks = {
        "core_positive_reports_passed_and_bounded": all(
            row.get("readable") is True
            and row.get("passed") is True
            and row.get("family_matches_expected") is True
            and row.get("bounded_claim_ok") is True
            for row in core_rows
        ),
        "state_artifacts_hmac_bounded_and_valid": all(
            row.get("readable") is True
            and row.get("schema_valid") is True
            and row.get("bounded_state_keys_only") is True
            and row.get("authenticator_algorithm") == "hmac-sha256"
            and isinstance(row.get("key_fingerprint_sha256"), str)
            and row.get("integrity_valid") is True
            for row in state_rows
        ),
        "missing_key_negative_control_failed_closed": all(
            row.get("expected_failure_observed") is True for row in negative_rows
        ),
        "exact_cross_runtime_limitation_recorded": all(
            row.get("expected_failure_observed") is True
            and row.get("bounded_to_internal_dynamics") is True
            for row in limitation_rows
        ),
        "fresh_limitations_report_links_cross_runtime_failure": (
            _read_json(fresh_rerun_limitations_report_json)
            .get("evidence", {})
            .get("cross_runtime_dynamics_report_json")
            == str(cross_runtime_dynamics_report_json)
        ),
        "forbidden_secret_scan_clean": (
            not forbidden_scan.get("enabled")
            or forbidden_scan.get("matches") == []
        ),
    }
    passed = all(checks.values())
    markdown_path = Path(output_markdown) if output_markdown is not None else None
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        limitation = limitation_rows[0]
        markdown_path.write_text(
            "# Phase2Homeostasis Bounded Publication Dossier\n\n"
            + _markdown_table(core_rows)
            + "\n\n"
            + "Boundary evidence: exact cross-runtime internal homeostatic "
            + "microdynamics are not claimed. The observed max active-threshold "
            + f"delta is {limitation.get('maximum_active_threshold_delta')} "
            + f"with limit {threshold_drift_limit}; task completion and side-effect "
            + "behavior are preserved by the fresh rerun limitation audit.\n",
            encoding="utf-8",
        )
    report = {
        "artifact_family": "phase2homeostasis_bounded_publication_dossier",
        "passed": passed,
        "ready_for_bounded_homeostasis_publication_dossier_claim": passed,
        "ready_for_exact_cross_runtime_homeostatic_dynamics_claim": False,
        "ready_for_unbounded_long_term_memory_claim": False,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "core_positive_report_count": len(core_rows),
            "state_artifact_count": len(state_rows),
            "negative_control_count": len(negative_rows),
            "limitation_evidence_count": len(limitation_rows),
            "threshold_drift_limit": threshold_drift_limit,
            "maximum_active_threshold_delta": limitation_rows[0].get(
                "maximum_active_threshold_delta"
            ),
            "hmac_state_artifact_count": sum(
                row.get("authenticator_algorithm") == "hmac-sha256"
                for row in state_rows
            ),
            "forbidden_secret_match_count": len(forbidden_scan.get("matches", [])),
        },
        "core_positive_evidence": core_rows,
        "state_artifacts": state_rows,
        "negative_controls": negative_rows,
        "limitation_evidence": limitation_rows,
        "forbidden_secret_scan": forbidden_scan,
        "claim_boundary": (
            "Phase2Homeostasis supports a bounded HMAC-authenticated, "
            "package-scope, config-bound homeostatic persistent-state transfer "
            "mechanism over the recorded fresh rerun: task completion succeeds, "
            "HMAC state transfer is verified, the missing-key control fails closed, "
            "and side-effect traces are stable across Python 3.13/3.12. It does "
            "not support exact cross-runtime internal homeostatic microdynamics, "
            "unbounded or semantic long-term memory, free-form shell autonomy, "
            "open-ended native perception, production autonomy, or epoch-making "
            "architecture claims."
        ),
        "supported_claims": [
            (
                "bounded fresh behavioral mechanism evidence for "
                "HMAC-authenticated homeostatic persistent-state transfer with "
                "explicit internal cross-runtime microdynamics limitation"
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
            "external_replay_or_deterministic_microdynamics_calibration"
            if passed
            else "repair_phase2homeostasis_bounded_publication_dossier"
        ),
        "evidence": {
            "output_markdown": str(markdown_path) if markdown_path is not None else None,
            "auth_key_env_used": None,
            "cross_runtime_dynamics_report_json": str(cross_runtime_dynamics_report_json),
            "fresh_rerun_limitations_report_json": str(
                fresh_rerun_limitations_report_json
            ),
        },
    }
    _write_json(output_report_json, report)
    return report


def _env_key(env_name: str | None) -> str | None:
    if env_name is None:
        return None
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(f"auth key environment variable is not set: {env_name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a bounded Phase2Homeostasis publication dossier."
    )
    parser.add_argument("--generation1-report-json", required=True)
    parser.add_argument("--generation2-py313-report-json", required=True)
    parser.add_argument("--generation2-py312-report-json", required=True)
    parser.add_argument("--chain-py313-report-json", required=True)
    parser.add_argument("--chain-py313-to-py312-report-json", required=True)
    parser.add_argument("--phase2cj-report-json", required=True)
    parser.add_argument("--fresh-rerun-limitations-report-json", required=True)
    parser.add_argument("--cross-runtime-dynamics-report-json", required=True)
    parser.add_argument("--no-key-negative-report-json", required=True)
    parser.add_argument("--state-json", action="append", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--output-markdown")
    parser.add_argument("--auth-key-env")
    parser.add_argument("--forbidden-string", action="append", default=[])
    parser.add_argument("--threshold-drift-limit", type=float, default=0.01)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_bounded_publication_dossier(
        generation1_report_json=args.generation1_report_json,
        generation2_py313_report_json=args.generation2_py313_report_json,
        generation2_py312_report_json=args.generation2_py312_report_json,
        chain_py313_report_json=args.chain_py313_report_json,
        chain_py313_to_py312_report_json=args.chain_py313_to_py312_report_json,
        phase2cj_report_json=args.phase2cj_report_json,
        fresh_rerun_limitations_report_json=args.fresh_rerun_limitations_report_json,
        cross_runtime_dynamics_report_json=args.cross_runtime_dynamics_report_json,
        no_key_negative_report_json=args.no_key_negative_report_json,
        state_jsons=args.state_json,
        output_report_json=args.output_report_json,
        output_markdown=args.output_markdown,
        authenticity_key=_env_key(args.auth_key_env),
        forbidden_strings=args.forbidden_string,
        threshold_drift_limit=args.threshold_drift_limit,
    )
    if args.auth_key_env:
        report["evidence"]["auth_key_env_used"] = args.auth_key_env
        _write_json(args.output_report_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
