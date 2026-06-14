from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2homeostasis_persistent_state_chain import (
    _artifact_integrity_valid,
)
from reflexlm.runtime.homeostasis import (
    HOMEOSTATIC_CONTROLLER_SCHEMA,
    HOMEOSTATIC_PERSISTENT_STATE_SCHEMA,
    PERSISTENT_STATE_KEYS,
)


OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_unbounded_long_term_memory_claim",
    "ready_for_general_runtime_interpreter_invariance_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


POSITIVE_REPORT_SPECS: tuple[dict[str, str], ...] = (
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
        "role": "cross_runtime_dynamics",
        "family": "phase2homeostasis_cross_runtime_dynamics",
    },
    {
        "role": "runtime_interpreter_invariance",
        "family": "phase2cj_runtime_interpreter_invariance_audit",
    },
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _report_row(path: str | Path, spec: dict[str, str]) -> dict[str, Any]:
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
    metrics = report.get("metrics", {})
    compact_metrics = {
        key: metrics[key]
        for key in (
            "repositories",
            "episodes",
            "executed_actions",
            "task_completion_success_rate",
            "maximum_active_threshold_delta",
            "episodes_per_runtime",
            "task_completion_success_rate_per_runtime",
            "changed_episode_count",
        )
        if isinstance(metrics, dict) and key in metrics
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
        "compact_metrics": compact_metrics,
        "next_required_experiment": report.get("next_required_experiment"),
    }


def _authenticator(artifact: dict[str, Any]) -> dict[str, Any]:
    value = artifact.get("authenticator")
    return value if isinstance(value, dict) else {}


def _state_row(
    path: str | Path,
    *,
    authenticity_key: str | bytes | None = None,
) -> dict[str, Any]:
    try:
        artifact = _read_json(path)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        return {
            "state_json": str(path),
            "readable": False,
            "read_error": type(exc).__name__,
            "schema_valid": False,
            "bounded_state_keys_only": False,
            "authenticator_algorithm": None,
            "key_fingerprint_sha256": None,
            "integrity_valid": False,
        }
    authenticator = _authenticator(artifact)
    state = artifact.get("state")
    config = artifact.get("config")
    schema_valid = (
        artifact.get("schema_version") == HOMEOSTATIC_PERSISTENT_STATE_SCHEMA
        and artifact.get("controller_schema_version") == HOMEOSTATIC_CONTROLLER_SCHEMA
        and isinstance(config, dict)
        and artifact.get("config_fingerprint") == _stable_hash(config)
    )
    bounded_state_keys_only = (
        isinstance(state, dict) and set(state) == PERSISTENT_STATE_KEYS
    )
    return {
        "state_json": str(path),
        "readable": True,
        "schema_valid": schema_valid,
        "bounded_state_keys_only": bounded_state_keys_only,
        "authenticator_algorithm": authenticator.get("algorithm"),
        "key_fingerprint_sha256": authenticator.get("key_fingerprint_sha256"),
        "integrity_valid": _artifact_integrity_valid(
            artifact,
            authenticity_key=authenticity_key,
        ),
    }


def _negative_row(path: str | Path) -> dict[str, Any]:
    try:
        report = _read_json(path)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        return {
            "role": "hmac_missing_key_negative_control",
            "report_json": str(path),
            "readable": False,
            "read_error": type(exc).__name__,
            "expected_failure_observed": False,
        }
    checks = report.get("checks", {})
    return {
        "role": "hmac_missing_key_negative_control",
        "report_json": str(path),
        "readable": True,
        "passed": report.get("passed") is True,
        "expected_failure_observed": (
            report.get("passed") is False
            and checks.get("both_artifact_integrities_valid") is False
            and report.get("ready_for_bounded_cross_process_homeostatic_memory_claim")
            is False
        ),
        "artifact_family": report.get("artifact_family"),
        "failed_check": "both_artifact_integrities_valid",
    }


def _forbidden_secret_scan(
    paths: list[str | Path],
    forbidden_strings: list[str],
) -> dict[str, Any]:
    if not forbidden_strings:
        return {"enabled": False, "matches": []}
    matches: list[dict[str, str]] = []
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
        except OSError:
            continue
        for secret in forbidden_strings:
            if secret and secret in text:
                matches.append({"path": str(path), "secret_sha256": _stable_hash(secret)})
    return {"enabled": True, "matches": matches}


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


def validate_phase2homeostasis_publication_bundle(
    report: dict[str, Any],
) -> dict[str, Any]:
    positive_rows = report.get("positive_evidence", [])
    state_rows = report.get("state_artifacts", [])
    negative_rows = report.get("negative_controls", [])
    if not isinstance(positive_rows, list):
        positive_rows = []
    if not isinstance(state_rows, list):
        state_rows = []
    if not isinstance(negative_rows, list):
        negative_rows = []
    forbidden_scan = report.get("forbidden_secret_scan", {})
    checks = {
        "artifact_family_matches": (
            report.get("artifact_family")
            == "phase2homeostasis_publication_bundle"
        ),
        "top_level_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_homeostasis_publication_bundle_claim")
            is True
            and all(report.get(flag) is not True for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "positive_evidence_count_met": len(positive_rows) == len(POSITIVE_REPORT_SPECS),
        "all_positive_reports_readable_passed_and_bounded": bool(positive_rows)
        and all(
            row.get("readable") is True
            and row.get("passed") is True
            and row.get("family_matches_expected") is True
            and row.get("bounded_claim_ok") is True
            for row in positive_rows
        ),
        "state_artifact_count_met": len(state_rows) >= 3,
        "all_state_artifacts_are_hmac_v3_bounded_and_valid": bool(state_rows)
        and all(
            row.get("readable") is True
            and row.get("schema_valid") is True
            and row.get("bounded_state_keys_only") is True
            and row.get("authenticator_algorithm") == "hmac-sha256"
            and isinstance(row.get("key_fingerprint_sha256"), str)
            and row.get("integrity_valid") is True
            for row in state_rows
        ),
        "hmac_missing_key_negative_control_failed_closed": bool(negative_rows)
        and all(row.get("expected_failure_observed") is True for row in negative_rows),
        "forbidden_secret_scan_clean": (
            not forbidden_scan.get("enabled")
            or forbidden_scan.get("matches") == []
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "positive_report_count": len(positive_rows),
            "state_artifact_count": len(state_rows),
            "negative_control_count": len(negative_rows),
            "forbidden_secret_match_count": len(forbidden_scan.get("matches", []))
            if isinstance(forbidden_scan, dict)
            else 0,
        },
    }


def audit_phase2homeostasis_publication_bundle(
    *,
    generation1_report_json: str | Path,
    generation2_py313_report_json: str | Path,
    generation2_py312_report_json: str | Path,
    chain_py313_report_json: str | Path,
    chain_py313_to_py312_report_json: str | Path,
    cross_runtime_dynamics_report_json: str | Path,
    phase2cj_report_json: str | Path,
    no_key_negative_report_json: str | Path,
    state_jsons: list[str | Path],
    output_report_json: str | Path,
    output_markdown: str | Path | None = None,
    authenticity_key: str | bytes | None = None,
    forbidden_strings: list[str] | None = None,
) -> dict[str, Any]:
    positive_paths = [
        generation1_report_json,
        generation2_py313_report_json,
        generation2_py312_report_json,
        chain_py313_report_json,
        chain_py313_to_py312_report_json,
        cross_runtime_dynamics_report_json,
        phase2cj_report_json,
    ]
    positive_rows = [
        _report_row(path, spec)
        for path, spec in zip(positive_paths, POSITIVE_REPORT_SPECS, strict=True)
    ]
    state_rows = [
        _state_row(path, authenticity_key=authenticity_key) for path in state_jsons
    ]
    negative_rows = [_negative_row(no_key_negative_report_json)]
    scanned_paths = [*positive_paths, no_key_negative_report_json, *state_jsons]
    forbidden_scan = _forbidden_secret_scan(
        scanned_paths,
        list(forbidden_strings or []),
    )
    checks = {
        "all_positive_reports_passed_and_bounded": all(
            row.get("readable") is True
            and row.get("passed") is True
            and row.get("family_matches_expected") is True
            and row.get("bounded_claim_ok") is True
            for row in positive_rows
        ),
        "all_state_artifacts_hmac_v3_bounded_and_valid": all(
            row.get("readable") is True
            and row.get("schema_valid") is True
            and row.get("bounded_state_keys_only") is True
            and row.get("authenticator_algorithm") == "hmac-sha256"
            and isinstance(row.get("key_fingerprint_sha256"), str)
            and row.get("integrity_valid") is True
            for row in state_rows
        ),
        "hmac_missing_key_negative_control_failed_closed": all(
            row.get("expected_failure_observed") is True for row in negative_rows
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
        markdown_path.write_text(
            "# Phase2Homeostasis HMAC v3 Publication Bundle\n\n"
            + _markdown_table(positive_rows)
            + "\n\n"
            + "Claim boundary: bounded package-scoped persistent homeostatic "
            + "state transfer and cross-runtime dynamics only. This does not "
            + "support unbounded memory, open-ended native perception, production "
            + "autonomy, or epoch-making architecture claims.\n",
            encoding="utf-8",
        )
    report = {
        "artifact_family": "phase2homeostasis_publication_bundle",
        "passed": passed,
        "ready_for_bounded_homeostasis_publication_bundle_claim": passed,
        "ready_for_unbounded_long_term_memory_claim": False,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "positive_report_count": len(positive_rows),
            "state_artifact_count": len(state_rows),
            "negative_control_count": len(negative_rows),
            "hmac_state_artifact_count": sum(
                row.get("authenticator_algorithm") == "hmac-sha256"
                for row in state_rows
            ),
            "runtime_count": 2,
            "forbidden_secret_match_count": len(forbidden_scan.get("matches", [])),
        },
        "positive_evidence": positive_rows,
        "state_artifacts": state_rows,
        "negative_controls": negative_rows,
        "forbidden_secret_scan": forbidden_scan,
        "claim_boundary": (
            "Phase2Homeostasis HMAC v3 supports a bounded, package-scoped, "
            "config-bound, controller-schema-bound homeostatic persistent-state "
            "mechanism with keyed authenticity, closed missing-key negative control, "
            "and exact cross-runtime homeostatic dynamics across the recorded Python "
            "3.13 and 3.12 runtimes. It does not support unbounded or semantic "
            "long-term memory, free-form shell autonomy, open-ended native "
            "perception, production autonomy, or epoch-making architecture claims."
        ),
        "supported_claims": [
            (
                "bounded HMAC-authenticated cross-process and cross-runtime "
                "homeostatic persistent-state mechanism evidence bundle"
            )
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
            "fresh_directory_replay_of_hmac_homeostatic_publication_bundle"
            if passed
            else "repair_phase2homeostasis_publication_bundle"
        ),
        "evidence": {
            "output_markdown": str(markdown_path) if markdown_path is not None else None,
            "auth_key_env_used": None,
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
        description="Build a bounded Phase2Homeostasis HMAC v3 publication bundle."
    )
    parser.add_argument("--generation1-report-json", required=True)
    parser.add_argument("--generation2-py313-report-json", required=True)
    parser.add_argument("--generation2-py312-report-json", required=True)
    parser.add_argument("--chain-py313-report-json", required=True)
    parser.add_argument("--chain-py313-to-py312-report-json", required=True)
    parser.add_argument("--cross-runtime-dynamics-report-json", required=True)
    parser.add_argument("--phase2cj-report-json", required=True)
    parser.add_argument("--no-key-negative-report-json", required=True)
    parser.add_argument("--state-json", action="append", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--output-markdown")
    parser.add_argument("--auth-key-env")
    parser.add_argument("--forbidden-string", action="append", default=[])
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_publication_bundle(
        generation1_report_json=args.generation1_report_json,
        generation2_py313_report_json=args.generation2_py313_report_json,
        generation2_py312_report_json=args.generation2_py312_report_json,
        chain_py313_report_json=args.chain_py313_report_json,
        chain_py313_to_py312_report_json=args.chain_py313_to_py312_report_json,
        cross_runtime_dynamics_report_json=args.cross_runtime_dynamics_report_json,
        phase2cj_report_json=args.phase2cj_report_json,
        no_key_negative_report_json=args.no_key_negative_report_json,
        state_jsons=args.state_json,
        output_report_json=args.output_report_json,
        output_markdown=args.output_markdown,
        authenticity_key=_env_key(args.auth_key_env),
        forbidden_strings=args.forbidden_string,
    )
    if args.auth_key_env:
        report["evidence"]["auth_key_env_used"] = args.auth_key_env
        _write_json(args.output_report_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
