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
from reflexlm.cli.audit_phase2fa_replay_bundle_summary_reproducibility_manifest import (
    CONTROL_ARTIFACT_ROLES,
    REQUIRED_ARTIFACT_ROLES,
    REQUIRED_REPRODUCTION_STEPS,
    REQUIRED_SOURCE_REPORT_ROLES,
    validate_phase2fa_replay_bundle_summary_reproducibility_manifest,
)


OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _entries(manifest: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = manifest.get(key, [])
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def _copy_entries(
    *,
    entries: list[dict[str, Any]],
    target_dir: Path,
    update_current_hash: bool,
) -> list[dict[str, Any]]:
    copied_entries = []
    for index, item in enumerate(entries):
        source = Path(str(item.get("path", "")))
        if not source.exists():
            raise ValueError(f"Phase2FC source path missing: {source}")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"e{index:03d}_{source.name}"
        shutil.copy2(source, target)
        copied = deepcopy(item)
        copied["path"] = str(target)
        copied["bytes"] = target.stat().st_size
        if update_current_hash:
            copied["current_sha256"] = _sha256(target)
            copied["hash_matches_bundle_manifest"] = (
                copied.get("current_sha256") == copied.get("sha256")
            )
        copied_entries.append(copied)
    return copied_entries


def _rewrite_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [_rewrite_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, path_map) for key, item in value.items()}
    return value


def _role_set(entries: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("role")) for item in entries}


def _control_artifact_count(entries: list[dict[str, Any]]) -> int:
    return sum(item.get("role") in CONTROL_ARTIFACT_ROLES for item in entries)


def validate_phase2fc_replay_bundle_summary_manifest_cross_directory_replay(
    report: dict[str, Any],
) -> dict[str, Any]:
    replay_report_path = report.get("evidence", {}).get("replayed_phase2fa_report")
    replay_validation = {}
    replay_report_readable = False
    if replay_report_path:
        try:
            replay_report = _read_json(replay_report_path)
            replay_report_readable = True
            replay_validation = (
                validate_phase2fa_replay_bundle_summary_reproducibility_manifest(
                    replay_report
                )
            )
        except (OSError, json.JSONDecodeError):
            replay_report_readable = False
            replay_validation = {}
    checks = {
        "artifact_family_matches_phase2fc": (
            report.get("artifact_family")
            == "phase2fc_replay_bundle_summary_manifest_cross_directory_replay"
        ),
        "top_level_phase2fc_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_cross_directory_replay_claim") is True
            and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2fa_and_phase2fb_passed": (
            report.get("source_summary", {}).get("phase2fa_passed") is True
            and report.get("source_summary", {}).get("phase2fb_passed") is True
        ),
        "source_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2fb_negative_control_count")
            == report.get("source_summary", {}).get("phase2fb_negative_controls_failed")
        ),
        "replayed_phase2fa_report_readable": replay_report_readable,
        "replayed_phase2fa_validation_passed": replay_validation.get("passed") is True,
        "replayed_source_report_roles_complete": set(
            REQUIRED_SOURCE_REPORT_ROLES
        ).issubset(set(report.get("replay_summary", {}).get("source_report_roles", []))),
        "replayed_bundle_artifact_roles_complete": set(REQUIRED_ARTIFACT_ROLES).issubset(
            set(report.get("replay_summary", {}).get("bundle_artifact_roles", []))
        ),
        "replayed_control_artifact_count_preserved": (
            report.get("replay_summary", {}).get("control_artifact_count")
            == report.get("source_summary", {}).get("phase2fa_control_artifact_count")
        ),
        "replayed_reproduction_steps_complete": set(
            REQUIRED_REPRODUCTION_STEPS
        ).issubset(
            set(report.get("replay_summary", {}).get("reproduction_step_ids", []))
        ),
        "replay_directory_is_distinct": (
            report.get("replay_summary", {}).get("source_manifest_parent")
            != report.get("replay_summary", {}).get("replay_manifest_parent")
        ),
        "all_replayed_bundle_artifact_hashes_match": (
            report.get("replay_summary", {}).get("bundle_artifact_hash_match_count")
            == report.get("replay_summary", {}).get("bundle_artifact_count")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "replayed_source_report_count": report.get("replay_summary", {}).get(
                "source_report_count", 0
            ),
            "replayed_bundle_artifact_count": report.get("replay_summary", {}).get(
                "bundle_artifact_count", 0
            ),
            "replayed_control_artifact_count": report.get("replay_summary", {}).get(
                "control_artifact_count", 0
            ),
            "replayed_reproduction_step_count": report.get("replay_summary", {}).get(
                "reproduction_step_count", 0
            ),
        },
    }


def audit_phase2fc_replay_bundle_summary_manifest_cross_directory_replay(
    *,
    phase2fb_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2fb = _read_json(phase2fb_report_json)
    phase2fa_report_json = phase2fb.get("evidence", {}).get("phase2fa_report_json")
    if not phase2fa_report_json:
        raise ValueError("Phase2FC requires Phase2FB evidence.phase2fa_report_json")
    phase2fa = _read_json(phase2fa_report_json)
    source_manifest_path = Path(
        str(phase2fa.get("evidence", {}).get("reproducibility_manifest", ""))
    )
    if not source_manifest_path.exists():
        raise ValueError("Phase2FC requires a readable Phase2FA manifest")
    source_manifest = _read_json(source_manifest_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    replay_source_reports = _copy_entries(
        entries=_entries(source_manifest, "source_reports"),
        target_dir=output_root / "source_reports",
        update_current_hash=False,
    )
    replay_bundle_artifacts = _copy_entries(
        entries=_entries(source_manifest, "bundle_artifacts"),
        target_dir=output_root / "bundle_artifacts",
        update_current_hash=True,
    )
    path_map = {
        str(original.get("path")): str(replayed.get("path"))
        for original, replayed in zip(
            _entries(source_manifest, "source_reports")
            + _entries(source_manifest, "bundle_artifacts"),
            replay_source_reports + replay_bundle_artifacts,
        )
    }
    replay_manifest = deepcopy(source_manifest)
    replay_manifest["source_reports"] = replay_source_reports
    replay_manifest["bundle_artifacts"] = replay_bundle_artifacts
    replay_manifest["reproduction_steps"] = _rewrite_paths(
        source_manifest.get("reproduction_steps", []),
        path_map,
    )
    replay_manifest_path = output_root / "phase2fa_replayed_manifest.json"
    _write_json(replay_manifest_path, replay_manifest)
    replay_phase2fa_report = deepcopy(phase2fa)
    replay_phase2fa_report.setdefault("evidence", {})[
        "reproducibility_manifest"
    ] = str(replay_manifest_path)
    replay_phase2fa_report_path = output_root / "phase2fa_replayed_report.json"
    _write_json(replay_phase2fa_report_path, replay_phase2fa_report)
    replay_validation = (
        validate_phase2fa_replay_bundle_summary_reproducibility_manifest(
            replay_phase2fa_report
        )
    )
    replay_validation_path = output_root / "phase2fa_replayed_validation.json"
    _write_json(replay_validation_path, replay_validation)
    source_report_roles = sorted(_role_set(replay_source_reports))
    bundle_artifact_roles = sorted(_role_set(replay_bundle_artifacts))
    reproduction_step_ids = [
        str(item.get("step_id"))
        for item in replay_manifest.get("reproduction_steps", [])
        if isinstance(item, dict)
    ]
    bundle_hash_match_count = sum(
        item.get("hash_matches_bundle_manifest") is True
        for item in replay_bundle_artifacts
    )
    control_artifact_count = _control_artifact_count(replay_bundle_artifacts)
    checks = {
        "source_phase2fb_passed": phase2fb.get("passed") is True,
        "source_phase2fa_passed": phase2fa.get("passed") is True,
        "source_phase2fb_negative_controls_complete": (
            phase2fb.get("metrics", {}).get("negative_control_count")
            == phase2fb.get("metrics", {}).get("negative_controls_failed")
        ),
        "replay_manifest_written": replay_manifest_path.exists(),
        "replay_phase2fa_report_written": replay_phase2fa_report_path.exists(),
        "replay_phase2fa_validation_passed": replay_validation.get("passed") is True,
        "source_report_roles_preserved": set(REQUIRED_SOURCE_REPORT_ROLES).issubset(
            set(source_report_roles)
        ),
        "bundle_artifact_roles_preserved": set(REQUIRED_ARTIFACT_ROLES).issubset(
            set(bundle_artifact_roles)
        ),
        "bundle_artifact_count_preserved": len(replay_bundle_artifacts)
        == len(_entries(source_manifest, "bundle_artifacts")),
        "bundle_artifact_hashes_match_after_replay": bundle_hash_match_count
        == len(replay_bundle_artifacts),
        "control_artifact_count_preserved": control_artifact_count
        == phase2fa.get("metrics", {}).get("control_artifact_count"),
        "reproduction_steps_preserved": set(REQUIRED_REPRODUCTION_STEPS).issubset(
            set(reproduction_step_ids)
        ),
        "replay_directory_is_distinct": source_manifest_path.parent.resolve()
        != output_root.resolve(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2fc_replay_bundle_summary_manifest_cross_directory_replay",
        "passed": passed,
        "ready_for_bounded_cross_directory_replay_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "replayed_source_report_count": len(replay_source_reports),
            "replayed_bundle_artifact_count": len(replay_bundle_artifacts),
            "replayed_bundle_artifact_hash_match_count": bundle_hash_match_count,
            "replayed_control_artifact_count": control_artifact_count,
            "replayed_reproduction_step_count": len(reproduction_step_ids),
            "phase2fb_negative_control_count": phase2fb.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2fb_negative_controls_failed": phase2fb.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "source_summary": {
            "phase2fa_passed": phase2fa.get("passed") is True,
            "phase2fb_passed": phase2fb.get("passed") is True,
            "phase2fa_control_artifact_count": phase2fa.get("metrics", {}).get(
                "control_artifact_count"
            ),
            "phase2fb_negative_control_count": phase2fb.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2fb_negative_controls_failed": phase2fb.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "replay_summary": {
            "source_report_count": len(replay_source_reports),
            "source_report_roles": source_report_roles,
            "bundle_artifact_count": len(replay_bundle_artifacts),
            "bundle_artifact_roles": bundle_artifact_roles,
            "bundle_artifact_hash_match_count": bundle_hash_match_count,
            "control_artifact_count": control_artifact_count,
            "reproduction_step_count": len(reproduction_step_ids),
            "reproduction_step_ids": reproduction_step_ids,
            "source_manifest_parent": str(source_manifest_path.parent),
            "replay_manifest_parent": str(replay_manifest_path.parent),
        },
        "supported_claims": [
            (
                "cross-directory replay of the bounded Phase2FA reproducibility "
                "manifest with preserved source reports, bundle artifacts, copied "
                "control artifacts, hashes, reproduction steps, and boundary checks"
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
            "phase2fd_replay_bundle_summary_manifest_cross_directory_negative_controls"
            if passed
            else "repair_phase2fc_replay_bundle_summary_manifest_cross_directory_replay"
        ),
        "evidence": {
            "phase2fb_report_json": str(phase2fb_report_json),
            "phase2fa_report_json": str(phase2fa_report_json),
            "source_reproducibility_manifest": str(source_manifest_path),
            "replay_dir": str(output_root),
            "replayed_manifest": str(replay_manifest_path),
            "replayed_phase2fa_report": str(replay_phase2fa_report_path),
            "replayed_phase2fa_validation": str(replay_validation_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Phase2FA reproducibility manifest in a separate directory."
    )
    parser.add_argument("--phase2fb-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2fc_replay_bundle_summary_manifest_cross_directory_replay(
        phase2fb_report_json=args.phase2fb_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
