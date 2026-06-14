from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _split_repo_ids(manifest: dict[str, Any], split: str) -> set[str]:
    payload = manifest.get("splits", {}).get(split, {})
    repo_ids = payload.get("repo_ids", [])
    return {str(repo_id) for repo_id in repo_ids if repo_id}


def _split_rows(manifest: dict[str, Any], split: str) -> int:
    payload = manifest.get("splits", {}).get(split, {})
    try:
        return int(payload.get("rows", 0))
    except (TypeError, ValueError):
        return 0


def _repo_reports(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    repos = manifest.get("repos", [])
    return [repo for repo in repos if isinstance(repo, dict)]


def _repo_count(manifest: dict[str, Any]) -> int:
    repo_ids = set()
    for repo in _repo_reports(manifest):
        repo_id = repo.get("repo_id")
        if repo_id:
            repo_ids.add(str(repo_id))
    return len(repo_ids)


def _repos_with_rejections(manifest: dict[str, Any]) -> list[str]:
    rejected: list[str] = []
    for repo in _repo_reports(manifest):
        reasons = repo.get("rejected_reasons") or []
        if reasons:
            rejected.append(str(repo.get("repo_id") or "unknown"))
    return sorted(rejected)


def _all_repo_rows_complete(manifest: dict[str, Any]) -> bool:
    for repo in _repo_reports(manifest):
        try:
            requested = int(repo.get("rows_requested", -1))
            emitted = int(repo.get("rows_emitted", -2))
        except (TypeError, ValueError):
            return False
        if requested < 1 or emitted != requested:
            return False
    return bool(_repo_reports(manifest))


def _all_splits_disjoint(manifest: dict[str, Any]) -> bool:
    train = _split_repo_ids(manifest, "train")
    val = _split_repo_ids(manifest, "val")
    holdout = _split_repo_ids(manifest, "holdout")
    return bool(train and val and holdout) and train.isdisjoint(val) and train.isdisjoint(holdout) and val.isdisjoint(holdout)


def build_phase2q_public_breadth_gate(
    *,
    collector_manifest_json: str | Path,
    data_health_json: str | Path,
    phase2p_summary_json: str | Path | None = None,
    min_total_repos: int = 8,
    min_train_repos: int = 4,
    min_val_repos: int = 2,
    min_holdout_repos: int = 2,
    min_train_rows: int = 96,
    min_val_rows: int = 48,
    min_holdout_rows: int = 48,
) -> dict[str, Any]:
    manifest = _read_json(collector_manifest_json)
    data_health = _read_json(data_health_json)
    phase2p_summary = _read_json(phase2p_summary_json) if phase2p_summary_json else None

    train_repos = _split_repo_ids(manifest, "train")
    val_repos = _split_repo_ids(manifest, "val")
    holdout_repos = _split_repo_ids(manifest, "holdout")
    repo_count = _repo_count(manifest)
    rejected_repos = _repos_with_rejections(manifest)
    data_checks = data_health.get("checks", {})

    checks = {
        "collector_manifest_exists": Path(collector_manifest_json).exists(),
        "data_health_exists": Path(data_health_json).exists(),
        "sealed_not_used_by_collector": manifest.get("sealed_v3_used") is False,
        "collector_read_only": manifest.get("writes_to_collected_repos") is False,
        "structured_watch_keys_enabled": manifest.get("structured_watch_keys") is True,
        "behavior_summary_suppressed": manifest.get("include_behavior_summary") is False,
        "total_repo_count_met": repo_count >= min_total_repos,
        "train_repo_count_met": len(train_repos) >= min_train_repos,
        "val_repo_count_met": len(val_repos) >= min_val_repos,
        "holdout_repo_count_met": len(holdout_repos) >= min_holdout_repos,
        "train_rows_met": _split_rows(manifest, "train") >= min_train_rows,
        "val_rows_met": _split_rows(manifest, "val") >= min_val_rows,
        "holdout_rows_met": _split_rows(manifest, "holdout") >= min_holdout_rows,
        "all_split_repos_disjoint": _all_splits_disjoint(manifest),
        "repo_rows_complete": _all_repo_rows_complete(manifest),
        "no_repo_collection_rejections": not rejected_repos,
        "data_health_passed": data_health.get("passed") is True,
        "data_health_no_sealed_reference": data_checks.get("phase2m_no_sealed_reference_anywhere") is True,
        "data_health_no_candidate_slot_marker": data_checks.get("phase2m_no_candidate_slot_marker_visible") is True,
        "data_health_baselines_measured": data_checks.get("phase2m_baseline_metadata_measured") is True,
        "data_health_baselines_match_computed": data_checks.get("phase2m_baselines_match_computed_predictions") is True,
        "data_health_source_overlap_below_threshold": data_checks.get("phase2m_source_overlap_val_below_threshold") is True,
        "data_health_native_head_only_below_threshold": data_checks.get("phase2m_native_head_only_val_below_threshold") is True,
        "data_health_all_baselines_below_threshold": data_checks.get("phase2m_all_required_baselines_val_below_threshold") is True,
        "phase2p_boundary_available": phase2p_summary is None or phase2p_summary.get("passed") is True,
        "phase2p_not_training_signal": phase2p_summary is None
        or phase2p_summary.get("sealed_v3_used_for_training_sampling_or_tuning") is False,
    }

    blocked_actions: list[str] = []
    if not checks["sealed_not_used_by_collector"] or not checks["data_health_no_sealed_reference"]:
        blocked_actions.append("do_not_use_sealed_or_sealed_failure_feedback")
    if not checks["collector_read_only"]:
        blocked_actions.append("do_not_collect_public_trace_with_repo_write_side_effects")
    if not checks["structured_watch_keys_enabled"]:
        blocked_actions.append("do_not_train_phase2q_without_structured_watch_keys")
    if not checks["behavior_summary_suppressed"]:
        blocked_actions.append("do_not_train_phase2q_with_behavior_summary_shortcut")
    if not checks["total_repo_count_met"] or not checks["all_split_repos_disjoint"]:
        blocked_actions.append("do_not_train_phase2q_until_public_repo_breadth_passes")
    if not checks["data_health_passed"]:
        blocked_actions.append("do_not_train_phase2q_until_data_health_passes")
    if not checks["data_health_all_baselines_below_threshold"]:
        blocked_actions.append("do_not_train_phase2q_when_baseline_solves_val")
    if not checks["phase2p_not_training_signal"]:
        blocked_actions.append("do_not_use_phase2p_sealed_results_as_training_signal")

    passed = all(checks.values())
    return {
        "gate_family": "phase2q_public_trace_breadth_gate",
        "passed": passed,
        "allowed_next_action": (
            "run_phase2q_nonsealed_smoke_only" if passed else "fix_public_trace_or_audit_before_training"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "thresholds": {
            "min_total_repos": min_total_repos,
            "min_train_repos": min_train_repos,
            "min_val_repos": min_val_repos,
            "min_holdout_repos": min_holdout_repos,
            "min_train_rows": min_train_rows,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
        },
        "rollup": {
            "repo_count": repo_count,
            "train_repos": sorted(train_repos),
            "val_repos": sorted(val_repos),
            "holdout_repos": sorted(holdout_repos),
            "train_rows": _split_rows(manifest, "train"),
            "val_rows": _split_rows(manifest, "val"),
            "holdout_rows": _split_rows(manifest, "holdout"),
            "rejected_repos": rejected_repos,
            "data_health_rollups": data_health.get("rollups", {}),
        },
        "inputs": {
            "collector_manifest_json": str(Path(collector_manifest_json)),
            "data_health_json": str(Path(data_health_json)),
            "phase2p_summary_json": str(Path(phase2p_summary_json)) if phase2p_summary_json else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2Q public trace breadth before any claim-bearing training."
    )
    parser.add_argument("--collector-manifest-json", required=True)
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--phase2p-summary-json")
    parser.add_argument("--output-json")
    parser.add_argument("--min-total-repos", type=int, default=8)
    parser.add_argument("--min-train-repos", type=int, default=4)
    parser.add_argument("--min-val-repos", type=int, default=2)
    parser.add_argument("--min-holdout-repos", type=int, default=2)
    parser.add_argument("--min-train-rows", type=int, default=96)
    parser.add_argument("--min-val-rows", type=int, default=48)
    parser.add_argument("--min-holdout-rows", type=int, default=48)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2q_public_breadth_gate(
        collector_manifest_json=args.collector_manifest_json,
        data_health_json=args.data_health_json,
        phase2p_summary_json=args.phase2p_summary_json,
        min_total_repos=args.min_total_repos,
        min_train_repos=args.min_train_repos,
        min_val_repos=args.min_val_repos,
        min_holdout_repos=args.min_holdout_repos,
        min_train_rows=args.min_train_rows,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
