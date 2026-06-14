from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2t_architecture_iteration import (
    REQUIRED_ARCHITECTURE_COMPONENTS,
    REQUIRED_FACTOR_LEVELS,
    REQUIRED_GRADED_FACTORS,
    REQUIRED_TASK_FAMILIES,
)


REQUIRED_ARTIFACT_KEYS = {
    "patch_diff",
    "command_log",
    "test_output",
    "rollback_log",
    "sandbox_integrity_report",
}
REQUIRED_BASELINES = {
    "source_overlap",
    "native_head_only_no_cache",
    "continuation_only",
    "prompt_only",
    "react",
    "modern_coding_agent_loop",
}
FORBIDDEN_MARKERS = (
    "external_trace_v3_semantic_required",
    "phase2i_external_trace",
    "phase2g_external_trace",
    "sealed_failure",
    "sealed_feedback",
    "hidden_hint",
    "gold_label",
    "gold_command",
    "correct_patch",
    "expected_patch",
    "correct_command_hint",
)
CANDIDATE_SLOT_RE = re.compile(r"(?i)\bcandidate[_-]?\d+\b|\bslot[_-]?\d+\b")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path | None) -> tuple[list[dict[str, Any]], bool]:
    if path is None:
        return [], False
    candidate = Path(path)
    if not candidate.exists():
        return [], False
    rows: list[dict[str, Any]] = []
    for line in candidate.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows, True


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _visible_payload(row: dict[str, Any]) -> str:
    return _canonical_json(
        {
            "current_visible_text": row.get("current_visible_text"),
            "runtime_visible_evidence": row.get("runtime_visible_evidence"),
            "repair_candidates": row.get("repair_candidates"),
        }
    )


def _row_mentions_forbidden_marker(row: dict[str, Any]) -> bool:
    text = _visible_payload(row).replace("\\", "/").lower()
    return any(marker in text for marker in FORBIDDEN_MARKERS) or bool(CANDIDATE_SLOT_RE.search(text))


def _value_mentions_markers(value: Any, markers: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(_value_mentions_markers(item, markers) for item in value.values())
    if isinstance(value, list):
        return any(_value_mentions_markers(item, markers) for item in value)
    if isinstance(value, str):
        text = value.replace("\\", "/").lower()
        return any(marker in text for marker in markers)
    return False


def _row_mentions_sealed_anywhere(row: dict[str, Any]) -> bool:
    # Audit values, not metadata key names such as "uses_sealed_feedback": false.
    return _value_mentions_markers(row, FORBIDDEN_MARKERS[:5])


def _artifact_paths_ok(row: dict[str, Any], root: Path) -> bool:
    paths = _dict(row.get("artifact_paths"))
    if not REQUIRED_ARTIFACT_KEYS.issubset(paths):
        return False
    return all((root / str(paths[key])).exists() for key in REQUIRED_ARTIFACT_KEYS)


def _architecture_targets_ok(row: dict[str, Any]) -> bool:
    targets = _dict(row.get("architecture_targets"))
    return all(_dict(targets.get(key)).get("required") is True for key in REQUIRED_ARCHITECTURE_COMPONENTS)


def _repair_loop_ok(row: dict[str, Any]) -> bool:
    episode = _dict(row.get("repair_loop_episode"))
    stages = [str(_dict(stage).get("stage")) for stage in _list(episode.get("stages"))]
    return (
        episode.get("loop_schema") == "phase2t_repair_loop_v1"
        and "inspect_runtime_evidence" in stages
        and "propose_bounded_patch" in stages
        and "run_verification_tests" in stages
        and "rollback_failed_or_unsafe_patch" in stages
        and "emit_verified_stop" in stages
    )


def _safety_controls_ok(row: dict[str, Any]) -> bool:
    safety = _dict(row.get("safety_controls"))
    return (
        safety.get("source_repo_read_only_observed") is True
        and safety.get("bounded_edit_scope_observed") is True
        and safety.get("command_allowlist_observed") is True
        and safety.get("rollback_recorded") is True
        and safety.get("sandbox_cleanup_recorded") is True
        and safety.get("stop_requires_verification") is True
        and safety.get("unauthorized_write_count") == 0
        and safety.get("low_level_qwen_calls") == 0
    )


def _modern_baseline_contract_ok(row: dict[str, Any]) -> bool:
    contract = _dict(row.get("modern_baseline_contract"))
    return (
        contract.get("required") is True
        and contract.get("measured_not_declared") is True
        and contract.get("same_repair_loop_artifacts_required") is True
        and contract.get("cost_and_command_budget_required") is True
    )


def _baseline_metadata_ok(row: dict[str, Any]) -> bool:
    metadata = _dict(row.get("baseline_metadata"))
    baselines = _dict(row.get("baselines"))
    if not REQUIRED_BASELINES.issubset(metadata) or not REQUIRED_BASELINES.issubset(baselines):
        return False
    for baseline in REQUIRED_BASELINES:
        payload = _dict(metadata.get(baseline))
        if payload.get("measured") is not True:
            return False
        if payload.get("uses_expected_repair_action") is not False:
            return False
        if payload.get("uses_sealed_feedback") is not False:
            return False
    return True


def _row_shape_ok(row: dict[str, Any], root: Path) -> bool:
    return (
        row.get("phase") == "Phase2T"
        and row.get("trace_construction_mode")
        == "phase2t_dynamic_public_repo_repair_loop_trace"
        and row.get("source_kind") == "public_repo"
        and _repair_loop_ok(row)
        and _architecture_targets_ok(row)
        and _safety_controls_ok(row)
        and _modern_baseline_contract_ok(row)
        and _baseline_metadata_ok(row)
        and _artifact_paths_ok(row, root)
    )


def _split_rows_have_split(rows: list[dict[str, Any]], split: str) -> bool:
    return all(str(row.get("split")) == split for row in rows)


def _all_split_repos_disjoint(*splits: list[dict[str, Any]]) -> bool:
    split_origins: list[set[str]] = [
        {str(row.get("repo_url_or_origin") or row.get("repo_id") or "") for row in rows}
        for rows in splits
    ]
    for index, origins in enumerate(split_origins):
        for other in split_origins[index + 1 :]:
            if origins & other:
                return False
    return True


def _dedup_ok(rows: list[dict[str, Any]]) -> bool:
    keys = [
        (
            str(row.get("repo_url_or_origin")),
            str(row.get("commit_hash")),
            str(row.get("trace_hash")),
        )
        for row in rows
    ]
    return len(keys) == len(set(keys))


def _difficulty(row: dict[str, Any], key: str) -> Any:
    difficulty = _dict(row.get("difficulty"))
    return difficulty.get(key)


def _task_family_coverage(rows: list[dict[str, Any]]) -> list[str]:
    observed = {str(_difficulty(row, "task_family")) for row in rows}
    return sorted(REQUIRED_TASK_FAMILIES - observed)


def _factor_coverage(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for factor, required in REQUIRED_FACTOR_LEVELS.items():
        observed = {str(_difficulty(row, factor)) for row in rows}
        diff = sorted(required - observed)
        if diff:
            missing[factor] = diff
    return missing


def _rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "repos": sorted({str(row.get("repo_id")) for row in rows}),
        "task_families": sorted({str(_difficulty(row, "task_family")) for row in rows}),
        "factor_levels": {
            factor: sorted({str(_difficulty(row, factor)) for row in rows})
            for factor in REQUIRED_GRADED_FACTORS
        },
    }


def build_phase2t_data_health(
    *,
    manifest_json: str | Path,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    dataset_root: str | Path | None = None,
    min_train_rows: int = 24,
    min_val_rows: int = 16,
    min_holdout_rows: int = 16,
) -> dict[str, Any]:
    manifest = _read_json(manifest_json)
    train_rows, train_exists = _read_jsonl(train_jsonl)
    val_rows, val_exists = _read_jsonl(val_jsonl)
    holdout_rows, holdout_exists = _read_jsonl(holdout_jsonl)
    all_rows = train_rows + val_rows + holdout_rows
    root = Path(dataset_root) if dataset_root else Path(train_jsonl).parent
    missing_train_task_families = _task_family_coverage(train_rows)
    missing_train_factor_levels = _factor_coverage(train_rows)
    missing_val_task_families = _task_family_coverage(val_rows)
    missing_val_factor_levels = _factor_coverage(val_rows)
    missing_holdout_task_families = _task_family_coverage(holdout_rows)
    missing_holdout_factor_levels = _factor_coverage(holdout_rows)
    checks = {
        "phase2t_manifest_exists": Path(manifest_json).exists(),
        "phase2t_collector_family_dynamic": manifest.get("collector_family")
        == "phase2t_dynamic_public_repo_repair_loop_trace_collector",
        "phase2t_manifest_training_still_blocked": manifest.get("claim_bearing_training_ready")
        is False,
        "phase2t_manifest_no_sealed_use": manifest.get("sealed_v3_used") is False,
        "phase2t_manifest_source_repos_readonly": manifest.get("writes_to_source_repos") is False,
        "phase2t_manifest_execution_sandbox_used": manifest.get("execution_sandbox_used") is True,
        "phase2t_train_jsonl_exists": train_exists,
        "phase2t_val_jsonl_exists": val_exists,
        "phase2t_holdout_jsonl_exists": holdout_exists,
        "phase2t_train_rows_minimum_met": len(train_rows) >= min_train_rows,
        "phase2t_val_rows_minimum_met": len(val_rows) >= min_val_rows,
        "phase2t_holdout_rows_minimum_met": len(holdout_rows) >= min_holdout_rows,
        "phase2t_split_labels_match_files": _split_rows_have_split(train_rows, "train")
        and _split_rows_have_split(val_rows, "val")
        and _split_rows_have_split(holdout_rows, "holdout"),
        "phase2t_rows_have_repair_loop_schema": bool(all_rows)
        and all(_row_shape_ok(row, root) for row in all_rows),
        "phase2t_no_forbidden_visible_markers": not any(
            _row_mentions_forbidden_marker(row) for row in all_rows
        ),
        "phase2t_no_sealed_reference_anywhere": not any(
            _row_mentions_sealed_anywhere(row) for row in all_rows
        ),
        "phase2t_split_repos_disjoint": _all_split_repos_disjoint(
            train_rows, val_rows, holdout_rows
        ),
        "phase2t_deduplicated_by_repo_commit_trace_hash": bool(all_rows)
        and _dedup_ok(all_rows),
        "phase2t_train_task_family_coverage": not missing_train_task_families,
        "phase2t_train_factor_level_coverage": not missing_train_factor_levels,
        "phase2t_val_task_family_coverage": not missing_val_task_families,
        "phase2t_val_factor_level_coverage": not missing_val_factor_levels,
        "phase2t_holdout_task_family_coverage": not missing_holdout_task_families,
        "phase2t_holdout_factor_level_coverage": not missing_holdout_factor_levels,
    }
    blocked_actions: list[str] = []
    if not all(checks.values()):
        blocked_actions.append("do_not_train_phase2t_until_data_health_passes")
    if not checks["phase2t_manifest_training_still_blocked"]:
        blocked_actions.append("do_not_skip_phase2t_data_health_with_collector_manifest")
    if not checks["phase2t_manifest_no_sealed_use"] or not checks["phase2t_no_sealed_reference_anywhere"]:
        blocked_actions.append("do_not_use_sealed_or_sealed_failure_feedback")
    if not checks["phase2t_rows_have_repair_loop_schema"]:
        blocked_actions.append("do_not_train_phase2t_without_repair_loop_schema")
    if not checks["phase2t_no_forbidden_visible_markers"]:
        blocked_actions.append("do_not_train_phase2t_with_gold_candidate_or_patch_markers")
    if not checks["phase2t_split_repos_disjoint"]:
        blocked_actions.append("do_not_train_phase2t_without_repo_disjoint_splits")
    if not checks["phase2t_train_task_family_coverage"] or not checks[
        "phase2t_train_factor_level_coverage"
    ]:
        blocked_actions.append("do_not_train_phase2t_until_train_pressure_matrix_is_covered")
    if not checks["phase2t_val_task_family_coverage"] or not checks[
        "phase2t_val_factor_level_coverage"
    ]:
        blocked_actions.append("do_not_train_phase2t_until_val_pressure_matrix_is_covered")
    if not checks["phase2t_holdout_task_family_coverage"] or not checks[
        "phase2t_holdout_factor_level_coverage"
    ]:
        blocked_actions.append("do_not_train_phase2t_until_holdout_pressure_matrix_is_covered")

    passed = all(checks.values())
    split_hashes = {
        "phase2t_train": _sha256(train_rows),
        "phase2t_val": _sha256(val_rows),
        "phase2t_holdout": _sha256(holdout_rows),
    }
    return {
        "audit_family": "phase2t_dynamic_repair_trace_data_health",
        "passed": passed,
        "claim_bearing_training_ready": passed,
        "allowed_next_action": (
            "run_phase2t_claim_bearing_smoke_training_only"
            if passed
            else "revise_phase2t_dynamic_repair_traces_before_training"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "thresholds": {
            "min_train_rows": min_train_rows,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "required_task_families": sorted(REQUIRED_TASK_FAMILIES),
            "required_factor_levels": {
                key: sorted(values) for key, values in REQUIRED_FACTOR_LEVELS.items()
            },
            "required_artifact_keys": sorted(REQUIRED_ARTIFACT_KEYS),
            "required_baselines": sorted(REQUIRED_BASELINES),
        },
        "rollups": {
            "train": _rollup(train_rows),
            "val": _rollup(val_rows),
            "holdout": _rollup(holdout_rows),
            "missing_train_task_families": missing_train_task_families,
            "missing_train_factor_levels": missing_train_factor_levels,
            "missing_val_task_families": missing_val_task_families,
            "missing_val_factor_levels": missing_val_factor_levels,
            "missing_holdout_task_families": missing_holdout_task_families,
            "missing_holdout_factor_levels": missing_holdout_factor_levels,
        },
        "effective_split_hashes": split_hashes,
        "inputs": {
            "manifest_json": str(Path(manifest_json)),
            "train_jsonl": str(Path(train_jsonl)),
            "val_jsonl": str(Path(val_jsonl)),
            "holdout_jsonl": str(Path(holdout_jsonl)),
            "dataset_root": str(root),
        },
    }


def build_phase2t_pretrain_gate(*, data_health_json: str | Path) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    split_hashes = _dict(data_health.get("effective_split_hashes"))
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "claim_bearing_training_ready": data_health.get("claim_bearing_training_ready") is True,
        "effective_split_hashes_present": all(
            split_hashes.get(key)
            for key in ("phase2t_train", "phase2t_val", "phase2t_holdout")
        ),
        "repair_loop_schema_present": data_health.get("checks", {}).get(
            "phase2t_rows_have_repair_loop_schema"
        )
        is True,
        "repo_disjoint_splits": data_health.get("checks", {}).get(
            "phase2t_split_repos_disjoint"
        )
        is True,
        "val_pressure_matrix_covered": data_health.get("checks", {}).get(
            "phase2t_val_task_family_coverage"
        )
        is True
        and data_health.get("checks", {}).get("phase2t_val_factor_level_coverage") is True,
        "train_pressure_matrix_covered": data_health.get("checks", {}).get(
            "phase2t_train_task_family_coverage"
        )
        is True
        and data_health.get("checks", {}).get("phase2t_train_factor_level_coverage") is True,
        "holdout_pressure_matrix_covered": data_health.get("checks", {}).get(
            "phase2t_holdout_task_family_coverage"
        )
        is True
        and data_health.get("checks", {}).get("phase2t_holdout_factor_level_coverage") is True,
        "sealed_not_used": data_health.get("checks", {}).get(
            "phase2t_no_sealed_reference_anywhere"
        )
        is True,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_train_phase2t_until_pretrain_gate_passes")
    return {
        "audit_family": "phase2t_dynamic_repair_trace_pretrain_gate",
        "passed": passed,
        "allowed_next_action": (
            "run_phase2t_claim_bearing_smoke_training_only"
            if passed
            else "revise_phase2t_dynamic_repair_traces_before_training"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "effective_split_hashes": split_hashes,
        "inputs": {"data_health_json": str(Path(data_health_json))},
    }


def _latest_val_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    history = _list(summary.get("history"))
    for item in reversed(history):
        if isinstance(item, dict) and isinstance(item.get("val_metrics"), dict):
            return item["val_metrics"]
    return {}


def _metric_from_summary(summary: dict[str, Any], metric: str) -> float | None:
    val_metrics = _latest_val_metrics(summary)
    value = val_metrics.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _smoke_duration_seconds(summary: dict[str, Any]) -> float | None:
    run_manifest = _dict(summary.get("run_manifest"))
    duration = run_manifest.get("duration_seconds")
    if isinstance(duration, (int, float)):
        return float(duration)
    total = 0.0
    observed = False
    for item in _list(summary.get("history")):
        if not isinstance(item, dict):
            continue
        train_elapsed = item.get("train_elapsed_seconds")
        if isinstance(train_elapsed, (int, float)):
            total += float(train_elapsed)
            observed = True
        val_metrics = _dict(item.get("val_metrics"))
        val_elapsed = val_metrics.get("elapsed_seconds")
        if isinstance(val_elapsed, (int, float)):
            total += float(val_elapsed)
            observed = True
    return total if observed else None


def build_phase2t_smoke_postflight(
    *,
    training_summary_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    head_manifest_json: str | Path | None = None,
    min_val_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.10,
    max_smoke_duration_seconds: float = 3600.0,
) -> dict[str, Any]:
    summary = _read_json(training_summary_json)
    data_health = _read_json(data_health_json)
    pretrain_gate = _read_json(pretrain_gate_json)
    head_manifest = _read_json(head_manifest_json) if head_manifest_json else {}

    data_hashes = _dict(data_health.get("effective_split_hashes"))
    gate_hashes = _dict(pretrain_gate.get("effective_split_hashes"))
    head_hashes = _dict(head_manifest.get("effective_split_hashes"))
    summary_hashes = _dict(summary.get("effective_split_hashes"))
    val_accuracy = _metric_from_summary(summary, "command_slot_accuracy")
    val_count = _metric_from_summary(summary, "command_slot_count")
    source_overlap_val = _dict(
        _dict(summary.get("source_overlap_command_slot_baseline")).get("val")
    )
    source_overlap_accuracy = source_overlap_val.get("accuracy")
    if isinstance(source_overlap_accuracy, (int, float)):
        source_overlap_accuracy = float(source_overlap_accuracy)
    else:
        source_overlap_accuracy = None
    model_minus_source_overlap = (
        float(val_accuracy) - source_overlap_accuracy
        if isinstance(val_accuracy, float) and isinstance(source_overlap_accuracy, float)
        else None
    )
    duration_seconds = _smoke_duration_seconds(summary)
    config = _dict(summary.get("config"))
    head_config = _dict(summary.get("head_config"))
    pairwise = summary.get("use_pairwise_command_reranker")
    if pairwise is None:
        pairwise = config.get("use_pairwise_command_reranker")
    candidate_encoder = summary.get("command_candidate_encoder") or config.get(
        "command_candidate_encoder"
    )
    latent_fusion = config.get("latent_fusion") or head_config.get("latent_fusion")
    pairwise_candidate_encoding = _dict(summary.get("pairwise_candidate_encoding"))
    pairwise_train = _dict(pairwise_candidate_encoding.get("train"))
    pairwise_val = _dict(pairwise_candidate_encoding.get("val"))
    pairwise_encoded_candidates = (
        int(pairwise_train.get("pairwise_scored_candidates", 0) or 0)
        + int(pairwise_val.get("pairwise_scored_candidates", 0) or 0)
    )
    head_splits = _dict(head_manifest.get("splits"))
    train_split = _dict(head_splits.get("train"))
    val_split = _dict(head_splits.get("val"))
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain_gate.get("passed") is True,
        "head_manifest_family_phase2t": (
            not head_manifest
            or head_manifest.get("dataset_family") == "phase2t_dynamic_repair_head_dataset"
        ),
        "head_manifest_source_gates_passed": (
            not head_manifest
            or (
                head_manifest.get("source_data_health_passed") is True
                and head_manifest.get("source_pretrain_gate_passed") is True
            )
        ),
        "head_manifest_command_identity_margin_gate_passed": (
            not head_manifest
            or head_manifest.get("command_identity_margin_gate_passed") is True
        ),
        "data_and_pretrain_hashes_match": bool(data_hashes)
        and bool(gate_hashes)
        and data_hashes == gate_hashes,
        "head_manifest_hashes_match_data_health": (
            not head_manifest or head_hashes == data_hashes
        ),
        "training_summary_head_hashes_present": all(
            summary_hashes.get(key) for key in ("phase2c_head_train", "phase2c_head_val")
        ),
        "training_counts_match_head_manifest": (
            not head_manifest
            or (
                summary.get("train_examples") == train_split.get("rows")
                and summary.get("val_examples") == val_split.get("rows")
            )
        ),
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "pairwise_disabled_for_phase2t_smoke": pairwise is False,
        "pairwise_encoded_candidates_zero": pairwise_encoded_candidates == 0,
        "features_only_candidate_encoder": candidate_encoder == "features_only",
        "additive_latent_fusion": latent_fusion == "additive",
        "val_command_slot_accuracy_min": isinstance(val_accuracy, float)
        and val_accuracy >= min_val_command_slot_accuracy,
        "model_minus_source_overlap_min": isinstance(model_minus_source_overlap, float)
        and model_minus_source_overlap >= min_model_minus_source_overlap,
        "smoke_duration_within_limit": isinstance(duration_seconds, float)
        and duration_seconds <= max_smoke_duration_seconds,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_run_phase2t_full_train_until_smoke_postflight_passes")
    if not checks["val_command_slot_accuracy_min"]:
        blocked_actions.append("do_not_claim_phase2t_val_gate")
    if not checks["model_minus_source_overlap_min"]:
        blocked_actions.append("do_not_claim_phase2t_mechanism_delta_from_source_overlap")
    if not checks["pairwise_disabled_for_phase2t_smoke"]:
        blocked_actions.append("do_not_mix_phase2t_smoke_with_pairwise_mechanism")
    if not checks["pairwise_encoded_candidates_zero"]:
        blocked_actions.append("do_not_allow_phase2t_smoke_with_pairwise_candidate_scoring")
    if not checks["head_manifest_command_identity_margin_gate_passed"]:
        blocked_actions.append("do_not_train_phase2t_with_ambiguous_command_identity_latent")
    if not checks["data_and_pretrain_hashes_match"] or not checks[
        "head_manifest_hashes_match_data_health"
    ]:
        blocked_actions.append("do_not_train_or_package_with_phase2t_hash_mismatch")
    return {
        "audit_family": "phase2t_dynamic_repair_trace_smoke_postflight",
        "passed": passed,
        "ready_for_full_train": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": (
            "run_phase2t_full_nonsealed_training_only"
            if passed
            else "freeze_phase2t_smoke_failure_and_analyze_nonsealed_design"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "metrics": {
            "val_command_slot_accuracy": val_accuracy,
            "val_command_slot_count": val_count,
            "source_overlap_val_accuracy": source_overlap_accuracy,
            "model_minus_source_overlap_accuracy": model_minus_source_overlap,
            "duration_seconds": duration_seconds,
            "low_level_qwen_calls_target": summary.get("low_level_qwen_calls_target"),
            "use_pairwise_command_reranker": pairwise,
            "pairwise_encoded_candidates": pairwise_encoded_candidates,
            "command_candidate_encoder": candidate_encoder,
            "latent_fusion": latent_fusion,
        },
        "thresholds": {
            "min_val_command_slot_accuracy": min_val_command_slot_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "max_smoke_duration_seconds": max_smoke_duration_seconds,
        },
        "effective_split_hashes": data_hashes,
        "training_summary_head_hashes": summary_hashes,
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "data_health_json": str(Path(data_health_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "head_manifest_json": str(Path(head_manifest_json)) if head_manifest_json else None,
        },
    }


def build_phase2t_full_postflight(
    *,
    training_summary_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    head_manifest_json: str | Path | None = None,
    smoke_postflight_json: str | Path | None = None,
    min_val_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_train_examples: int = 96,
    min_val_examples: int = 64,
    max_full_duration_seconds: float = 7200.0,
) -> dict[str, Any]:
    base = build_phase2t_smoke_postflight(
        training_summary_json=training_summary_json,
        data_health_json=data_health_json,
        pretrain_gate_json=pretrain_gate_json,
        head_manifest_json=head_manifest_json,
        min_val_command_slot_accuracy=min_val_command_slot_accuracy,
        min_model_minus_source_overlap=min_model_minus_source_overlap,
        max_smoke_duration_seconds=max_full_duration_seconds,
    )
    summary = _read_json(training_summary_json)
    smoke_postflight = _read_json(smoke_postflight_json) if smoke_postflight_json else {}
    duration_seconds = base["metrics"].get("duration_seconds")
    checks = {
        key: value
        for key, value in _dict(base.get("checks")).items()
        if key
        not in {
            "smoke_duration_within_limit",
            "pairwise_disabled_for_phase2t_smoke",
            "pairwise_encoded_candidates_zero",
        }
    }
    checks.update(
        {
            "pairwise_disabled_for_phase2t_full": base["checks"].get(
                "pairwise_disabled_for_phase2t_smoke"
            )
            is True,
            "pairwise_encoded_candidates_zero_for_phase2t_full": base["checks"].get(
                "pairwise_encoded_candidates_zero"
            )
            is True,
            "full_train_examples_min": isinstance(summary.get("train_examples"), int)
            and summary.get("train_examples") >= min_train_examples,
            "full_val_examples_min": isinstance(summary.get("val_examples"), int)
            and summary.get("val_examples") >= min_val_examples,
            "full_duration_within_limit": isinstance(duration_seconds, float)
            and duration_seconds <= max_full_duration_seconds,
            "smoke_postflight_passed_when_provided": (
                not smoke_postflight or smoke_postflight.get("passed") is True
            ),
        }
    )
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_run_phase2t_holdout_controls_until_full_postflight_passes")
    if not checks["val_command_slot_accuracy_min"]:
        blocked_actions.append("do_not_claim_phase2t_full_val_gate")
    if not checks["model_minus_source_overlap_min"]:
        blocked_actions.append("do_not_claim_phase2t_full_mechanism_delta_from_source_overlap")
    if not checks["pairwise_disabled_for_phase2t_full"]:
        blocked_actions.append("do_not_mix_phase2t_full_with_pairwise_mechanism")
    if not checks["pairwise_encoded_candidates_zero_for_phase2t_full"]:
        blocked_actions.append("do_not_allow_phase2t_full_with_pairwise_candidate_scoring")
    if not checks["head_manifest_command_identity_margin_gate_passed"]:
        blocked_actions.append("do_not_package_phase2t_with_ambiguous_command_identity_latent")
    if not checks["data_and_pretrain_hashes_match"] or not checks[
        "head_manifest_hashes_match_data_health"
    ]:
        blocked_actions.append("do_not_package_phase2t_with_hash_mismatch")
    if not checks["full_train_examples_min"] or not checks["full_val_examples_min"]:
        blocked_actions.append("do_not_claim_phase2t_full_before_minimum_nonsealed_split_size")
    if not checks["smoke_postflight_passed_when_provided"]:
        blocked_actions.append("do_not_skip_phase2t_smoke_postflight_before_full_claim")
    return {
        "audit_family": "phase2t_dynamic_repair_trace_full_postflight",
        "passed": passed,
        "ready_for_full_train": False,
        "ready_for_holdout_controls": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": (
            "run_phase2t_holdout_control_diagnostics_before_package"
            if passed
            else "freeze_phase2t_full_failure_and_analyze_nonsealed_design"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "metrics": {
            **_dict(base.get("metrics")),
            "train_examples": summary.get("train_examples"),
            "val_examples": summary.get("val_examples"),
        },
        "thresholds": {
            "min_val_command_slot_accuracy": min_val_command_slot_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_train_examples": min_train_examples,
            "min_val_examples": min_val_examples,
            "max_full_duration_seconds": max_full_duration_seconds,
        },
        "effective_split_hashes": base.get("effective_split_hashes"),
        "training_summary_head_hashes": base.get("training_summary_head_hashes"),
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "data_health_json": str(Path(data_health_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "head_manifest_json": str(Path(head_manifest_json)) if head_manifest_json else None,
            "smoke_postflight_json": str(Path(smoke_postflight_json))
            if smoke_postflight_json
            else None,
        },
    }


def _diagnostic_source_accuracy(diagnostics: dict[str, Any], source: str) -> float | None:
    value = _dict(_dict(diagnostics.get("sources")).get(source)).get("accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def _diagnostic_source_total(diagnostics: dict[str, Any], source: str) -> int | None:
    value = _dict(_dict(diagnostics.get("sources")).get(source)).get("total")
    return int(value) if isinstance(value, int) else None


def build_phase2t_holdout_control_postflight(
    *,
    full_postflight_json: str | Path,
    holdout_diagnostics_json: str | Path,
    zero_nsi_diagnostics_json: str | Path,
    zero_nsi_manifest_json: str | Path | None = None,
    min_holdout_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_full_minus_zero_nsi: float = 0.15,
    min_full_minus_slot_head: float = 0.10,
    min_holdout_records: int = 96,
) -> dict[str, Any]:
    full_postflight = _read_json(full_postflight_json)
    holdout = _read_json(holdout_diagnostics_json)
    zero_nsi = _read_json(zero_nsi_diagnostics_json)
    zero_nsi_manifest = _read_json(zero_nsi_manifest_json) if zero_nsi_manifest_json else {}

    holdout_accuracy = _diagnostic_source_accuracy(holdout, "effective")
    holdout_total = _diagnostic_source_total(holdout, "effective")
    source_overlap_accuracy = _diagnostic_source_accuracy(holdout, "source_overlap_baseline")
    slot_head_accuracy = _diagnostic_source_accuracy(holdout, "slot_head")
    zero_nsi_accuracy = _diagnostic_source_accuracy(zero_nsi, "effective")
    zero_nsi_total = _diagnostic_source_total(zero_nsi, "effective")
    model_minus_source_overlap = (
        holdout_accuracy - source_overlap_accuracy
        if isinstance(holdout_accuracy, float) and isinstance(source_overlap_accuracy, float)
        else None
    )
    full_minus_zero_nsi = (
        holdout_accuracy - zero_nsi_accuracy
        if isinstance(holdout_accuracy, float) and isinstance(zero_nsi_accuracy, float)
        else None
    )
    full_minus_slot_head = (
        holdout_accuracy - slot_head_accuracy
        if isinstance(holdout_accuracy, float) and isinstance(slot_head_accuracy, float)
        else None
    )
    checks = {
        "full_postflight_passed": full_postflight.get("passed") is True,
        "holdout_diagnostics_nonsealed": holdout.get("sealed_data_used_for_training_or_tuning")
        is False,
        "zero_nsi_diagnostics_nonsealed": zero_nsi.get(
            "sealed_data_used_for_training_or_tuning"
        )
        is False,
        "holdout_records_min": isinstance(holdout_total, int)
        and holdout_total >= min_holdout_records,
        "zero_nsi_records_match_holdout": isinstance(zero_nsi_total, int)
        and zero_nsi_total == holdout_total,
        "holdout_val_gate_passed": isinstance(holdout_accuracy, float)
        and holdout_accuracy >= min_holdout_accuracy,
        "holdout_beats_source_overlap": isinstance(model_minus_source_overlap, float)
        and model_minus_source_overlap >= min_model_minus_source_overlap,
        "holdout_beats_zero_nsi_control": isinstance(full_minus_zero_nsi, float)
        and full_minus_zero_nsi >= min_full_minus_zero_nsi,
        "holdout_beats_raw_slot_head_control": isinstance(full_minus_slot_head, float)
        and full_minus_slot_head >= min_full_minus_slot_head,
        "zero_nsi_manifest_erased_nsi_reference": (
            not zero_nsi_manifest or zero_nsi_manifest.get("nsi_reference_erased") is True
        ),
        "zero_nsi_manifest_nonsealed": (
            not zero_nsi_manifest
            or zero_nsi_manifest.get("sealed_v3_used_for_training_or_tuning") is False
        ),
        "pairwise_disabled_in_holdout_diagnostics": holdout.get(
            "use_pairwise_command_reranker"
        )
        is False,
        "features_only_candidate_encoder_in_holdout": holdout.get("command_candidate_encoder")
        == "features_only",
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_package_until_phase2t_holdout_controls_pass")
    if not checks["holdout_val_gate_passed"]:
        blocked_actions.append("do_not_claim_phase2t_holdout_val_gate")
    if not checks["holdout_beats_source_overlap"]:
        blocked_actions.append("do_not_claim_phase2t_holdout_delta_vs_source_overlap")
    if not checks["holdout_beats_zero_nsi_control"]:
        blocked_actions.append("do_not_claim_phase2t_nsi_latent_necessity")
    if not checks["holdout_beats_raw_slot_head_control"]:
        blocked_actions.append("do_not_claim_phase2t_candidate_identity_gain_over_raw_slot_head")
    return {
        "audit_family": "phase2t_dynamic_repair_trace_holdout_control_postflight",
        "passed": passed,
        "ready_for_additional_controls": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": (
            "run_phase2t_package_gate_or_additional_controls_before_package"
            if passed
            else "freeze_phase2t_holdout_control_failure_and_analyze_nonsealed_design"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "metrics": {
            "holdout_effective_accuracy": holdout_accuracy,
            "holdout_effective_total": holdout_total,
            "source_overlap_holdout_accuracy": source_overlap_accuracy,
            "slot_head_holdout_accuracy": slot_head_accuracy,
            "zero_nsi_holdout_accuracy": zero_nsi_accuracy,
            "zero_nsi_holdout_total": zero_nsi_total,
            "model_minus_source_overlap_holdout_accuracy": model_minus_source_overlap,
            "full_minus_zero_nsi_holdout_accuracy": full_minus_zero_nsi,
            "full_minus_raw_slot_head_holdout_accuracy": full_minus_slot_head,
        },
        "thresholds": {
            "min_holdout_accuracy": min_holdout_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_full_minus_zero_nsi": min_full_minus_zero_nsi,
            "min_full_minus_slot_head": min_full_minus_slot_head,
            "min_holdout_records": min_holdout_records,
        },
        "inputs": {
            "full_postflight_json": str(Path(full_postflight_json)),
            "holdout_diagnostics_json": str(Path(holdout_diagnostics_json)),
            "zero_nsi_diagnostics_json": str(Path(zero_nsi_diagnostics_json)),
            "zero_nsi_manifest_json": str(Path(zero_nsi_manifest_json))
            if zero_nsi_manifest_json
            else None,
        },
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_phase2t_multiseed_summary(
    *,
    holdout_postflight_jsons: list[str | Path],
    min_seed_count: int = 3,
    min_holdout_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_full_minus_zero_nsi: float = 0.15,
) -> dict[str, Any]:
    reports = [_read_json(path) for path in holdout_postflight_jsons]
    metrics_rows: list[dict[str, Any]] = []
    for path, report in zip(holdout_postflight_jsons, reports):
        metrics = _dict(report.get("metrics"))
        metrics_rows.append(
            {
                "path": str(Path(path)),
                "passed": report.get("passed") is True,
                "holdout_effective_accuracy": metrics.get("holdout_effective_accuracy"),
                "model_minus_source_overlap_holdout_accuracy": metrics.get(
                    "model_minus_source_overlap_holdout_accuracy"
                ),
                "full_minus_zero_nsi_holdout_accuracy": metrics.get(
                    "full_minus_zero_nsi_holdout_accuracy"
                ),
                "full_minus_raw_slot_head_holdout_accuracy": metrics.get(
                    "full_minus_raw_slot_head_holdout_accuracy"
                ),
            }
        )
    holdout_accs = [
        float(row["holdout_effective_accuracy"])
        for row in metrics_rows
        if isinstance(row.get("holdout_effective_accuracy"), (int, float))
    ]
    source_deltas = [
        float(row["model_minus_source_overlap_holdout_accuracy"])
        for row in metrics_rows
        if isinstance(row.get("model_minus_source_overlap_holdout_accuracy"), (int, float))
    ]
    zero_nsi_deltas = [
        float(row["full_minus_zero_nsi_holdout_accuracy"])
        for row in metrics_rows
        if isinstance(row.get("full_minus_zero_nsi_holdout_accuracy"), (int, float))
    ]
    checks = {
        "seed_count_min": len(reports) >= min_seed_count,
        "all_holdout_postflights_passed": bool(reports)
        and all(report.get("passed") is True for report in reports),
        "all_holdout_accuracy_present": len(holdout_accs) == len(reports),
        "all_source_delta_present": len(source_deltas) == len(reports),
        "all_zero_nsi_delta_present": len(zero_nsi_deltas) == len(reports),
        "min_holdout_accuracy_gate": bool(holdout_accs)
        and min(holdout_accs) >= min_holdout_accuracy,
        "min_model_minus_source_overlap_gate": bool(source_deltas)
        and min(source_deltas) >= min_model_minus_source_overlap,
        "min_full_minus_zero_nsi_gate": bool(zero_nsi_deltas)
        and min(zero_nsi_deltas) >= min_full_minus_zero_nsi,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_claim_phase2t_multiseed_reproducibility")
    if not checks["seed_count_min"]:
        blocked_actions.append("do_not_claim_multiseed_with_too_few_seeds")
    if not checks["min_holdout_accuracy_gate"]:
        blocked_actions.append("do_not_claim_phase2t_multiseed_holdout_val_gate")
    if not checks["min_model_minus_source_overlap_gate"]:
        blocked_actions.append("do_not_claim_phase2t_multiseed_source_overlap_delta")
    if not checks["min_full_minus_zero_nsi_gate"]:
        blocked_actions.append("do_not_claim_phase2t_multiseed_nsi_latent_delta")
    return {
        "audit_family": "phase2t_dynamic_repair_trace_multiseed_summary",
        "passed": passed,
        "ready_for_multimodel_reproduction": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": (
            "run_phase2t_cross_model_reproduction_before_package"
            if passed
            else "freeze_phase2t_multiseed_failure_and_analyze_nonsealed_design"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "metrics": {
            "seed_count": len(reports),
            "mean_holdout_effective_accuracy": _mean(holdout_accs),
            "min_holdout_effective_accuracy": min(holdout_accs) if holdout_accs else None,
            "mean_model_minus_source_overlap_holdout_accuracy": _mean(source_deltas),
            "min_model_minus_source_overlap_holdout_accuracy": (
                min(source_deltas) if source_deltas else None
            ),
            "mean_full_minus_zero_nsi_holdout_accuracy": _mean(zero_nsi_deltas),
            "min_full_minus_zero_nsi_holdout_accuracy": (
                min(zero_nsi_deltas) if zero_nsi_deltas else None
            ),
        },
        "per_seed": metrics_rows,
        "thresholds": {
            "min_seed_count": min_seed_count,
            "min_holdout_accuracy": min_holdout_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_full_minus_zero_nsi": min_full_minus_zero_nsi,
        },
        "inputs": {"holdout_postflight_jsons": [str(Path(path)) for path in holdout_postflight_jsons]},
    }


def build_phase2t_cross_model_summary(
    *,
    multiseed_summary_jsons: list[str | Path],
    min_model_count: int = 2,
    min_seed_count_per_model: int = 3,
    min_holdout_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_full_minus_zero_nsi: float = 0.15,
) -> dict[str, Any]:
    reports = [_read_json(path) for path in multiseed_summary_jsons]
    model_rows: list[dict[str, Any]] = []
    for path, report in zip(multiseed_summary_jsons, reports):
        metrics = _dict(report.get("metrics"))
        model_rows.append(
            {
                "path": str(Path(path)),
                "passed": report.get("passed") is True,
                "seed_count": metrics.get("seed_count"),
                "min_holdout_effective_accuracy": metrics.get(
                    "min_holdout_effective_accuracy"
                ),
                "min_model_minus_source_overlap_holdout_accuracy": metrics.get(
                    "min_model_minus_source_overlap_holdout_accuracy"
                ),
                "min_full_minus_zero_nsi_holdout_accuracy": metrics.get(
                    "min_full_minus_zero_nsi_holdout_accuracy"
                ),
            }
        )
    seed_counts = [
        int(row["seed_count"])
        for row in model_rows
        if isinstance(row.get("seed_count"), int)
    ]
    min_holdout_accs = [
        float(row["min_holdout_effective_accuracy"])
        for row in model_rows
        if isinstance(row.get("min_holdout_effective_accuracy"), (int, float))
    ]
    source_deltas = [
        float(row["min_model_minus_source_overlap_holdout_accuracy"])
        for row in model_rows
        if isinstance(row.get("min_model_minus_source_overlap_holdout_accuracy"), (int, float))
    ]
    zero_nsi_deltas = [
        float(row["min_full_minus_zero_nsi_holdout_accuracy"])
        for row in model_rows
        if isinstance(row.get("min_full_minus_zero_nsi_holdout_accuracy"), (int, float))
    ]
    checks = {
        "model_count_min": len(reports) >= min_model_count,
        "all_multiseed_summaries_passed": bool(reports)
        and all(report.get("passed") is True for report in reports),
        "seed_count_per_model_min": bool(seed_counts)
        and len(seed_counts) == len(reports)
        and min(seed_counts) >= min_seed_count_per_model,
        "all_model_holdout_accuracy_present": len(min_holdout_accs) == len(reports),
        "all_model_source_delta_present": len(source_deltas) == len(reports),
        "all_model_zero_nsi_delta_present": len(zero_nsi_deltas) == len(reports),
        "cross_model_min_holdout_accuracy_gate": bool(min_holdout_accs)
        and min(min_holdout_accs) >= min_holdout_accuracy,
        "cross_model_min_source_delta_gate": bool(source_deltas)
        and min(source_deltas) >= min_model_minus_source_overlap,
        "cross_model_min_zero_nsi_delta_gate": bool(zero_nsi_deltas)
        and min(zero_nsi_deltas) >= min_full_minus_zero_nsi,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_claim_phase2t_cross_model_reproducibility")
    if not checks["model_count_min"]:
        blocked_actions.append("do_not_claim_cross_model_with_too_few_models")
    if not checks["seed_count_per_model_min"]:
        blocked_actions.append("do_not_claim_cross_model_without_per_model_seed_reproduction")
    if not checks["cross_model_min_source_delta_gate"]:
        blocked_actions.append("do_not_claim_phase2t_cross_model_source_overlap_delta")
    if not checks["cross_model_min_zero_nsi_delta_gate"]:
        blocked_actions.append("do_not_claim_phase2t_cross_model_nsi_latent_delta")
    return {
        "audit_family": "phase2t_dynamic_repair_trace_cross_model_summary",
        "passed": passed,
        "ready_for_package_gate_design": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": (
            "design_phase2t_package_gate_or_expand_model_family_before_package"
            if passed
            else "freeze_phase2t_cross_model_failure_and_analyze_nonsealed_design"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "metrics": {
            "model_count": len(reports),
            "min_seed_count_per_model": min(seed_counts) if seed_counts else None,
            "min_holdout_effective_accuracy_across_models": (
                min(min_holdout_accs) if min_holdout_accs else None
            ),
            "min_source_overlap_delta_across_models": (
                min(source_deltas) if source_deltas else None
            ),
            "min_zero_nsi_delta_across_models": (
                min(zero_nsi_deltas) if zero_nsi_deltas else None
            ),
        },
        "per_model": model_rows,
        "thresholds": {
            "min_model_count": min_model_count,
            "min_seed_count_per_model": min_seed_count_per_model,
            "min_holdout_accuracy": min_holdout_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_full_minus_zero_nsi": min_full_minus_zero_nsi,
        },
        "inputs": {
            "multiseed_summary_jsons": [str(Path(path)) for path in multiseed_summary_jsons]
        },
    }


def build_phase2t_package_gate(
    *,
    cross_model_summary_json: str | Path,
    canonical_training_summary_json: str | Path,
    canonical_full_postflight_json: str | Path,
    canonical_holdout_control_postflight_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    head_manifest_json: str | Path,
    adapter_dir: str | Path,
    min_cross_model_count: int = 2,
    min_seed_count_per_model: int = 3,
    min_holdout_accuracy: float = 0.85,
    min_source_overlap_delta: float = 0.15,
    min_zero_nsi_delta: float = 0.15,
) -> dict[str, Any]:
    cross_model = _read_json(cross_model_summary_json)
    summary = _read_json(canonical_training_summary_json)
    full_postflight = _read_json(canonical_full_postflight_json)
    holdout_postflight = _read_json(canonical_holdout_control_postflight_json)
    data_health = _read_json(data_health_json)
    pretrain_gate = _read_json(pretrain_gate_json)
    head_manifest = _read_json(head_manifest_json)
    cross_metrics = _dict(cross_model.get("metrics"))
    holdout_metrics = _dict(holdout_postflight.get("metrics"))
    full_metrics = _dict(full_postflight.get("metrics"))
    data_hashes = _dict(data_health.get("effective_split_hashes"))
    pretrain_hashes = _dict(pretrain_gate.get("effective_split_hashes"))
    head_hashes = _dict(head_manifest.get("effective_split_hashes"))
    summary_hashes = _dict(summary.get("effective_split_hashes"))
    head_config = _dict(summary.get("head_config"))
    config = _dict(summary.get("config"))
    adapter_path = Path(adapter_dir)

    checks = {
        "cross_model_summary_passed": cross_model.get("passed") is True,
        "cross_model_count_min": int(cross_metrics.get("model_count") or 0)
        >= min_cross_model_count,
        "cross_model_seed_count_min": int(cross_metrics.get("min_seed_count_per_model") or 0)
        >= min_seed_count_per_model,
        "cross_model_holdout_accuracy_gate": float(
            cross_metrics.get("min_holdout_effective_accuracy_across_models") or 0.0
        )
        >= min_holdout_accuracy,
        "cross_model_source_delta_gate": float(
            cross_metrics.get("min_source_overlap_delta_across_models") or 0.0
        )
        >= min_source_overlap_delta,
        "cross_model_zero_nsi_delta_gate": float(
            cross_metrics.get("min_zero_nsi_delta_across_models") or 0.0
        )
        >= min_zero_nsi_delta,
        "canonical_full_postflight_passed": full_postflight.get("passed") is True,
        "canonical_holdout_control_passed": holdout_postflight.get("passed") is True,
        "canonical_holdout_accuracy_gate": float(
            holdout_metrics.get("holdout_effective_accuracy") or 0.0
        )
        >= min_holdout_accuracy,
        "canonical_source_delta_gate": float(
            holdout_metrics.get("model_minus_source_overlap_holdout_accuracy") or 0.0
        )
        >= min_source_overlap_delta,
        "canonical_zero_nsi_delta_gate": float(
            holdout_metrics.get("full_minus_zero_nsi_holdout_accuracy") or 0.0
        )
        >= min_zero_nsi_delta,
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain_gate.get("passed") is True,
        "head_manifest_source_gates_passed": head_manifest.get("source_data_health_passed")
        is True
        and head_manifest.get("source_pretrain_gate_passed") is True,
        "head_manifest_command_identity_margin_gate_passed": head_manifest.get(
            "command_identity_margin_gate_passed"
        )
        is True,
        "data_pretrain_head_hashes_match": bool(data_hashes)
        and data_hashes == pretrain_hashes
        and data_hashes == head_hashes,
        "training_summary_head_hashes_present": all(
            summary_hashes.get(key) for key in ("phase2c_head_train", "phase2c_head_val")
        ),
        "adapter_dir_exists": adapter_path.exists() and adapter_path.is_dir(),
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "pairwise_disabled_for_phase2t_package": summary.get("use_pairwise_command_reranker")
        is False
        and head_config.get("use_pairwise_command_reranker") is False,
        "features_only_candidate_encoder": (
            summary.get("command_candidate_encoder")
            or config.get("command_candidate_encoder")
            or head_config.get("command_candidate_encoder")
        )
        == "features_only",
        "additive_latent_fusion": (
            config.get("latent_fusion") == "additive"
            and head_config.get("latent_fusion") == "additive"
        ),
        "full_postflight_not_sealed_ready": full_postflight.get("ready_for_sealed_eval")
        is False,
        "holdout_postflight_not_sealed_ready": holdout_postflight.get("ready_for_sealed_eval")
        is False,
        "cross_model_summary_not_sealed_ready": cross_model.get("ready_for_sealed_eval")
        is False,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_package_phase2t_until_package_gate_passes")
    if not checks["cross_model_summary_passed"]:
        blocked_actions.append("do_not_package_without_phase2t_cross_model_summary")
    if not checks["canonical_holdout_control_passed"]:
        blocked_actions.append("do_not_package_without_phase2t_holdout_control_gate")
    if not checks["data_pretrain_head_hashes_match"]:
        blocked_actions.append("do_not_package_phase2t_with_split_hash_mismatch")
    if not checks["adapter_dir_exists"]:
        blocked_actions.append("do_not_package_missing_phase2t_adapter")
    if not checks["pairwise_disabled_for_phase2t_package"]:
        blocked_actions.append("do_not_package_phase2t_with_pairwise_confounded_mechanism")
    if not checks["cross_model_zero_nsi_delta_gate"] or not checks[
        "canonical_zero_nsi_delta_gate"
    ]:
        blocked_actions.append("do_not_package_without_phase2t_nsi_latent_delta")
    supported_claims = [
        "phase2t_nonsealed_public_repair_loop_command_identity_delta_supported",
        "phase2t_nsi_latent_needed_for_holdout_command_selection_supported",
        "phase2t_qwen2_5_3b_7b_three_seed_reproduction_supported",
    ]
    unsupported_claims = [
        "sealed_cross_model_transfer_not_established_by_package_gate",
        "production_autonomy_not_established",
        "open_ended_debugging_generalization_not_established",
        "epoch_making_architecture_claim_not_established",
    ]
    return {
        "audit_family": "phase2t_dynamic_repair_trace_package_gate",
        "passed": passed,
        "ready_for_package": passed,
        "ready_for_sealed_eval": False,
        "allowed_next_action": (
            "run_phase2t_package_only_then_sealed_eval_gate"
            if passed
            else "revise_phase2t_before_package"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "metrics": {
            "cross_model_count": cross_metrics.get("model_count"),
            "min_seed_count_per_model": cross_metrics.get("min_seed_count_per_model"),
            "cross_model_min_holdout_accuracy": cross_metrics.get(
                "min_holdout_effective_accuracy_across_models"
            ),
            "cross_model_min_source_overlap_delta": cross_metrics.get(
                "min_source_overlap_delta_across_models"
            ),
            "cross_model_min_zero_nsi_delta": cross_metrics.get(
                "min_zero_nsi_delta_across_models"
            ),
            "canonical_val_accuracy": full_metrics.get("val_command_slot_accuracy"),
            "canonical_holdout_accuracy": holdout_metrics.get("holdout_effective_accuracy"),
            "canonical_source_overlap_delta": holdout_metrics.get(
                "model_minus_source_overlap_holdout_accuracy"
            ),
            "canonical_zero_nsi_delta": holdout_metrics.get(
                "full_minus_zero_nsi_holdout_accuracy"
            ),
        },
        "thresholds": {
            "min_cross_model_count": min_cross_model_count,
            "min_seed_count_per_model": min_seed_count_per_model,
            "min_holdout_accuracy": min_holdout_accuracy,
            "min_source_overlap_delta": min_source_overlap_delta,
            "min_zero_nsi_delta": min_zero_nsi_delta,
        },
        "supported_claims": supported_claims if passed else [],
        "unsupported_claims": unsupported_claims,
        "inputs": {
            "cross_model_summary_json": str(Path(cross_model_summary_json)),
            "canonical_training_summary_json": str(Path(canonical_training_summary_json)),
            "canonical_full_postflight_json": str(Path(canonical_full_postflight_json)),
            "canonical_holdout_control_postflight_json": str(
                Path(canonical_holdout_control_postflight_json)
            ),
            "data_health_json": str(Path(data_health_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "head_manifest_json": str(Path(head_manifest_json)),
            "adapter_dir": str(adapter_path),
        },
    }


def _package_manifest(path: str | Path) -> dict[str, Any]:
    package_path = Path(path)
    manifest_path = package_path if package_path.is_file() else package_path / "native_nervous_package.json"
    if not manifest_path.exists():
        return {}
    return _read_json(manifest_path)


def build_phase2t_postpackage_gate(
    *,
    package_gate_json: str | Path,
    full_package_path: str | Path,
    no_nsi_package_path: str | Path,
    native_head_only_package_path: str | Path,
    continuation_only_package_path: str | Path,
) -> dict[str, Any]:
    package_gate = _read_json(package_gate_json)
    full_manifest = _package_manifest(full_package_path)
    no_nsi_manifest = _package_manifest(no_nsi_package_path)
    native_head_only_manifest = _package_manifest(native_head_only_package_path)
    continuation_only_manifest = _package_manifest(continuation_only_package_path)
    full_label = str(full_manifest.get("policy_label") or "")
    expected_control_labels = {
        "no_nsi": full_label + "_no_nsi_latent",
        "native_head_only": full_label + "_native_head_only",
        "continuation_only": full_label + "_continuation_only",
    }
    checks = {
        "package_gate_passed": package_gate.get("passed") is True,
        "package_gate_ready_for_package": package_gate.get("ready_for_package") is True,
        "package_gate_not_sealed_ready": package_gate.get("ready_for_sealed_eval") is False,
        "full_package_manifest_exists": bool(full_manifest),
        "no_nsi_package_manifest_exists": bool(no_nsi_manifest),
        "native_head_only_package_manifest_exists": bool(native_head_only_manifest),
        "continuation_only_package_manifest_exists": bool(continuation_only_manifest),
        "full_package_family_valid": full_manifest.get("package_family")
        == "phase2d_native_nervous_package",
        "full_package_explicit_heads": full_manifest.get("motor_output")
        == "explicit_heads_runtime_serialization",
        "full_package_no_json_target": full_manifest.get("json_text_target") is False,
        "full_package_nsi_enabled": full_manifest.get("zero_nsi_latent") is False,
        "full_package_native_heads_enabled": full_manifest.get("native_head_calls_enabled")
        is True,
        "full_package_continuation_enabled": full_manifest.get("continuation_cache_enabled")
        is True,
        "no_nsi_control_zeroes_latent": no_nsi_manifest.get("zero_nsi_latent") is True,
        "no_nsi_control_keeps_native_heads": no_nsi_manifest.get("native_head_calls_enabled")
        is True,
        "native_head_only_control_disables_continuation": native_head_only_manifest.get(
            "continuation_cache_enabled"
        )
        is False,
        "native_head_only_control_keeps_native_heads": native_head_only_manifest.get(
            "native_head_calls_enabled"
        )
        is True,
        "continuation_only_control_disables_native_heads": continuation_only_manifest.get(
            "native_head_calls_enabled"
        )
        is False,
        "continuation_only_control_keeps_continuation": continuation_only_manifest.get(
            "continuation_cache_enabled"
        )
        is True,
        "control_labels_derive_from_full_label": (
            no_nsi_manifest.get("policy_label") == expected_control_labels["no_nsi"]
            and native_head_only_manifest.get("policy_label")
            == expected_control_labels["native_head_only"]
            and continuation_only_manifest.get("policy_label")
            == expected_control_labels["continuation_only"]
        ),
        "all_packages_reference_same_backbone_and_adapter": bool(full_manifest)
        and all(
            manifest.get("base_model_name") == full_manifest.get("base_model_name")
            and manifest.get("native_head_path") == full_manifest.get("native_head_path")
            and manifest.get("low_level_checkpoint_path")
            == full_manifest.get("low_level_checkpoint_path")
            for manifest in [
                no_nsi_manifest,
                native_head_only_manifest,
                continuation_only_manifest,
            ]
        ),
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_run_phase2t_sealed_eval_until_postpackage_gate_passes")
    if not checks["package_gate_passed"]:
        blocked_actions.append("do_not_bypass_phase2t_package_gate")
    if not checks["control_labels_derive_from_full_label"]:
        blocked_actions.append("do_not_run_sealed_eval_with_mismatched_control_packages")
    if not checks["all_packages_reference_same_backbone_and_adapter"]:
        blocked_actions.append("do_not_run_sealed_eval_with_drifted_package_artifacts")
    return {
        "audit_family": "phase2t_dynamic_repair_trace_postpackage_gate",
        "passed": passed,
        "ready_for_sealed_eval": passed,
        "ready_for_claim_upgrade": False,
        "allowed_next_action": (
            "run_phase2t_sealed_eval_only_no_training_feedback"
            if passed
            else "rebuild_phase2t_packages_before_sealed_eval"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "package_paths": {
            "full": str(Path(full_package_path)),
            "no_nsi": str(Path(no_nsi_package_path)),
            "native_head_only": str(Path(native_head_only_package_path)),
            "continuation_only": str(Path(continuation_only_package_path)),
        },
        "unsupported_claims": [
            "sealed_cross_model_transfer_not_established_until_sealed_eval_passes",
            "production_autonomy_not_established",
            "open_ended_debugging_generalization_not_established",
            "epoch_making_architecture_claim_not_established",
        ],
        "inputs": {"package_gate_json": str(Path(package_gate_json))},
    }


def _aggregate_metric(payload: dict[str, Any], name: str) -> float | None:
    metric = _dict(_dict(_dict(payload.get("metrics")).get("aggregate")).get(name))
    value = metric.get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _eval_run_rows(payload: dict[str, Any], filename: str) -> list[dict[str, Any]]:
    run_path = payload.get("run_path")
    if not run_path:
        return []
    rows, exists = _read_jsonl(Path(str(run_path)) / filename)
    return rows if exists else []


def _control_zero_classification(
    *,
    name: str,
    payload: dict[str, Any],
    trace_summary: dict[str, Any],
) -> dict[str, Any]:
    completion = _aggregate_metric(payload, "task_completion_rate")
    policy = _dict(payload.get("policy"))
    if completion is None:
        return {
            "classification": "not_evaluable_for_control",
            "reason": "missing task_completion_rate metric",
            "evidence_use": "exclude from performance delta until evaluator output is fixed",
        }
    if completion > 0.0:
        return {
            "classification": "nonzero_control",
            "reason": "control achieved nonzero task completion",
            "evidence_use": "usable as measured baseline delta",
        }
    if int(trace_summary.get("parse_failures") or 0) > 0:
        return {
            "classification": "suspicious_zero_requires_redesign",
            "reason": "zero may be caused by parser failures rather than model/control capability",
            "evidence_use": "do not use for claim support until parser failure is eliminated",
        }
    if int(trace_summary.get("hallucinated_actions") or 0) > 0:
        return {
            "classification": "valid_zero_failure",
            "reason": "control produced hallucinated or non-allowlisted actions",
            "evidence_use": "usable only with hallucination caveat, not as mechanism-only evidence",
        }
    if policy.get("zero_nsi_latent") is True:
        return {
            "classification": "expected_zero_due_to_missing_capability",
            "reason": "control explicitly removes NSI latent contribution required by this sealed task",
            "evidence_use": "mechanism-ablation evidence only; not standalone generic baseline weakness",
        }
    if policy.get("native_head_calls_enabled") is False:
        return {
            "classification": "expected_zero_due_to_missing_capability",
            "reason": "control explicitly disables native head calls required for command selection",
            "evidence_use": "mechanism-ablation evidence only; not standalone generic baseline weakness",
        }
    if policy.get("continuation_cache_enabled") is False:
        return {
            "classification": "expected_zero_due_to_missing_capability",
            "reason": "control disables continuation cache on a continuation-sensitive closed-loop task",
            "evidence_use": "mechanism-ablation evidence only; not standalone generic baseline weakness",
        }
    if policy.get("policy_family") == "huggingface_json":
        return {
            "classification": "valid_zero_failure",
            "reason": "text baseline produced valid JSON actions but chose the wrong closed-loop action type/order",
            "evidence_use": "valid text-loop baseline failure; stronger claims still need graded sanity controls",
        }
    return {
        "classification": "suspicious_zero_requires_redesign",
        "reason": f"zero result for {name} has no recognized capability-removal or text-baseline explanation",
        "evidence_use": "do not use for claim support until classified",
    }


def _summarize_eval_for_zero_audit(name: str, eval_json: str | Path) -> dict[str, Any]:
    payload = _read_json(eval_json)
    trace_rows = _eval_run_rows(payload, "trace_rows.jsonl")
    episode_rows = _eval_run_rows(payload, "episode_results.jsonl")
    action_types = Counter(
        str(_dict(row.get("action")).get("type") or "missing") for row in trace_rows
    )
    oracle_types = Counter(
        str(_dict(row.get("oracle_action")).get("type") or "missing") for row in trace_rows
    )
    mismatch_pairs = Counter(
        f"{_dict(row.get('action')).get('type') or 'missing'}->{_dict(row.get('oracle_action')).get('type') or 'missing'}"
        for row in trace_rows
        if row.get("correct") is not True
    )
    first_step_mismatch_pairs = Counter(
        f"{_dict(row.get('action')).get('type') or 'missing'}->{_dict(row.get('oracle_action')).get('type') or 'missing'}"
        for row in trace_rows
        if row.get("step_index") == 0 and row.get("correct") is not True
    )
    parse_failures = sum(
        int(row.get("parse_failures") or 0)
        for row in episode_rows
        if isinstance(row.get("parse_failures"), int)
    )
    hallucinated_actions = sum(1 for row in trace_rows if row.get("hallucinated") is True)
    summary = {
        "name": name,
        "eval_json": str(Path(eval_json)),
        "policy_family": _dict(payload.get("policy")).get("policy_family"),
        "policy_label": _dict(payload.get("policy")).get("policy_label"),
        "task_completion_rate": _aggregate_metric(payload, "task_completion_rate"),
        "oracle_step_accuracy": _aggregate_metric(payload, "oracle_step_accuracy"),
        "command_decision_accuracy": _aggregate_metric(payload, "command_decision_accuracy"),
        "read_file_decision_accuracy": _aggregate_metric(payload, "read_file_decision_accuracy"),
        "state_hallucination_rate": _aggregate_metric(payload, "state_hallucination_rate"),
        "model_calls": _aggregate_metric(payload, "model_calls"),
        "trace_rows": len(trace_rows),
        "episode_rows": len(episode_rows),
        "parse_failures": parse_failures,
        "hallucinated_actions": hallucinated_actions,
        "action_type_counts": dict(sorted(action_types.items())),
        "oracle_action_type_counts": dict(sorted(oracle_types.items())),
        "top_mismatch_pairs": dict(mismatch_pairs.most_common(8)),
        "top_first_step_mismatch_pairs": dict(first_step_mismatch_pairs.most_common(8)),
        "qwen_called_rows": sum(1 for row in trace_rows if row.get("qwen_called") is True),
        "cache_hit_rows": sum(1 for row in trace_rows if row.get("cache_hit") is True),
        "cache_reset_reasons": dict(
            Counter(
                str(row.get("cache_reset_reason"))
                for row in trace_rows
                if row.get("cache_reset_reason")
            )
        ),
    }
    if name != "full":
        summary["zero_classification"] = _control_zero_classification(
            name=name,
            payload=payload,
            trace_summary=summary,
        )
    return summary


def build_phase2t_sealed_zero_baseline_audit(
    *,
    external_gate_json: str | Path,
    full_eval_json: str | Path,
    prompt_eval_json: str | Path,
    react_eval_json: str | Path,
    no_nsi_eval_json: str | Path,
    native_head_only_eval_json: str | Path,
    continuation_only_eval_json: str | Path,
) -> dict[str, Any]:
    external_gate = _read_json(external_gate_json)
    evals = {
        "full": full_eval_json,
        "prompt_only": prompt_eval_json,
        "react": react_eval_json,
        "no_nsi": no_nsi_eval_json,
        "native_head_only": native_head_only_eval_json,
        "continuation_only": continuation_only_eval_json,
    }
    summaries = {
        name: _summarize_eval_for_zero_audit(name, path) for name, path in evals.items()
    }
    control_summaries = {
        name: payload for name, payload in summaries.items() if name != "full"
    }
    classifications = {
        name: _dict(payload.get("zero_classification")).get("classification")
        for name, payload in control_summaries.items()
    }
    zero_controls = [
        name
        for name, payload in control_summaries.items()
        if payload.get("task_completion_rate") == 0.0
    ]
    suspicious = [
        name
        for name, classification in classifications.items()
        if classification == "suspicious_zero_requires_redesign"
    ]
    all_controls_zero = len(zero_controls) == len(control_summaries)
    checks = {
        "external_gate_passed": external_gate.get("passed") is True,
        "full_nonzero": (summaries["full"].get("task_completion_rate") or 0.0) > 0.0,
        "all_zero_controls_classified": all(
            isinstance(_dict(payload.get("zero_classification")).get("classification"), str)
            for payload in control_summaries.values()
            if payload.get("task_completion_rate") == 0.0
        ),
        "no_suspicious_unexplained_zero": not suspicious,
        "zero_results_not_used_as_unqualified_performance_proof": True,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if suspicious:
        blocked_actions.append("do_not_claim_phase2t_sealed_delta_until_zero_roots_are_explained")
    if all_controls_zero:
        blocked_actions.append("do_not_upgrade_to_strong_architecture_claim_from_all_zero_controls")
        blocked_actions.append("add_graded_sanity_subset_with_nonzero_baseline_feasibility")
    return {
        "audit_family": "phase2t_sealed_zero_baseline_root_cause_audit",
        "passed": passed,
        "ready_for_bounded_sealed_claim": passed and external_gate.get("passed") is True,
        "ready_for_strong_architecture_claim": False,
        "checks": checks,
        "all_controls_zero": all_controls_zero,
        "zero_controls": zero_controls,
        "suspicious_zero_controls": suspicious,
        "classification_counts": dict(Counter(classifications.values())),
        "interpretation": {
            "bounded_claim": (
                "sealed v3 supports a bounded semantic-required package transfer result"
                if passed and external_gate.get("passed") is True
                else "sealed v3 result requires additional audit before claim use"
            ),
            "zero_baseline_caveat": (
                "Because every control baseline is zero, deltas are not sufficient by themselves "
                "for a broad architecture claim; they must be paired with baseline-feasibility "
                "sanity subsets and nonzero controls."
            ),
            "forbidden_inference": (
                "Do not infer production autonomy, open-ended debugging generalization, or an "
                "epoch-making architecture solely from these zero-control sealed results."
            ),
        },
        "blocked_actions": sorted(set(blocked_actions)),
        "eval_summaries": summaries,
        "inputs": {
            "external_gate_json": str(Path(external_gate_json)),
            "full_eval_json": str(Path(full_eval_json)),
            "prompt_eval_json": str(Path(prompt_eval_json)),
            "react_eval_json": str(Path(react_eval_json)),
            "no_nsi_eval_json": str(Path(no_nsi_eval_json)),
            "native_head_only_eval_json": str(Path(native_head_only_eval_json)),
            "continuation_only_eval_json": str(Path(continuation_only_eval_json)),
        },
    }


def _source_accuracy(payload: dict[str, Any], source_name: str) -> float | None:
    source = _dict(_dict(payload.get("sources")).get(source_name))
    value = source.get("accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def _source_total(payload: dict[str, Any], source_name: str) -> int | None:
    source = _dict(_dict(payload.get("sources")).get(source_name))
    value = source.get("total")
    return int(value) if isinstance(value, int) else None


def build_phase2t_baseline_feasibility_sanity_audit(
    *,
    sealed_zero_audit_json: str | Path,
    holdout_control_postflight_json: str | Path,
    holdout_diagnostics_json: str | Path,
    zero_nsi_diagnostics_json: str | Path,
    data_health_json: str | Path,
    min_full_holdout_accuracy: float = 0.85,
    min_nonzero_baseline_accuracy: float = 0.01,
    max_source_overlap_accuracy: float = 0.85,
    min_full_minus_source_overlap: float = 0.15,
    min_full_minus_zero_nsi: float = 0.15,
    min_full_minus_slot_head: float = 0.10,
) -> dict[str, Any]:
    sealed_zero = _read_json(sealed_zero_audit_json)
    holdout_postflight = _read_json(holdout_control_postflight_json)
    holdout_diag = _read_json(holdout_diagnostics_json)
    zero_nsi_diag = _read_json(zero_nsi_diagnostics_json)
    data_health = _read_json(data_health_json)
    post_metrics = _dict(holdout_postflight.get("metrics"))
    full_holdout = post_metrics.get("holdout_effective_accuracy")
    source_overlap = _source_accuracy(holdout_diag, "source_overlap_baseline")
    slot_head = _source_accuracy(holdout_diag, "slot_head")
    zero_nsi = _source_accuracy(zero_nsi_diag, "effective")
    totals = {
        "holdout": post_metrics.get("holdout_effective_total"),
        "source_overlap": _source_total(holdout_diag, "source_overlap_baseline"),
        "slot_head": _source_total(holdout_diag, "slot_head"),
        "zero_nsi": _source_total(zero_nsi_diag, "effective"),
    }
    nonzero_baselines = [
        name
        for name, value in {
            "source_overlap": source_overlap,
            "zero_nsi": zero_nsi,
            "slot_head": slot_head,
        }.items()
        if isinstance(value, float) and value >= min_nonzero_baseline_accuracy
    ]
    source_delta = (
        float(full_holdout) - source_overlap
        if isinstance(full_holdout, (int, float)) and isinstance(source_overlap, float)
        else None
    )
    zero_nsi_delta = (
        float(full_holdout) - zero_nsi
        if isinstance(full_holdout, (int, float)) and isinstance(zero_nsi, float)
        else None
    )
    slot_head_delta = (
        float(full_holdout) - slot_head
        if isinstance(full_holdout, (int, float)) and isinstance(slot_head, float)
        else None
    )
    data_checks = _dict(data_health.get("checks"))
    checks = {
        "sealed_zero_audit_passed": sealed_zero.get("passed") is True,
        "sealed_all_controls_zero_acknowledged": sealed_zero.get("all_controls_zero") is True,
        "sealed_strong_claim_still_blocked": sealed_zero.get("ready_for_strong_architecture_claim")
        is False,
        "data_health_passed": data_health.get("passed") is True,
        "data_health_no_sealed_reference": data_checks.get("phase2t_no_sealed_reference_anywhere")
        is True,
        "data_health_pressure_matrix_covered": all(
            data_checks.get(key) is True
            for key in [
                "phase2t_train_task_family_coverage",
                "phase2t_train_factor_level_coverage",
                "phase2t_val_task_family_coverage",
                "phase2t_val_factor_level_coverage",
                "phase2t_holdout_task_family_coverage",
                "phase2t_holdout_factor_level_coverage",
            ]
        ),
        "holdout_control_postflight_passed": holdout_postflight.get("passed") is True,
        "holdout_diagnostics_nonsealed": holdout_diag.get("sealed_data_used_for_training_or_tuning")
        is False,
        "zero_nsi_diagnostics_nonsealed": zero_nsi_diag.get(
            "sealed_data_used_for_training_or_tuning"
        )
        is False,
        "full_holdout_accuracy_min": isinstance(full_holdout, (int, float))
        and float(full_holdout) >= min_full_holdout_accuracy,
        "source_overlap_baseline_nonzero": isinstance(source_overlap, float)
        and source_overlap >= min_nonzero_baseline_accuracy,
        "source_overlap_baseline_not_sufficient": isinstance(source_overlap, float)
        and source_overlap <= max_source_overlap_accuracy,
        "zero_nsi_baseline_nonzero": isinstance(zero_nsi, float)
        and zero_nsi >= min_nonzero_baseline_accuracy,
        "slot_head_baseline_nonzero": isinstance(slot_head, float)
        and slot_head >= min_nonzero_baseline_accuracy,
        "full_beats_source_overlap": isinstance(source_delta, float)
        and source_delta >= min_full_minus_source_overlap,
        "full_beats_zero_nsi": isinstance(zero_nsi_delta, float)
        and zero_nsi_delta >= min_full_minus_zero_nsi,
        "full_beats_slot_head": isinstance(slot_head_delta, float)
        and slot_head_delta >= min_full_minus_slot_head,
        "baseline_totals_match": len(set(totals.values())) == 1
        and all(isinstance(value, int) and value > 0 for value in totals.values()),
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_use_phase2t_baseline_feasibility_for_claims")
    if not checks["source_overlap_baseline_nonzero"] or not checks["zero_nsi_baseline_nonzero"] or not checks[
        "slot_head_baseline_nonzero"
    ]:
        blocked_actions.append("add_nonsealed_sanity_tasks_where_controls_can_score_nonzero")
    if not checks["source_overlap_baseline_not_sufficient"]:
        blocked_actions.append("redesign_nonsealed_sanity_subset_source_overlap_is_too_easy")
    if not checks["data_health_pressure_matrix_covered"]:
        blocked_actions.append("do_not_claim_baseline_feasibility_without_graded_coverage")
    if not checks["sealed_strong_claim_still_blocked"]:
        blocked_actions.append("do_not_upgrade_strong_claim_from_baseline_feasibility_alone")
    return {
        "audit_family": "phase2t_baseline_feasibility_sanity_audit",
        "passed": passed,
        "ready_for_bounded_sealed_claim_with_zero_caveat": passed
        and sealed_zero.get("ready_for_bounded_sealed_claim") is True,
        "ready_for_strong_architecture_claim": False,
        "checks": checks,
        "blocked_actions": sorted(set(blocked_actions)),
        "nonzero_baselines": nonzero_baselines,
        "metrics": {
            "full_holdout_accuracy": float(full_holdout)
            if isinstance(full_holdout, (int, float))
            else None,
            "source_overlap_holdout_accuracy": source_overlap,
            "zero_nsi_holdout_accuracy": zero_nsi,
            "slot_head_holdout_accuracy": slot_head,
            "full_minus_source_overlap_holdout_accuracy": source_delta,
            "full_minus_zero_nsi_holdout_accuracy": zero_nsi_delta,
            "full_minus_slot_head_holdout_accuracy": slot_head_delta,
            "baseline_totals": totals,
        },
        "interpretation": {
            "what_this_fixes": (
                "Non-sealed holdout sanity controls show that baselines can score nonzero, "
                "so sealed all-zero controls are not used as an unqualified performance proof."
            ),
            "remaining_boundary": (
                "This supports a bounded mechanism-transfer claim only; it still does not "
                "establish production autonomy, open-ended debugging generalization, or an "
                "epoch-making architecture."
            ),
        },
        "thresholds": {
            "min_full_holdout_accuracy": min_full_holdout_accuracy,
            "min_nonzero_baseline_accuracy": min_nonzero_baseline_accuracy,
            "max_source_overlap_accuracy": max_source_overlap_accuracy,
            "min_full_minus_source_overlap": min_full_minus_source_overlap,
            "min_full_minus_zero_nsi": min_full_minus_zero_nsi,
            "min_full_minus_slot_head": min_full_minus_slot_head,
        },
        "inputs": {
            "sealed_zero_audit_json": str(Path(sealed_zero_audit_json)),
            "holdout_control_postflight_json": str(Path(holdout_control_postflight_json)),
            "holdout_diagnostics_json": str(Path(holdout_diagnostics_json)),
            "zero_nsi_diagnostics_json": str(Path(zero_nsi_diagnostics_json)),
            "data_health_json": str(Path(data_health_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2T dynamic repair trace data.")
    sub = parser.add_subparsers(dest="command", required=True)
    health = sub.add_parser("data-health")
    health.add_argument("--manifest-json", required=True)
    health.add_argument("--train-jsonl", required=True)
    health.add_argument("--val-jsonl", required=True)
    health.add_argument("--holdout-jsonl", required=True)
    health.add_argument("--dataset-root")
    health.add_argument("--min-train-rows", type=int, default=24)
    health.add_argument("--min-val-rows", type=int, default=16)
    health.add_argument("--min-holdout-rows", type=int, default=16)
    health.add_argument("--output-json")
    health.add_argument("--no-fail", action="store_true")

    gate = sub.add_parser("pretrain-gate")
    gate.add_argument("--data-health-json", required=True)
    gate.add_argument("--output-json")
    gate.add_argument("--no-fail", action="store_true")

    postflight = sub.add_parser("smoke-postflight")
    postflight.add_argument("--training-summary-json", required=True)
    postflight.add_argument("--data-health-json", required=True)
    postflight.add_argument("--pretrain-gate-json", required=True)
    postflight.add_argument("--head-manifest-json")
    postflight.add_argument("--output-json")
    postflight.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    postflight.add_argument("--min-model-minus-source-overlap", type=float, default=0.10)
    postflight.add_argument("--max-smoke-duration-seconds", type=float, default=3600.0)
    postflight.add_argument("--no-fail", action="store_true")

    full_postflight = sub.add_parser("full-postflight")
    full_postflight.add_argument("--training-summary-json", required=True)
    full_postflight.add_argument("--data-health-json", required=True)
    full_postflight.add_argument("--pretrain-gate-json", required=True)
    full_postflight.add_argument("--head-manifest-json")
    full_postflight.add_argument("--smoke-postflight-json")
    full_postflight.add_argument("--output-json")
    full_postflight.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    full_postflight.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    full_postflight.add_argument("--min-train-examples", type=int, default=96)
    full_postflight.add_argument("--min-val-examples", type=int, default=64)
    full_postflight.add_argument("--max-full-duration-seconds", type=float, default=7200.0)
    full_postflight.add_argument("--no-fail", action="store_true")

    holdout_postflight = sub.add_parser("holdout-control-postflight")
    holdout_postflight.add_argument("--full-postflight-json", required=True)
    holdout_postflight.add_argument("--holdout-diagnostics-json", required=True)
    holdout_postflight.add_argument("--zero-nsi-diagnostics-json", required=True)
    holdout_postflight.add_argument("--zero-nsi-manifest-json")
    holdout_postflight.add_argument("--output-json")
    holdout_postflight.add_argument("--min-holdout-accuracy", type=float, default=0.85)
    holdout_postflight.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    holdout_postflight.add_argument("--min-full-minus-zero-nsi", type=float, default=0.15)
    holdout_postflight.add_argument("--min-full-minus-slot-head", type=float, default=0.10)
    holdout_postflight.add_argument("--min-holdout-records", type=int, default=96)
    holdout_postflight.add_argument("--no-fail", action="store_true")

    multiseed = sub.add_parser("multiseed-summary")
    multiseed.add_argument("--holdout-postflight-json", action="append", required=True)
    multiseed.add_argument("--output-json")
    multiseed.add_argument("--min-seed-count", type=int, default=3)
    multiseed.add_argument("--min-holdout-accuracy", type=float, default=0.85)
    multiseed.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    multiseed.add_argument("--min-full-minus-zero-nsi", type=float, default=0.15)
    multiseed.add_argument("--no-fail", action="store_true")

    cross_model = sub.add_parser("cross-model-summary")
    cross_model.add_argument("--multiseed-summary-json", action="append", required=True)
    cross_model.add_argument("--output-json")
    cross_model.add_argument("--min-model-count", type=int, default=2)
    cross_model.add_argument("--min-seed-count-per-model", type=int, default=3)
    cross_model.add_argument("--min-holdout-accuracy", type=float, default=0.85)
    cross_model.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    cross_model.add_argument("--min-full-minus-zero-nsi", type=float, default=0.15)
    cross_model.add_argument("--no-fail", action="store_true")

    package_gate = sub.add_parser("package-gate")
    package_gate.add_argument("--cross-model-summary-json", required=True)
    package_gate.add_argument("--canonical-training-summary-json", required=True)
    package_gate.add_argument("--canonical-full-postflight-json", required=True)
    package_gate.add_argument("--canonical-holdout-control-postflight-json", required=True)
    package_gate.add_argument("--data-health-json", required=True)
    package_gate.add_argument("--pretrain-gate-json", required=True)
    package_gate.add_argument("--head-manifest-json", required=True)
    package_gate.add_argument("--adapter-dir", required=True)
    package_gate.add_argument("--output-json")
    package_gate.add_argument("--min-cross-model-count", type=int, default=2)
    package_gate.add_argument("--min-seed-count-per-model", type=int, default=3)
    package_gate.add_argument("--min-holdout-accuracy", type=float, default=0.85)
    package_gate.add_argument("--min-source-overlap-delta", type=float, default=0.15)
    package_gate.add_argument("--min-zero-nsi-delta", type=float, default=0.15)
    package_gate.add_argument("--no-fail", action="store_true")

    postpackage_gate = sub.add_parser("postpackage-gate")
    postpackage_gate.add_argument("--package-gate-json", required=True)
    postpackage_gate.add_argument("--full-package-path", required=True)
    postpackage_gate.add_argument("--no-nsi-package-path", required=True)
    postpackage_gate.add_argument("--native-head-only-package-path", required=True)
    postpackage_gate.add_argument("--continuation-only-package-path", required=True)
    postpackage_gate.add_argument("--output-json")
    postpackage_gate.add_argument("--no-fail", action="store_true")

    zero_audit = sub.add_parser("sealed-zero-baseline-audit")
    zero_audit.add_argument("--external-gate-json", required=True)
    zero_audit.add_argument("--full-eval-json", required=True)
    zero_audit.add_argument("--prompt-eval-json", required=True)
    zero_audit.add_argument("--react-eval-json", required=True)
    zero_audit.add_argument("--no-nsi-eval-json", required=True)
    zero_audit.add_argument("--native-head-only-eval-json", required=True)
    zero_audit.add_argument("--continuation-only-eval-json", required=True)
    zero_audit.add_argument("--output-json")
    zero_audit.add_argument("--no-fail", action="store_true")

    baseline_sanity = sub.add_parser("baseline-feasibility-sanity")
    baseline_sanity.add_argument("--sealed-zero-audit-json", required=True)
    baseline_sanity.add_argument("--holdout-control-postflight-json", required=True)
    baseline_sanity.add_argument("--holdout-diagnostics-json", required=True)
    baseline_sanity.add_argument("--zero-nsi-diagnostics-json", required=True)
    baseline_sanity.add_argument("--data-health-json", required=True)
    baseline_sanity.add_argument("--output-json")
    baseline_sanity.add_argument("--min-full-holdout-accuracy", type=float, default=0.85)
    baseline_sanity.add_argument("--min-nonzero-baseline-accuracy", type=float, default=0.01)
    baseline_sanity.add_argument("--max-source-overlap-accuracy", type=float, default=0.85)
    baseline_sanity.add_argument("--min-full-minus-source-overlap", type=float, default=0.15)
    baseline_sanity.add_argument("--min-full-minus-zero-nsi", type=float, default=0.15)
    baseline_sanity.add_argument("--min-full-minus-slot-head", type=float, default=0.10)
    baseline_sanity.add_argument("--no-fail", action="store_true")

    args = parser.parse_args()
    if args.command == "data-health":
        report = build_phase2t_data_health(
            manifest_json=args.manifest_json,
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            holdout_jsonl=args.holdout_jsonl,
            dataset_root=args.dataset_root,
            min_train_rows=args.min_train_rows,
            min_val_rows=args.min_val_rows,
            min_holdout_rows=args.min_holdout_rows,
        )
    elif args.command == "pretrain-gate":
        report = build_phase2t_pretrain_gate(data_health_json=args.data_health_json)
    elif args.command == "smoke-postflight":
        report = build_phase2t_smoke_postflight(
            training_summary_json=args.training_summary_json,
            data_health_json=args.data_health_json,
            pretrain_gate_json=args.pretrain_gate_json,
            head_manifest_json=args.head_manifest_json,
            min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
            min_model_minus_source_overlap=args.min_model_minus_source_overlap,
            max_smoke_duration_seconds=args.max_smoke_duration_seconds,
        )
    elif args.command == "full-postflight":
        report = build_phase2t_full_postflight(
            training_summary_json=args.training_summary_json,
            data_health_json=args.data_health_json,
            pretrain_gate_json=args.pretrain_gate_json,
            head_manifest_json=args.head_manifest_json,
            smoke_postflight_json=args.smoke_postflight_json,
            min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
            min_model_minus_source_overlap=args.min_model_minus_source_overlap,
            min_train_examples=args.min_train_examples,
            min_val_examples=args.min_val_examples,
            max_full_duration_seconds=args.max_full_duration_seconds,
        )
    elif args.command == "holdout-control-postflight":
        report = build_phase2t_holdout_control_postflight(
            full_postflight_json=args.full_postflight_json,
            holdout_diagnostics_json=args.holdout_diagnostics_json,
            zero_nsi_diagnostics_json=args.zero_nsi_diagnostics_json,
            zero_nsi_manifest_json=args.zero_nsi_manifest_json,
            min_holdout_accuracy=args.min_holdout_accuracy,
            min_model_minus_source_overlap=args.min_model_minus_source_overlap,
            min_full_minus_zero_nsi=args.min_full_minus_zero_nsi,
            min_full_minus_slot_head=args.min_full_minus_slot_head,
            min_holdout_records=args.min_holdout_records,
        )
    elif args.command == "multiseed-summary":
        report = build_phase2t_multiseed_summary(
            holdout_postflight_jsons=args.holdout_postflight_json,
            min_seed_count=args.min_seed_count,
            min_holdout_accuracy=args.min_holdout_accuracy,
            min_model_minus_source_overlap=args.min_model_minus_source_overlap,
            min_full_minus_zero_nsi=args.min_full_minus_zero_nsi,
        )
    elif args.command == "cross-model-summary":
        report = build_phase2t_cross_model_summary(
            multiseed_summary_jsons=args.multiseed_summary_json,
            min_model_count=args.min_model_count,
            min_seed_count_per_model=args.min_seed_count_per_model,
            min_holdout_accuracy=args.min_holdout_accuracy,
            min_model_minus_source_overlap=args.min_model_minus_source_overlap,
            min_full_minus_zero_nsi=args.min_full_minus_zero_nsi,
        )
    elif args.command == "package-gate":
        report = build_phase2t_package_gate(
            cross_model_summary_json=args.cross_model_summary_json,
            canonical_training_summary_json=args.canonical_training_summary_json,
            canonical_full_postflight_json=args.canonical_full_postflight_json,
            canonical_holdout_control_postflight_json=args.canonical_holdout_control_postflight_json,
            data_health_json=args.data_health_json,
            pretrain_gate_json=args.pretrain_gate_json,
            head_manifest_json=args.head_manifest_json,
            adapter_dir=args.adapter_dir,
            min_cross_model_count=args.min_cross_model_count,
            min_seed_count_per_model=args.min_seed_count_per_model,
            min_holdout_accuracy=args.min_holdout_accuracy,
            min_source_overlap_delta=args.min_source_overlap_delta,
            min_zero_nsi_delta=args.min_zero_nsi_delta,
        )
    elif args.command == "postpackage-gate":
        report = build_phase2t_postpackage_gate(
            package_gate_json=args.package_gate_json,
            full_package_path=args.full_package_path,
            no_nsi_package_path=args.no_nsi_package_path,
            native_head_only_package_path=args.native_head_only_package_path,
            continuation_only_package_path=args.continuation_only_package_path,
        )
    elif args.command == "sealed-zero-baseline-audit":
        report = build_phase2t_sealed_zero_baseline_audit(
            external_gate_json=args.external_gate_json,
            full_eval_json=args.full_eval_json,
            prompt_eval_json=args.prompt_eval_json,
            react_eval_json=args.react_eval_json,
            no_nsi_eval_json=args.no_nsi_eval_json,
            native_head_only_eval_json=args.native_head_only_eval_json,
            continuation_only_eval_json=args.continuation_only_eval_json,
        )
    else:
        report = build_phase2t_baseline_feasibility_sanity_audit(
            sealed_zero_audit_json=args.sealed_zero_audit_json,
            holdout_control_postflight_json=args.holdout_control_postflight_json,
            holdout_diagnostics_json=args.holdout_diagnostics_json,
            zero_nsi_diagnostics_json=args.zero_nsi_diagnostics_json,
            data_health_json=args.data_health_json,
            min_full_holdout_accuracy=args.min_full_holdout_accuracy,
            min_nonzero_baseline_accuracy=args.min_nonzero_baseline_accuracy,
            max_source_overlap_accuracy=args.max_source_overlap_accuracy,
            min_full_minus_source_overlap=args.min_full_minus_source_overlap,
            min_full_minus_zero_nsi=args.min_full_minus_zero_nsi,
            min_full_minus_slot_head=args.min_full_minus_slot_head,
        )
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
