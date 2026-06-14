from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_SOURCE_KINDS = {"public_repo", "synthetic_safe_repo"}
CLAIM_BEARING_SOURCE_KINDS = {"public_repo"}
REQUIRED_TASK_FAMILIES = {
    "dependency_or_import_mismatch",
    "localized_unit_assertion",
    "stale_snapshot_update",
    "config_or_environment_marker",
    "multi_file_traceback_relation",
}
REQUIRED_FACTOR_LEVELS = {
    "candidate_count": {"2", "3", "4"},
    "evidence_density": {"low", "medium", "high"},
    "repair_depth": {"one_edit", "two_edits", "stale_state_refresh"},
    "failure_observability": {
        "direct_traceback",
        "indirect_changed_file_relation",
        "ambiguous_same_intent_command",
    },
    "ambiguity_class": {
        "same_intent_command",
        "same_file_read",
        "stage_transition",
    },
}
REQUIRED_BASELINES = {
    "source_overlap",
    "native_head_only_no_cache",
    "continuation_only",
    "prompt_only",
    "react",
    "modern_coding_agent_loop",
}
BASELINE_METHODS = {
    "source_overlap": "repair_candidate_overlap_current_visible_v1",
    "native_head_only_no_cache": "first_repair_candidate_no_latent_v1",
    "continuation_only": "prior_repair_summary_overlap_v1",
    "prompt_only": "current_plus_runtime_repair_overlap_v1",
    "react": "runtime_evidence_repair_overlap_v1",
    "modern_coding_agent_loop": "sandbox_visible_repair_overlap_smoke_v1",
}
REQUIRED_ARTIFACT_KEYS = {
    "patch_diff",
    "command_log",
    "test_output",
    "rollback_log",
    "sandbox_integrity_report",
}
REQUIRED_RUNTIME_FLAGS = {
    "patch_application_recorded",
    "post_patch_tests_recorded",
    "rollback_recorded",
    "sandbox_cleanup_recorded",
    "source_repo_read_only_observed",
    "bounded_edit_scope_observed",
    "command_allowlist_observed",
}
SEALED_MARKERS = (
    "external_trace_v3_semantic_required",
    "phase2i_external_trace",
    "phase2g_external_trace",
    "sealed_failure",
)
HIDDEN_MARKERS = (
    "hidden_hint",
    "gold_label",
    "gold_command",
    "correct_patch",
    "correct_command_hint",
    "sealed_feedback",
)
CANDIDATE_SLOT_RE = re.compile(r"(?i)\bcandidate[_-]?\d+\b|\bslot[_-]?\d+\b")
ABSOLUTE_PATH_RE = re.compile(
    r"(?i)((?<![A-Za-z])[A-Z]:[\\/]|\\\\[A-Za-z0-9_.-]+\\|/Users/|/home/|/root/|/var/folders/)"
)
SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[A-Za-z0-9_./+\-]{8,}"
)


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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _get_dict(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def _get_list(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    return value if isinstance(value, list) else []


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^A-Za-z0-9_]+", text.lower()) if token}


def _repair_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in _get_list(row, "repair_candidates"):
        if isinstance(item, dict):
            candidates.append(item)
    return candidates


def _candidate_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for candidate in _repair_candidates(row):
        candidate_id = candidate.get("repair_action")
        if candidate_id is not None:
            ids.append(str(candidate_id))
    return ids


def _candidate_texts(row: dict[str, Any]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for candidate in _repair_candidates(row):
        candidate_id = candidate.get("repair_action")
        if candidate_id is None:
            continue
        texts[str(candidate_id)] = _canonical_json(
            {
                "intent": candidate.get("intent"),
                "edit_scope": candidate.get("edit_scope"),
                "description": candidate.get("description"),
                "verification_command": candidate.get("verification_command"),
            }
        )
    return texts


def _overlap_prediction(text: str, candidate_texts: dict[str, str]) -> str | None:
    if not candidate_texts:
        return None
    text_tokens = _tokens(text)
    scored = [
        (len(text_tokens & _tokens(candidate_text)), -index, candidate_id)
        for index, (candidate_id, candidate_text) in enumerate(candidate_texts.items())
    ]
    return max(scored)[2]


def compute_phase2s_baseline_predictions(row: dict[str, Any]) -> dict[str, str | None]:
    candidate_ids = _candidate_ids(row)
    candidate_texts = _candidate_texts(row)
    current_visible = str(row.get("current_visible_text") or "")
    runtime = _canonical_json(row.get("runtime_visible_evidence") or {})
    prior = str(_get_dict(row, "runtime_visible_evidence").get("prior_repair_summary") or "")
    sandbox_visible = _canonical_json(
        {
            "current_visible_text": current_visible,
            "runtime_visible_evidence": row.get("runtime_visible_evidence") or {},
            "repair_candidates": row.get("repair_candidates") or [],
        }
    )
    return {
        "source_overlap": _overlap_prediction(current_visible, candidate_texts),
        "native_head_only_no_cache": candidate_ids[0] if candidate_ids else None,
        "continuation_only": _overlap_prediction(prior, candidate_texts),
        "prompt_only": _overlap_prediction(f"{current_visible}\n{runtime}", candidate_texts),
        "react": _overlap_prediction(runtime, candidate_texts),
        "modern_coding_agent_loop": _overlap_prediction(sandbox_visible, candidate_texts),
    }


def _baseline_prediction(row: dict[str, Any], baseline: str) -> str | None:
    baselines = _get_dict(row, "baselines")
    value = baselines.get(baseline)
    if isinstance(value, dict):
        value = value.get("predicted_repair_action")
    return str(value) if value is not None else None


def _baseline_metadata_ok(row: dict[str, Any]) -> bool:
    metadata = _get_dict(row, "baseline_metadata")
    if not metadata:
        return False
    for baseline, method in BASELINE_METHODS.items():
        payload = metadata.get(baseline)
        if not isinstance(payload, dict):
            return False
        if payload.get("measured") is not True:
            return False
        if payload.get("method") != method:
            return False
        if payload.get("uses_expected_repair_action") is not False:
            return False
        if payload.get("uses_sealed_feedback") is not False:
            return False
    return True


def _baselines_match_computed(row: dict[str, Any]) -> bool:
    computed = compute_phase2s_baseline_predictions(row)
    return all(_baseline_prediction(row, name) == computed[name] for name in REQUIRED_BASELINES)


def _difficulty(row: dict[str, Any], key: str) -> Any:
    difficulty = _get_dict(row, "difficulty")
    return difficulty.get(key, row.get(key))


def _visible_payload(row: dict[str, Any]) -> str:
    return _canonical_json(
        {
            "current_visible_text": row.get("current_visible_text"),
            "runtime_visible_evidence": row.get("runtime_visible_evidence"),
            "repair_candidates": row.get("repair_candidates"),
        }
    )


def _row_mentions_sealed(row: dict[str, Any]) -> bool:
    text = _canonical_json(row).replace("\\", "/").lower()
    return any(marker in text for marker in SEALED_MARKERS)


def _row_mentions_hidden_marker(row: dict[str, Any]) -> bool:
    text = _visible_payload(row).lower()
    return any(marker in text for marker in HIDDEN_MARKERS)


def _row_mentions_candidate_slot_marker(row: dict[str, Any]) -> bool:
    return bool(CANDIDATE_SLOT_RE.search(_visible_payload(row)))


def _row_has_redaction_leak(row: dict[str, Any]) -> bool:
    text = _visible_payload(row)
    return bool(ABSOLUTE_PATH_RE.search(text) or SECRET_RE.search(text))


def _normalization_ok(row: dict[str, Any]) -> bool:
    normalization = _get_dict(row, "normalization")
    return all(
        normalization.get(key) is True
        for key in (
            "deterministic",
            "redacted_absolute_local_paths",
            "redacted_secrets_tokens_and_emails",
            "preserved_runtime_visible_evidence",
        )
    )


def _provenance_ok(row: dict[str, Any]) -> bool:
    return (
        str(row.get("source_kind") or "") in ALLOWED_SOURCE_KINDS
        and bool(row.get("repo_id"))
        and bool(row.get("repo_url_or_origin"))
        and len(str(row.get("commit_hash") or "")) >= 7
        and bool(row.get("license_or_synthetic_origin"))
        and len(str(row.get("collection_script_hash") or "")) >= 16
        and bool(row.get("trace_hash"))
    )


def _row_shape_ok(row: dict[str, Any]) -> bool:
    candidate_ids = _candidate_ids(row)
    expected = row.get("expected_repair_action")
    try:
        declared_count = int(_difficulty(row, "candidate_count"))
    except (TypeError, ValueError):
        declared_count = -1
    return (
        bool(row.get("trace_id"))
        and str(row.get("split") or "") in {"train", "val", "holdout"}
        and expected in candidate_ids
        and len(candidate_ids) >= 2
        and declared_count == len(candidate_ids)
        and len(set(candidate_ids)) == len(candidate_ids)
        and all(_baseline_prediction(row, baseline) in candidate_ids for baseline in REQUIRED_BASELINES)
    )


def _runtime_flags_ok(row: dict[str, Any]) -> bool:
    runtime = _get_dict(row, "repair_runtime")
    return all(runtime.get(key) is True for key in REQUIRED_RUNTIME_FLAGS)


def _artifact_paths_ok(row: dict[str, Any], dataset_root: Path) -> bool:
    paths = _get_dict(row, "artifact_paths")
    if not REQUIRED_ARTIFACT_KEYS.issubset(set(paths)):
        return False
    for key in REQUIRED_ARTIFACT_KEYS:
        value = paths.get(key)
        if not value:
            return False
        path = (dataset_root / str(value)).resolve()
        try:
            path.relative_to(dataset_root.resolve())
        except ValueError:
            return False
        if not path.exists() or path.stat().st_size <= 0:
            return False
    return True


def _all_artifacts_ok(rows: list[dict[str, Any]], dataset_root: Path) -> bool:
    return bool(rows) and all(_artifact_paths_ok(row, dataset_root) for row in rows)


def _split_rows_have_split(rows: list[dict[str, Any]], split: str) -> bool:
    return bool(rows) and all(str(row.get("split")) == split for row in rows)


def _repo_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("repo_id")) for row in rows if row.get("repo_id")}


def _normalize_repo_origin(row: dict[str, Any]) -> str | None:
    origin = str(row.get("repo_url_or_origin") or row.get("repo_id") or "").strip()
    if not origin:
        return None
    normalized = origin.replace("\\", "/").rstrip("/").lower()
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


def _repo_origins(rows: list[dict[str, Any]]) -> set[str]:
    return {origin for row in rows if (origin := _normalize_repo_origin(row))}


def _all_split_repos_disjoint(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> bool:
    train = _repo_ids(train_rows)
    val = _repo_ids(val_rows)
    holdout = _repo_ids(holdout_rows)
    train_origins = _repo_origins(train_rows)
    val_origins = _repo_origins(val_rows)
    holdout_origins = _repo_origins(holdout_rows)
    ids_disjoint = (
        bool(train and val and holdout)
        and train.isdisjoint(val)
        and train.isdisjoint(holdout)
        and val.isdisjoint(holdout)
    )
    origins_disjoint = (
        bool(train_origins and val_origins and holdout_origins)
        and train_origins.isdisjoint(val_origins)
        and train_origins.isdisjoint(holdout_origins)
        and val_origins.isdisjoint(holdout_origins)
    )
    return ids_disjoint and origins_disjoint


def _dedup_ok(all_rows: list[dict[str, Any]]) -> bool:
    keys = [
        (
            str(row.get("repo_id")),
            str(row.get("commit_hash")),
            str(row.get("trace_hash")),
        )
        for row in all_rows
    ]
    return len(keys) == len(set(keys))


def _factor_coverage(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for factor, required in REQUIRED_FACTOR_LEVELS.items():
        observed = {str(_difficulty(row, factor)) for row in rows}
        factor_missing = sorted(required - observed)
        if factor_missing:
            missing[factor] = factor_missing
    return missing


def _task_family_coverage(rows: list[dict[str, Any]]) -> list[str]:
    observed = {str(_difficulty(row, "task_family")) for row in rows}
    return sorted(REQUIRED_TASK_FAMILIES - observed)


def _baseline_rollup(rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    total = 0
    correct = 0
    for row in rows:
        expected = row.get("expected_repair_action")
        prediction = _baseline_prediction(row, baseline)
        if expected is None or prediction is None:
            continue
        total += 1
        correct += int(prediction == expected)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
    }


def _expected_repair_slot(row: dict[str, Any]) -> int | None:
    candidate_ids = _candidate_ids(row)
    expected = row.get("expected_repair_action")
    if expected not in candidate_ids:
        return None
    return candidate_ids.index(expected)


def _repair_slot_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    slots: Counter[int] = Counter()
    for row in rows:
        slot = _expected_repair_slot(row)
        if slot is not None:
            slots[slot] += 1
    total = sum(slots.values())
    return {
        "total": total,
        "slots": {str(key): value for key, value in sorted(slots.items())},
        "max_share": max((value / total for value in slots.values()), default=None),
    }


def _split_hash(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    stable_rows = sorted(rows, key=lambda row: str(row.get("trace_id") or row.get("trace_hash")))
    return _sha256(stable_rows)


def _rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "source_kinds": sorted({str(row.get("source_kind")) for row in rows if row.get("source_kind")}),
        "repo_ids": sorted(_repo_ids(rows)),
        "repo_origins": sorted(_repo_origins(rows)),
        "task_families": sorted({str(_difficulty(row, "task_family")) for row in rows}),
        "factor_levels": {
            factor: sorted({str(_difficulty(row, factor)) for row in rows})
            for factor in REQUIRED_FACTOR_LEVELS
        },
    }


def build_phase2s_data_health(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    dataset_root: str | Path | None = None,
    min_train_rows: int = 15,
    min_val_rows: int = 15,
    min_holdout_rows: int = 9,
    max_required_baseline_val_accuracy: float = 0.75,
    max_repair_slot_share: float = 0.60,
) -> dict[str, Any]:
    train_rows, train_exists = _read_jsonl(train_jsonl)
    val_rows, val_exists = _read_jsonl(val_jsonl)
    holdout_rows, holdout_exists = _read_jsonl(holdout_jsonl)
    all_rows = train_rows + val_rows + holdout_rows
    root = Path(dataset_root) if dataset_root else Path(train_jsonl).parent
    baselines = {
        "train": {name: _baseline_rollup(train_rows, name) for name in REQUIRED_BASELINES},
        "val": {name: _baseline_rollup(val_rows, name) for name in REQUIRED_BASELINES},
        "holdout": {
            name: _baseline_rollup(holdout_rows, name) for name in REQUIRED_BASELINES
        },
    }
    baseline_val_accuracies = {
        name: baselines["val"][name]["accuracy"] for name in REQUIRED_BASELINES
    }
    repair_slots = {
        "train": _repair_slot_rollup(train_rows),
        "val": _repair_slot_rollup(val_rows),
        "holdout": _repair_slot_rollup(holdout_rows),
    }
    all_source_kinds = {str(row.get("source_kind") or "") for row in all_rows}
    claim_bearing_training_ready = bool(all_rows) and all_source_kinds.issubset(
        CLAIM_BEARING_SOURCE_KINDS
    )

    checks = {
        "phase2s_train_jsonl_exists": train_exists,
        "phase2s_val_jsonl_exists": val_exists,
        "phase2s_holdout_jsonl_exists": holdout_exists,
        "phase2s_train_rows_minimum_met": len(train_rows) >= min_train_rows,
        "phase2s_val_rows_minimum_met": len(val_rows) >= min_val_rows,
        "phase2s_holdout_rows_minimum_met": len(holdout_rows) >= min_holdout_rows,
        "phase2s_split_labels_match_files": (
            _split_rows_have_split(train_rows, "train")
            and _split_rows_have_split(val_rows, "val")
            and _split_rows_have_split(holdout_rows, "holdout")
        ),
        "phase2s_provenance_and_license_present": bool(all_rows)
        and all(_provenance_ok(row) for row in all_rows),
        "phase2s_normalization_and_redaction_flags_present": bool(all_rows)
        and all(_normalization_ok(row) for row in all_rows),
        "phase2s_no_redaction_leaks_visible": not any(
            _row_has_redaction_leak(row) for row in all_rows
        ),
        "phase2s_no_hidden_gold_or_sealed_visible": not any(
            _row_mentions_hidden_marker(row) for row in all_rows
        ),
        "phase2s_no_candidate_slot_marker_visible": not any(
            _row_mentions_candidate_slot_marker(row) for row in all_rows
        ),
        "phase2s_no_sealed_reference_anywhere": not any(
            _row_mentions_sealed(row) for row in all_rows
        ),
        "phase2s_rows_have_candidates_labels_and_baselines": bool(all_rows)
        and all(_row_shape_ok(row) for row in all_rows),
        "phase2s_runtime_flags_present": bool(all_rows)
        and all(_runtime_flags_ok(row) for row in all_rows),
        "phase2s_required_artifacts_present": _all_artifacts_ok(all_rows, root),
        "phase2s_baseline_metadata_measured": bool(all_rows)
        and all(_baseline_metadata_ok(row) for row in all_rows),
        "phase2s_baselines_match_computed_predictions": bool(all_rows)
        and all(_baselines_match_computed(row) for row in all_rows),
        "phase2s_all_split_repos_disjoint": _all_split_repos_disjoint(
            train_rows, val_rows, holdout_rows
        ),
        "phase2s_deduplicated_by_repo_commit_trace_hash": bool(all_rows)
        and _dedup_ok(all_rows),
        "phase2s_val_task_family_coverage": not _task_family_coverage(val_rows),
        "phase2s_val_factor_level_coverage": not _factor_coverage(val_rows),
        "phase2s_all_required_baselines_val_below_threshold": all(
            isinstance(value, float) and value <= max_required_baseline_val_accuracy
            for value in baseline_val_accuracies.values()
        ),
        "phase2s_repair_slot_share_below_threshold": all(
            isinstance(repair_slots[split]["max_share"], float)
            and repair_slots[split]["max_share"] <= max_repair_slot_share
            for split in ("train", "val")
        ),
        "phase2s_synthetic_safe_only_is_smoke_not_claim_bearing": (
            True if claim_bearing_training_ready else all_source_kinds == {"synthetic_safe_repo"}
        ),
    }

    blocked_actions: list[str] = []
    if not all(checks.values()):
        blocked_actions.append("do_not_train_phase2s_until_data_health_passes")
    if not checks["phase2s_no_sealed_reference_anywhere"]:
        blocked_actions.append("do_not_use_sealed_or_sealed_failure_feedback")
    if not checks["phase2s_no_candidate_slot_marker_visible"]:
        blocked_actions.append("do_not_train_with_candidate_slot_markers_visible")
    if not checks["phase2s_no_redaction_leaks_visible"]:
        blocked_actions.append("do_not_train_with_unredacted_phase2s_traces")
    if not checks["phase2s_baseline_metadata_measured"]:
        blocked_actions.append("measure_phase2s_baselines_with_code_before_training")
    if not checks["phase2s_baselines_match_computed_predictions"]:
        blocked_actions.append("do_not_train_with_declared_or_stale_phase2s_baselines")
    if not checks["phase2s_all_split_repos_disjoint"]:
        blocked_actions.append("do_not_train_without_repo_disjoint_splits")
    if not checks["phase2s_val_task_family_coverage"] or not checks[
        "phase2s_val_factor_level_coverage"
    ]:
        blocked_actions.append("do_not_train_phase2s_until_val_pressure_matrix_is_covered")
    if not checks["phase2s_all_required_baselines_val_below_threshold"]:
        blocked_actions.append("do_not_train_when_any_required_baseline_solves_phase2s_val")
    if not checks["phase2s_required_artifacts_present"]:
        blocked_actions.append("do_not_train_without_patch_test_rollback_sandbox_artifacts")
    if not checks["phase2s_runtime_flags_present"]:
        blocked_actions.append("do_not_train_without_repair_runtime_contract_evidence")

    passed = all(checks.values())
    return {
        "audit_family": "phase2s_open_repair_data_health",
        "passed": passed,
        "claim_bearing_training_ready": claim_bearing_training_ready and passed,
        "allowed_next_action": (
            "run_phase2s_claim_bearing_smoke_training_only"
            if claim_bearing_training_ready and passed
            else "collect_public_phase2s_repair_traces_before_training"
            if passed
            else "revise_phase2s_repair_data_before_training"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "thresholds": {
            "min_train_rows": min_train_rows,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "max_required_baseline_val_accuracy": max_required_baseline_val_accuracy,
            "max_repair_slot_share": max_repair_slot_share,
            "required_task_families": sorted(REQUIRED_TASK_FAMILIES),
            "required_factor_levels": {
                key: sorted(values) for key, values in REQUIRED_FACTOR_LEVELS.items()
            },
            "required_baselines": sorted(REQUIRED_BASELINES),
            "baseline_methods": BASELINE_METHODS,
        },
        "rollups": {
            "train": _rollup(train_rows),
            "val": _rollup(val_rows),
            "holdout": _rollup(holdout_rows),
            "baselines": baselines,
            "repair_slots": repair_slots,
            "missing_val_task_families": _task_family_coverage(val_rows),
            "missing_val_factor_levels": _factor_coverage(val_rows),
        },
        "effective_split_hashes": {
            "phase2s_train": _split_hash(train_rows),
            "phase2s_val": _split_hash(val_rows),
            "phase2s_holdout": _split_hash(holdout_rows),
        },
        "inputs": {
            "train_jsonl": str(Path(train_jsonl)),
            "val_jsonl": str(Path(val_jsonl)),
            "holdout_jsonl": str(Path(holdout_jsonl)),
            "dataset_root": str(root),
        },
    }


def build_phase2s_pretrain_gate(*, data_health_json: str | Path) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    split_hashes = data_health.get("effective_split_hashes", {})
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "claim_bearing_training_ready": data_health.get("claim_bearing_training_ready") is True,
        "effective_split_hashes_present": all(
            split_hashes.get(key)
            for key in ("phase2s_train", "phase2s_val", "phase2s_holdout")
        ),
        "baselines_below_threshold": data_health.get("checks", {}).get(
            "phase2s_all_required_baselines_val_below_threshold"
        )
        is True,
        "repair_runtime_contract_evidenced": data_health.get("checks", {}).get(
            "phase2s_runtime_flags_present"
        )
        is True,
        "artifacts_present": data_health.get("checks", {}).get(
            "phase2s_required_artifacts_present"
        )
        is True,
        "sealed_not_used": data_health.get("checks", {}).get(
            "phase2s_no_sealed_reference_anywhere"
        )
        is True,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_train_phase2s_until_pretrain_gate_passes")
    if not checks["claim_bearing_training_ready"]:
        blocked_actions.append("do_not_train_phase2s_from_synthetic_smoke_only")
    return {
        "audit_family": "phase2s_open_repair_pretrain_gate",
        "passed": passed,
        "allowed_next_action": (
            "run_phase2s_claim_bearing_smoke_training_only"
            if passed
            else "collect_public_phase2s_repair_traces_before_training"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "effective_split_hashes": split_hashes,
        "inputs": {"data_health_json": str(Path(data_health_json))},
    }


def _metric_from_summary(summary: dict[str, Any], metric: str) -> float | None:
    history = summary.get("history")
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict):
            val_metrics = latest.get("val_metrics")
            if isinstance(val_metrics, dict) and isinstance(val_metrics.get(metric), (int, float)):
                return float(val_metrics[metric])
    run_manifest = summary.get("run_manifest")
    if isinstance(run_manifest, dict):
        history = run_manifest.get("history")
        if isinstance(history, list) and history:
            latest = history[-1]
            if isinstance(latest, dict):
                val_metrics = latest.get("val_metrics")
                if isinstance(val_metrics, dict) and isinstance(val_metrics.get(metric), (int, float)):
                    return float(val_metrics[metric])
    return None


def build_phase2s_smoke_postflight(
    *,
    training_summary_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    head_manifest_json: str | Path | None = None,
    zero_nsi_diagnostics_json: str | Path | None = None,
    min_val_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.10,
    min_model_minus_zero_nsi: float | None = None,
    max_smoke_duration_seconds: float = 3600.0,
) -> dict[str, Any]:
    summary = _read_json(training_summary_json)
    data_health = _read_json(data_health_json)
    pretrain_gate = _read_json(pretrain_gate_json)
    head_manifest = _read_json(head_manifest_json) if head_manifest_json else {}
    zero_nsi_diagnostics = (
        _read_json(zero_nsi_diagnostics_json) if zero_nsi_diagnostics_json else {}
    )
    val_command_slot_accuracy = _metric_from_summary(summary, "command_slot_accuracy")
    source_overlap = _get_dict(summary, "source_overlap_command_slot_baseline")
    source_overlap_val = _get_dict(source_overlap, "val")
    source_overlap_accuracy = source_overlap_val.get("accuracy")
    if not isinstance(source_overlap_accuracy, (int, float)):
        source_overlap_accuracy = None
    model_minus_source_overlap = (
        float(val_command_slot_accuracy) - float(source_overlap_accuracy)
        if isinstance(val_command_slot_accuracy, float)
        and isinstance(source_overlap_accuracy, (int, float))
        else None
    )
    zero_nsi_accuracy = None
    if zero_nsi_diagnostics:
        zero_sources = zero_nsi_diagnostics.get("sources")
        if isinstance(zero_sources, dict):
            zero_effective = zero_sources.get("effective")
            if isinstance(zero_effective, dict) and isinstance(
                zero_effective.get("accuracy"), (int, float)
            ):
                zero_nsi_accuracy = float(zero_effective["accuracy"])
    model_minus_zero_nsi = (
        float(val_command_slot_accuracy) - zero_nsi_accuracy
        if isinstance(val_command_slot_accuracy, float)
        and isinstance(zero_nsi_accuracy, float)
        else None
    )
    run_manifest = _get_dict(summary, "run_manifest")
    duration_seconds = run_manifest.get("duration_seconds")
    pairwise = bool(summary.get("use_pairwise_command_reranker"))
    low_level_calls_target = summary.get("low_level_qwen_calls_target")
    no_json_motor_target = summary.get("no_json_motor_target")
    data_hashes = data_health.get("effective_split_hashes")
    gate_hashes = pretrain_gate.get("effective_split_hashes")
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain_gate.get("passed") is True,
        "head_manifest_source_gates_passed": (
            not head_manifest
            or (
                head_manifest.get("source_data_health_passed") is True
                and head_manifest.get("source_pretrain_gate_passed") is True
            )
        ),
        "data_and_pretrain_hashes_match": bool(data_hashes)
        and bool(gate_hashes)
        and data_hashes == gate_hashes,
        "head_manifest_hashes_match_data_health": (
            not head_manifest
            or head_manifest.get("effective_split_hashes") == data_hashes
        ),
        "no_json_motor_target": no_json_motor_target is True,
        "low_level_qwen_calls_target_zero": low_level_calls_target == 0,
        "pairwise_disabled_for_phase2s_smoke": pairwise is False,
        "val_command_slot_accuracy_min": isinstance(val_command_slot_accuracy, float)
        and val_command_slot_accuracy >= min_val_command_slot_accuracy,
        "model_minus_source_overlap_min": isinstance(model_minus_source_overlap, float)
        and model_minus_source_overlap >= min_model_minus_source_overlap,
        "zero_nsi_diagnostics_gate": (
            True
            if min_model_minus_zero_nsi is None
            else isinstance(model_minus_zero_nsi, float)
            and model_minus_zero_nsi >= min_model_minus_zero_nsi
        ),
        "smoke_duration_within_limit": isinstance(duration_seconds, (int, float))
        and float(duration_seconds) <= max_smoke_duration_seconds,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_run_phase2s_full_train_until_smoke_postflight_passes")
    if not checks["model_minus_source_overlap_min"]:
        blocked_actions.append("do_not_claim_phase2s_mechanism_delta_from_source_overlap")
    if not checks["pairwise_disabled_for_phase2s_smoke"]:
        blocked_actions.append("do_not_mix_phase2s_smoke_with_pairwise_mechanism")
    if not checks["zero_nsi_diagnostics_gate"]:
        blocked_actions.append("do_not_claim_phase2s_nsi_identity_delta_from_smoke")
    if not checks["data_and_pretrain_hashes_match"] or not checks[
        "head_manifest_hashes_match_data_health"
    ]:
        blocked_actions.append("do_not_train_or_package_with_phase2s_hash_mismatch")
    return {
        "audit_family": "phase2s_open_repair_smoke_postflight",
        "passed": passed,
        "allowed_next_action": (
            "run_phase2s_full_nonsealed_training_only"
            if passed
            else "freeze_phase2s_smoke_failure_and_analyze_nonsealed_design"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "metrics": {
            "val_command_slot_accuracy": val_command_slot_accuracy,
            "source_overlap_val_accuracy": (
                float(source_overlap_accuracy)
                if isinstance(source_overlap_accuracy, (int, float))
                else None
            ),
            "model_minus_source_overlap_accuracy": model_minus_source_overlap,
            "zero_nsi_effective_accuracy": zero_nsi_accuracy,
            "model_minus_zero_nsi_accuracy": model_minus_zero_nsi,
            "duration_seconds": float(duration_seconds)
            if isinstance(duration_seconds, (int, float))
            else None,
            "low_level_qwen_calls_target": low_level_calls_target,
            "use_pairwise_command_reranker": pairwise,
            "command_candidate_encoder": summary.get("command_candidate_encoder"),
        },
        "thresholds": {
            "min_val_command_slot_accuracy": min_val_command_slot_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_model_minus_zero_nsi": min_model_minus_zero_nsi,
            "max_smoke_duration_seconds": max_smoke_duration_seconds,
        },
        "effective_split_hashes": data_hashes,
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "data_health_json": str(Path(data_health_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "head_manifest_json": str(Path(head_manifest_json)) if head_manifest_json else None,
            "zero_nsi_diagnostics_json": (
                str(Path(zero_nsi_diagnostics_json)) if zero_nsi_diagnostics_json else None
            ),
        },
    }


def _accuracy_from_diagnostics(
    diagnostics: dict[str, Any],
    source_name: str,
) -> float | None:
    sources = diagnostics.get("sources")
    if not isinstance(sources, dict):
        return None
    source = sources.get(source_name)
    if not isinstance(source, dict):
        return None
    accuracy = source.get("accuracy")
    return float(accuracy) if isinstance(accuracy, (int, float)) else None


def build_phase2s_full_holdout_postflight(
    *,
    training_summary_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    head_manifest_json: str | Path | None = None,
    holdout_diagnostics_json: str | Path,
    holdout_zero_nsi_diagnostics_json: str | Path | None = None,
    min_val_command_slot_accuracy: float = 0.85,
    min_holdout_command_slot_accuracy: float = 0.85,
    min_val_model_minus_source_overlap: float = 0.15,
    min_holdout_model_minus_source_overlap: float = 0.15,
    min_holdout_model_minus_zero_nsi: float | None = None,
) -> dict[str, Any]:
    summary = _read_json(training_summary_json)
    data_health = _read_json(data_health_json)
    pretrain_gate = _read_json(pretrain_gate_json)
    head_manifest = _read_json(head_manifest_json) if head_manifest_json else {}
    holdout_diagnostics = _read_json(holdout_diagnostics_json)
    zero_nsi_diagnostics = (
        _read_json(holdout_zero_nsi_diagnostics_json)
        if holdout_zero_nsi_diagnostics_json
        else {}
    )

    val_command_slot_accuracy = _metric_from_summary(summary, "command_slot_accuracy")
    source_overlap = _get_dict(summary, "source_overlap_command_slot_baseline")
    source_overlap_val = _get_dict(source_overlap, "val")
    source_overlap_val_accuracy = source_overlap_val.get("accuracy")
    if not isinstance(source_overlap_val_accuracy, (int, float)):
        source_overlap_val_accuracy = None
    val_model_minus_source_overlap = (
        float(val_command_slot_accuracy) - float(source_overlap_val_accuracy)
        if isinstance(val_command_slot_accuracy, float)
        and isinstance(source_overlap_val_accuracy, (int, float))
        else None
    )

    holdout_accuracy = _accuracy_from_diagnostics(holdout_diagnostics, "effective")
    holdout_source_overlap_accuracy = _accuracy_from_diagnostics(
        holdout_diagnostics,
        "source_overlap_baseline",
    )
    holdout_model_minus_source_overlap = (
        holdout_accuracy - holdout_source_overlap_accuracy
        if isinstance(holdout_accuracy, float)
        and isinstance(holdout_source_overlap_accuracy, float)
        else None
    )
    holdout_zero_nsi_accuracy = (
        _accuracy_from_diagnostics(zero_nsi_diagnostics, "effective")
        if zero_nsi_diagnostics
        else None
    )
    holdout_model_minus_zero_nsi = (
        holdout_accuracy - holdout_zero_nsi_accuracy
        if isinstance(holdout_accuracy, float)
        and isinstance(holdout_zero_nsi_accuracy, float)
        else None
    )

    data_hashes = data_health.get("effective_split_hashes")
    gate_hashes = pretrain_gate.get("effective_split_hashes")
    run_manifest = _get_dict(summary, "run_manifest")
    duration_seconds = run_manifest.get("duration_seconds")
    data_holdout_baseline = (
        _get_dict(_get_dict(_get_dict(data_health, "rollups"), "baselines"), "holdout")
        .get("source_overlap", {})
        .get("accuracy")
    )
    holdout_baseline_matches_data_health = (
        isinstance(data_holdout_baseline, (int, float))
        and isinstance(holdout_source_overlap_accuracy, float)
        and abs(float(data_holdout_baseline) - holdout_source_overlap_accuracy) < 1e-12
    )
    command_record_count = holdout_diagnostics.get("command_record_count")
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain_gate.get("passed") is True,
        "head_manifest_source_gates_passed": (
            not head_manifest
            or (
                head_manifest.get("source_data_health_passed") is True
                and head_manifest.get("source_pretrain_gate_passed") is True
            )
        ),
        "data_and_pretrain_hashes_match": bool(data_hashes)
        and bool(gate_hashes)
        and data_hashes == gate_hashes,
        "head_manifest_hashes_match_data_health": (
            not head_manifest
            or head_manifest.get("effective_split_hashes") == data_hashes
        ),
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "pairwise_disabled_for_phase2s_full": summary.get("use_pairwise_command_reranker")
        is False,
        "val_command_slot_accuracy_min": isinstance(val_command_slot_accuracy, float)
        and val_command_slot_accuracy >= min_val_command_slot_accuracy,
        "val_model_minus_source_overlap_min": isinstance(val_model_minus_source_overlap, float)
        and val_model_minus_source_overlap >= min_val_model_minus_source_overlap,
        "holdout_command_slot_accuracy_min": isinstance(holdout_accuracy, float)
        and holdout_accuracy >= min_holdout_command_slot_accuracy,
        "holdout_model_minus_source_overlap_min": isinstance(
            holdout_model_minus_source_overlap,
            float,
        )
        and holdout_model_minus_source_overlap >= min_holdout_model_minus_source_overlap,
        "holdout_zero_nsi_diagnostics_gate": (
            True
            if min_holdout_model_minus_zero_nsi is None
            else isinstance(holdout_model_minus_zero_nsi, float)
            and holdout_model_minus_zero_nsi >= min_holdout_model_minus_zero_nsi
        ),
        "holdout_baseline_matches_data_health": holdout_baseline_matches_data_health,
        "holdout_diagnostics_not_sealed_tuned": holdout_diagnostics.get(
            "sealed_data_used_for_training_or_tuning"
        )
        is False,
        "holdout_record_count_matches_data_health": isinstance(command_record_count, int)
        and command_record_count
        == _get_dict(_get_dict(data_health, "rollups"), "holdout").get("rows"),
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_package_phase2s_until_full_holdout_postflight_passes")
    if not checks["holdout_model_minus_source_overlap_min"]:
        blocked_actions.append("do_not_claim_phase2s_holdout_delta_from_source_overlap")
    if not checks["holdout_zero_nsi_diagnostics_gate"]:
        blocked_actions.append("do_not_claim_phase2s_holdout_nsi_identity_delta")
    if not checks["data_and_pretrain_hashes_match"] or not checks[
        "head_manifest_hashes_match_data_health"
    ]:
        blocked_actions.append("do_not_package_with_phase2s_hash_mismatch")
    return {
        "audit_family": "phase2s_open_repair_full_holdout_postflight",
        "passed": passed,
        "allowed_next_action": (
            "review_phase2s_boundary_before_package_or_sealed_final_eval"
            if passed
            else "freeze_phase2s_full_failure_and_analyze_nonsealed_design"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "metrics": {
            "val_command_slot_accuracy": val_command_slot_accuracy,
            "source_overlap_val_accuracy": (
                float(source_overlap_val_accuracy)
                if isinstance(source_overlap_val_accuracy, (int, float))
                else None
            ),
            "val_model_minus_source_overlap_accuracy": val_model_minus_source_overlap,
            "holdout_command_slot_accuracy": holdout_accuracy,
            "holdout_source_overlap_accuracy": holdout_source_overlap_accuracy,
            "holdout_model_minus_source_overlap_accuracy": holdout_model_minus_source_overlap,
            "holdout_zero_nsi_effective_accuracy": holdout_zero_nsi_accuracy,
            "holdout_model_minus_zero_nsi_accuracy": holdout_model_minus_zero_nsi,
            "duration_seconds": float(duration_seconds)
            if isinstance(duration_seconds, (int, float))
            else None,
            "holdout_command_record_count": command_record_count,
            "low_level_qwen_calls_target": summary.get("low_level_qwen_calls_target"),
            "use_pairwise_command_reranker": bool(
                summary.get("use_pairwise_command_reranker")
            ),
            "command_candidate_encoder": summary.get("command_candidate_encoder"),
        },
        "thresholds": {
            "min_val_command_slot_accuracy": min_val_command_slot_accuracy,
            "min_holdout_command_slot_accuracy": min_holdout_command_slot_accuracy,
            "min_val_model_minus_source_overlap": min_val_model_minus_source_overlap,
            "min_holdout_model_minus_source_overlap": min_holdout_model_minus_source_overlap,
            "min_holdout_model_minus_zero_nsi": min_holdout_model_minus_zero_nsi,
        },
        "effective_split_hashes": data_hashes,
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "data_health_json": str(Path(data_health_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "head_manifest_json": str(Path(head_manifest_json)) if head_manifest_json else None,
            "holdout_diagnostics_json": str(Path(holdout_diagnostics_json)),
            "holdout_zero_nsi_diagnostics_json": (
                str(Path(holdout_zero_nsi_diagnostics_json))
                if holdout_zero_nsi_diagnostics_json
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2S sandboxed open-repair data before any training."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    data = subparsers.add_parser("data-health")
    data.add_argument("--train-jsonl", required=True)
    data.add_argument("--val-jsonl", required=True)
    data.add_argument("--holdout-jsonl", required=True)
    data.add_argument("--dataset-root")
    data.add_argument("--output-json")
    data.add_argument("--min-train-rows", type=int, default=15)
    data.add_argument("--min-val-rows", type=int, default=15)
    data.add_argument("--min-holdout-rows", type=int, default=9)
    data.add_argument("--max-required-baseline-val-accuracy", type=float, default=0.75)
    data.add_argument("--max-repair-slot-share", type=float, default=0.60)
    data.add_argument("--no-fail", action="store_true")

    gate = subparsers.add_parser("pretrain-gate")
    gate.add_argument("--data-health-json", required=True)
    gate.add_argument("--output-json")
    gate.add_argument("--no-fail", action="store_true")

    postflight = subparsers.add_parser("smoke-postflight")
    postflight.add_argument("--training-summary-json", required=True)
    postflight.add_argument("--data-health-json", required=True)
    postflight.add_argument("--pretrain-gate-json", required=True)
    postflight.add_argument("--head-manifest-json")
    postflight.add_argument("--zero-nsi-diagnostics-json")
    postflight.add_argument("--output-json")
    postflight.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    postflight.add_argument("--min-model-minus-source-overlap", type=float, default=0.10)
    postflight.add_argument("--min-model-minus-zero-nsi", type=float)
    postflight.add_argument("--max-smoke-duration-seconds", type=float, default=3600.0)
    postflight.add_argument("--no-fail", action="store_true")

    holdout = subparsers.add_parser("full-holdout-postflight")
    holdout.add_argument("--training-summary-json", required=True)
    holdout.add_argument("--data-health-json", required=True)
    holdout.add_argument("--pretrain-gate-json", required=True)
    holdout.add_argument("--head-manifest-json")
    holdout.add_argument("--holdout-diagnostics-json", required=True)
    holdout.add_argument("--holdout-zero-nsi-diagnostics-json")
    holdout.add_argument("--output-json")
    holdout.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    holdout.add_argument("--min-holdout-command-slot-accuracy", type=float, default=0.85)
    holdout.add_argument("--min-val-model-minus-source-overlap", type=float, default=0.15)
    holdout.add_argument("--min-holdout-model-minus-source-overlap", type=float, default=0.15)
    holdout.add_argument("--min-holdout-model-minus-zero-nsi", type=float)
    holdout.add_argument("--no-fail", action="store_true")

    args = parser.parse_args()
    if args.command == "data-health":
        report = build_phase2s_data_health(
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            holdout_jsonl=args.holdout_jsonl,
            dataset_root=args.dataset_root,
            min_train_rows=args.min_train_rows,
            min_val_rows=args.min_val_rows,
            min_holdout_rows=args.min_holdout_rows,
            max_required_baseline_val_accuracy=args.max_required_baseline_val_accuracy,
            max_repair_slot_share=args.max_repair_slot_share,
        )
    elif args.command == "pretrain-gate":
        report = build_phase2s_pretrain_gate(data_health_json=args.data_health_json)
    elif args.command == "smoke-postflight":
        report = build_phase2s_smoke_postflight(
            training_summary_json=args.training_summary_json,
            data_health_json=args.data_health_json,
            pretrain_gate_json=args.pretrain_gate_json,
            head_manifest_json=args.head_manifest_json,
            zero_nsi_diagnostics_json=args.zero_nsi_diagnostics_json,
            min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
            min_model_minus_source_overlap=args.min_model_minus_source_overlap,
            min_model_minus_zero_nsi=args.min_model_minus_zero_nsi,
            max_smoke_duration_seconds=args.max_smoke_duration_seconds,
        )
    else:
        report = build_phase2s_full_holdout_postflight(
            training_summary_json=args.training_summary_json,
            data_health_json=args.data_health_json,
            pretrain_gate_json=args.pretrain_gate_json,
            head_manifest_json=args.head_manifest_json,
            holdout_diagnostics_json=args.holdout_diagnostics_json,
            holdout_zero_nsi_diagnostics_json=args.holdout_zero_nsi_diagnostics_json,
            min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
            min_holdout_command_slot_accuracy=args.min_holdout_command_slot_accuracy,
            min_val_model_minus_source_overlap=args.min_val_model_minus_source_overlap,
            min_holdout_model_minus_source_overlap=args.min_holdout_model_minus_source_overlap,
            min_holdout_model_minus_zero_nsi=args.min_holdout_model_minus_zero_nsi,
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
