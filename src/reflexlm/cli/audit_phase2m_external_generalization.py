from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import source_overlap_command_slot_prediction


ALLOWED_SOURCE_KINDS = {"public_repo", "synthetic_safe_repo"}
REQUIRED_BASELINES = {
    "source_overlap",
    "native_head_only",
    "continuation_only",
    "prompt_only",
    "react",
}
BASELINE_METHODS = {
    "source_overlap": "source_overlap_current_visible_v2",
    "native_head_only": "first_candidate_no_latent_v2",
    "continuation_only": "prior_summary_overlap_v2",
    "prompt_only": "current_plus_runtime_text_overlap_v2",
    "react": "runtime_evidence_overlap_v2",
}
REQUIRED_EVIDENCE_DENSITIES = {"low", "medium", "high"}
REQUIRED_CANDIDATE_COUNTS = {2, 3, 4}
REQUIRED_CONTINUATION_DEPTHS = {"one_step", "two_step", "stale_state_refresh"}
REQUIRED_AMBIGUITY_CLASSES = {
    "same_intent_command",
    "same_file_read",
    "stage_transition",
}
REQUIRED_TRACE_TYPES = {
    "test_failure_traceback_to_symbol",
    "changed_file_to_watched_test",
    "module_ownership_to_command",
    "stale_state_refresh",
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
    "correct_command_hint",
    "sealed_feedback",
)
CANDIDATE_SLOT_MARKER_RE = re.compile(r"(?i)\bcandidate[_-]?\d+\b")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[A-Za-z0-9_./+\-]{8,}"
)
ABSOLUTE_PATH_RE = re.compile(
    r"(?i)([A-Z]:\\|\\\\[A-Za-z0-9_.-]+\\|/Users/|/home/|/root/|/var/folders/)"
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


def _read_head_jsonl(path: str | Path | None) -> tuple[list[dict[str, Any]], bool]:
    return _read_jsonl(path)


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


def _candidate_commands(row: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for candidate in _get_list(row, "command_candidates"):
        if isinstance(candidate, str):
            commands.append(candidate)
        elif isinstance(candidate, dict) and candidate.get("command") is not None:
            commands.append(str(candidate["command"]))
    return commands


def _head_candidate_commands(row: dict[str, Any]) -> list[str]:
    value = row.get("candidate_commands")
    return [str(item) for item in value] if isinstance(value, list) else []


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if token}


def _overlap_prediction(text: str, commands: list[str]) -> str | None:
    if not commands:
        return None
    text_tokens = _tokens(text)
    scored = [
        (len(text_tokens & _tokens(command)), -index, command)
        for index, command in enumerate(commands)
    ]
    return max(scored)[2]


def compute_phase2m_baseline_predictions(row: dict[str, Any]) -> dict[str, str | None]:
    """Compute label-free Phase2M v2 baseline predictions from visible fields."""
    commands = _candidate_commands(row)
    runtime_evidence = _canonical_json(row.get("runtime_visible_evidence") or {})
    current_visible = str(row.get("current_visible_text") or "")
    prior_summary = str(_get_dict(row, "runtime_visible_evidence").get("prior_read_summary") or "")
    return {
        "source_overlap": _overlap_prediction(current_visible, commands),
        "native_head_only": commands[0] if commands else None,
        "continuation_only": _overlap_prediction(prior_summary, commands),
        "prompt_only": _overlap_prediction(f"{current_visible}\n{runtime_evidence}", commands),
        "react": _overlap_prediction(runtime_evidence, commands),
    }


def _candidate_intents(row: dict[str, Any]) -> list[str]:
    intents: list[str] = []
    for candidate in _get_list(row, "command_candidates"):
        if isinstance(candidate, dict) and candidate.get("intent") is not None:
            intents.append(str(candidate["intent"]))
    return intents


def _difficulty(row: dict[str, Any], key: str) -> Any:
    difficulty = _get_dict(row, "difficulty")
    return difficulty.get(key, row.get(key))


def _baseline_prediction(row: dict[str, Any], name: str) -> str | None:
    baselines = _get_dict(row, "baselines")
    value = baselines.get(name, row.get(f"{name}_prediction"))
    if isinstance(value, dict):
        value = value.get("predicted_command")
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
    return True


def _baselines_match_computed(row: dict[str, Any]) -> bool:
    computed = compute_phase2m_baseline_predictions(row)
    return all(_baseline_prediction(row, name) == computed[name] for name in REQUIRED_BASELINES)


def _text_for_redaction(row: dict[str, Any]) -> str:
    visible_payload = {
        "repo_url_or_origin": row.get("repo_url_or_origin"),
        "current_visible_text": row.get("current_visible_text"),
        "runtime_visible_evidence": row.get("runtime_visible_evidence"),
        "command_candidates": row.get("command_candidates"),
    }
    return _canonical_json(visible_payload)


def _row_mentions_forbidden_visible_marker(row: dict[str, Any]) -> bool:
    text = _text_for_redaction(row).lower()
    return any(marker in text for marker in HIDDEN_MARKERS)


def _row_mentions_candidate_slot_marker(row: dict[str, Any]) -> bool:
    text = _text_for_redaction(row)
    return bool(CANDIDATE_SLOT_MARKER_RE.search(text))


def _row_mentions_sealed(row: dict[str, Any]) -> bool:
    text = _canonical_json(row).replace("\\", "/").lower()
    return any(marker in text for marker in SEALED_MARKERS)


def _row_has_redaction_leak(row: dict[str, Any]) -> bool:
    text = _text_for_redaction(row)
    return bool(
        EMAIL_RE.search(text)
        or SECRET_ASSIGNMENT_RE.search(text)
        or ABSOLUTE_PATH_RE.search(text)
    )


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
    source_kind = str(row.get("source_kind") or "")
    return (
        source_kind in ALLOWED_SOURCE_KINDS
        and bool(row.get("repo_id"))
        and bool(row.get("repo_url_or_origin"))
        and len(str(row.get("commit_hash") or "")) >= 7
        and bool(row.get("license_or_synthetic_origin"))
        and len(str(row.get("collection_script_hash") or "")) >= 16
        and bool(row.get("trace_hash"))
    )


def _row_shape_ok(row: dict[str, Any]) -> bool:
    commands = _candidate_commands(row)
    expected = row.get("expected_command")
    difficulty_count = _difficulty(row, "candidate_count")
    try:
        declared_count = int(difficulty_count)
    except (TypeError, ValueError):
        declared_count = -1
    return (
        bool(row.get("trace_id"))
        and expected in commands
        and len(commands) >= 2
        and declared_count == len(commands)
        and len(set(commands)) == len(commands)
        and all(_baseline_prediction(row, baseline) in commands for baseline in REQUIRED_BASELINES)
    )


def _same_intent_competition_ok(row: dict[str, Any]) -> bool:
    if _difficulty(row, "ambiguity_class") != "same_intent_command":
        return True
    intents = _candidate_intents(row)
    return len(intents) == len(_candidate_commands(row)) and len(set(intents)) == 1


def _baseline_rollup(rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    total = 0
    correct = 0
    for row in rows:
        expected = row.get("expected_command")
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


def _expected_command_slot(row: dict[str, Any]) -> int | None:
    commands = _candidate_commands(row)
    expected = row.get("expected_command")
    if expected not in commands:
        return None
    return commands.index(expected)


def _command_slot_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    slots: Counter[int] = Counter()
    for row in rows:
        slot = _expected_command_slot(row)
        if slot is not None:
            slots[slot] += 1
    total = sum(slots.values())
    max_share = max((count / total for count in slots.values()), default=None)
    return {
        "total": total,
        "slots": {str(key): value for key, value in sorted(slots.items())},
        "max_share": max_share,
    }


def _split_hash(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    stable_rows = sorted(rows, key=lambda row: str(row.get("trace_id") or row.get("trace_hash")))
    return _sha256(stable_rows)


def _head_row_shape_ok(row: dict[str, Any]) -> bool:
    candidates = _head_candidate_commands(row)
    command_slot = row.get("command_slot")
    return (
        bool(row.get("example_id") or row.get("episode_id"))
        and bool(row.get("state_prompt"))
        and isinstance(command_slot, int)
        and 0 <= command_slot < len(candidates)
        and len(candidates) >= 2
        and len(set(candidates)) == len(candidates)
    )


def _head_source_overlap_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    predicted_slots: Counter[int] = Counter()
    expected_slots: Counter[int] = Counter()
    examples: list[dict[str, Any]] = []
    for row in rows:
        if not _head_row_shape_ok(row):
            continue
        candidates = _head_candidate_commands(row)
        expected_slot = int(row["command_slot"])
        predicted_slot = source_overlap_command_slot_prediction(
            str(row.get("state_prompt") or ""),
            candidates,
        )
        total += 1
        correct += int(predicted_slot == expected_slot)
        predicted_slots[predicted_slot] += 1
        expected_slots[expected_slot] += 1
        if predicted_slot == expected_slot and len(examples) < 12:
            examples.append(
                {
                    "example_id": str(row.get("example_id") or row.get("episode_id")),
                    "predicted_slot": predicted_slot,
                    "expected_slot": expected_slot,
                    "predicted_command": candidates[predicted_slot],
                }
            )
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "predicted_slots": {str(key): value for key, value in sorted(predicted_slots.items())},
        "expected_slots": {str(key): value for key, value in sorted(expected_slots.items())},
        "correct_examples": examples,
    }


def build_phase2m_head_state_baseline_audit(
    *,
    head_train_jsonl: str | Path,
    head_val_jsonl: str | Path,
    max_val_source_overlap_accuracy: float = 0.50,
    min_train_rows: int = 24,
    min_val_rows: int = 24,
) -> dict[str, Any]:
    """Measure lexical source-overlap on final native-head prompts before training.

    Raw Phase2M trace baselines are necessary but not sufficient. The head dataset
    adds runtime evidence and formatting that can make a lexical candidate baseline
    stronger than the raw-trace audit reports. This audit catches that shortcut
    before launching expensive 7B training.
    """

    train_rows, train_exists = _read_head_jsonl(head_train_jsonl)
    val_rows, val_exists = _read_head_jsonl(head_val_jsonl)
    train_rollup = _head_source_overlap_rollup(train_rows)
    val_rollup = _head_source_overlap_rollup(val_rows)
    checks = {
        "phase2m_head_train_jsonl_exists": train_exists,
        "phase2m_head_val_jsonl_exists": val_exists,
        "phase2m_head_train_rows_minimum_met": len(train_rows) >= min_train_rows,
        "phase2m_head_val_rows_minimum_met": len(val_rows) >= min_val_rows,
        "phase2m_head_rows_have_candidates_and_command_slots": bool(train_rows and val_rows)
        and all(_head_row_shape_ok(row) for row in train_rows + val_rows),
        "phase2m_head_state_source_overlap_val_below_threshold": (
            isinstance(val_rollup["accuracy"], float)
            and val_rollup["accuracy"] <= max_val_source_overlap_accuracy
        ),
    }
    blocked_actions: list[str] = []
    if not all(checks.values()):
        blocked_actions.append("do_not_train_phase2m_until_head_state_baseline_passes")
    if not checks["phase2m_head_state_source_overlap_val_below_threshold"]:
        blocked_actions.append("do_not_train_when_head_state_source_overlap_solves_phase2m_val")
    passed = all(checks.values())
    return {
        "audit_family": "phase2m_head_state_baseline_audit",
        "passed": passed,
        "allowed_next_action": (
            "run_phase2m_smoke_training_only"
            if passed
            else "revise_phase2m_head_prompt_or_trace_design_before_training"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "thresholds": {
            "max_val_source_overlap_accuracy": max_val_source_overlap_accuracy,
            "min_train_rows": min_train_rows,
            "min_val_rows": min_val_rows,
        },
        "rollups": {
            "source_overlap": {
                "train": train_rollup,
                "val": val_rollup,
            }
        },
        "effective_split_hashes": {
            "phase2m_head_train": _split_hash(train_rows),
            "phase2m_head_val": _split_hash(val_rows),
        },
        "inputs": {
            "head_train_jsonl": str(Path(head_train_jsonl)),
            "head_val_jsonl": str(Path(head_val_jsonl)),
        },
    }


def _rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "repo_ids": sorted({str(row.get("repo_id")) for row in rows if row.get("repo_id")}),
        "source_kinds": sorted(
            {str(row.get("source_kind")) for row in rows if row.get("source_kind")}
        ),
        "evidence_densities": sorted(
            {str(_difficulty(row, "evidence_density")) for row in rows if _difficulty(row, "evidence_density")}
        ),
        "candidate_counts": sorted(
            {
                int(_difficulty(row, "candidate_count"))
                for row in rows
                if _difficulty(row, "candidate_count") is not None
            }
        ),
        "continuation_depths": sorted(
            {str(_difficulty(row, "continuation_depth")) for row in rows if _difficulty(row, "continuation_depth")}
        ),
        "ambiguity_classes": sorted(
            {str(_difficulty(row, "ambiguity_class")) for row in rows if _difficulty(row, "ambiguity_class")}
        ),
        "trace_types": sorted(
            {str(_difficulty(row, "trace_type")) for row in rows if _difficulty(row, "trace_type")}
        ),
    }


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


def _repo_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("repo_id")) for row in rows if row.get("repo_id")}


def _split_rows_have_split(rows: list[dict[str, Any]], split: str) -> bool:
    return bool(rows) and all(str(row.get("split")) == split for row in rows)


def build_phase2m_data_health(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    min_train_rows: int = 24,
    min_val_rows: int = 24,
    min_holdout_rows: int = 12,
    max_source_overlap_val_accuracy: float = 0.75,
    max_native_head_only_val_accuracy: float = 0.75,
    require_measured_baselines: bool = False,
    require_computed_baselines_match: bool = False,
    forbid_candidate_slot_markers: bool = False,
    require_all_baselines_below_threshold: bool = False,
    max_required_baseline_val_accuracy: float = 0.75,
    max_command_slot_share: float | None = None,
) -> dict[str, Any]:
    train_rows, train_exists = _read_jsonl(train_jsonl)
    val_rows, val_exists = _read_jsonl(val_jsonl)
    holdout_rows, holdout_exists = _read_jsonl(holdout_jsonl)
    all_rows = train_rows + val_rows + holdout_rows
    val_rollup = _rollup(val_rows)
    baselines = {
        "train": {name: _baseline_rollup(train_rows, name) for name in REQUIRED_BASELINES},
        "val": {name: _baseline_rollup(val_rows, name) for name in REQUIRED_BASELINES},
        "holdout": {
            name: _baseline_rollup(holdout_rows, name) for name in REQUIRED_BASELINES
        },
    }
    source_overlap_val_accuracy = baselines["val"]["source_overlap"]["accuracy"]
    native_head_only_val_accuracy = baselines["val"]["native_head_only"]["accuracy"]
    required_baseline_val_accuracies = {
        name: baselines["val"][name]["accuracy"] for name in REQUIRED_BASELINES
    }
    command_slots = {
        "train": _command_slot_rollup(train_rows),
        "val": _command_slot_rollup(val_rows),
        "holdout": _command_slot_rollup(holdout_rows),
    }
    holdout_repos = _repo_ids(holdout_rows)
    train_val_repos = _repo_ids(train_rows) | _repo_ids(val_rows)

    checks = {
        "phase2m_train_jsonl_exists": train_exists,
        "phase2m_val_jsonl_exists": val_exists,
        "phase2m_holdout_jsonl_exists": holdout_exists,
        "phase2m_train_rows_minimum_met": len(train_rows) >= min_train_rows,
        "phase2m_val_rows_minimum_met": len(val_rows) >= min_val_rows,
        "phase2m_holdout_rows_minimum_met": len(holdout_rows) >= min_holdout_rows,
        "phase2m_split_labels_match_files": (
            _split_rows_have_split(train_rows, "train")
            and _split_rows_have_split(val_rows, "val")
            and _split_rows_have_split(holdout_rows, "holdout")
        ),
        "phase2m_provenance_and_license_present": bool(all_rows)
        and all(_provenance_ok(row) for row in all_rows),
        "phase2m_normalization_and_redaction_flags_present": bool(all_rows)
        and all(_normalization_ok(row) for row in all_rows),
        "phase2m_no_redaction_leaks_visible": not any(
            _row_has_redaction_leak(row) for row in all_rows
        ),
        "phase2m_no_hidden_gold_or_sealed_visible": not any(
            _row_mentions_forbidden_visible_marker(row) for row in all_rows
        ),
        "phase2m_no_candidate_slot_marker_visible": (
            True
            if not forbid_candidate_slot_markers
            else bool(all_rows)
            and not any(_row_mentions_candidate_slot_marker(row) for row in all_rows)
        ),
        "phase2m_no_sealed_reference_anywhere": not any(
            _row_mentions_sealed(row) for row in all_rows
        ),
        "phase2m_rows_have_candidates_labels_and_baselines": bool(all_rows)
        and all(_row_shape_ok(row) for row in all_rows),
        "phase2m_same_intent_competition_valid": bool(all_rows)
        and all(_same_intent_competition_ok(row) for row in all_rows),
        "phase2m_repo_disjoint_holdout": bool(holdout_repos)
        and holdout_repos.isdisjoint(train_val_repos),
        "phase2m_deduplicated_by_repo_commit_trace_hash": bool(all_rows)
        and _dedup_ok(all_rows),
        "phase2m_evidence_density_coverage": REQUIRED_EVIDENCE_DENSITIES.issubset(
            set(val_rollup["evidence_densities"])
        ),
        "phase2m_candidate_count_coverage": REQUIRED_CANDIDATE_COUNTS.issubset(
            set(val_rollup["candidate_counts"])
        ),
        "phase2m_continuation_depth_coverage": REQUIRED_CONTINUATION_DEPTHS.issubset(
            set(val_rollup["continuation_depths"])
        ),
        "phase2m_ambiguity_class_coverage": REQUIRED_AMBIGUITY_CLASSES.issubset(
            set(val_rollup["ambiguity_classes"])
        ),
        "phase2m_trace_type_coverage": REQUIRED_TRACE_TYPES.issubset(
            set(val_rollup["trace_types"])
        ),
        "phase2m_all_required_baselines_measured": all(
            baselines["val"][name]["total"] == len(val_rows) and len(val_rows) > 0
            for name in REQUIRED_BASELINES
        ),
        "phase2m_baseline_metadata_measured": (
            True
            if not require_measured_baselines
            else bool(all_rows)
            and all(_baseline_metadata_ok(row) for row in all_rows)
        ),
        "phase2m_baselines_match_computed_predictions": (
            True
            if not require_computed_baselines_match
            else bool(all_rows)
            and all(_baselines_match_computed(row) for row in all_rows)
        ),
        "phase2m_source_overlap_val_below_threshold": (
            isinstance(source_overlap_val_accuracy, float)
            and source_overlap_val_accuracy <= max_source_overlap_val_accuracy
        ),
        "phase2m_native_head_only_val_below_threshold": (
            isinstance(native_head_only_val_accuracy, float)
            and native_head_only_val_accuracy <= max_native_head_only_val_accuracy
        ),
        "phase2m_all_required_baselines_val_below_threshold": (
            True
            if not require_all_baselines_below_threshold
            else all(
                isinstance(value, float) and value <= max_required_baseline_val_accuracy
                for value in required_baseline_val_accuracies.values()
            )
        ),
        "phase2m_command_slot_share_below_threshold": (
            True
            if max_command_slot_share is None
            else all(
                isinstance(command_slots[split]["max_share"], float)
                and command_slots[split]["max_share"] <= max_command_slot_share
                for split in ("train", "val")
            )
        ),
    }

    blocked_actions: list[str] = []
    if not all(checks.values()):
        blocked_actions.append("do_not_train_phase2m_until_data_health_passes")
    if not checks["phase2m_no_sealed_reference_anywhere"]:
        blocked_actions.append("do_not_use_sealed_or_sealed_failure_feedback")
    if not checks["phase2m_no_candidate_slot_marker_visible"]:
        blocked_actions.append("do_not_train_with_candidate_slot_markers_visible")
    if not checks["phase2m_no_redaction_leaks_visible"]:
        blocked_actions.append("do_not_train_with_unredacted_phase2m_traces")
    if not checks["phase2m_baseline_metadata_measured"]:
        blocked_actions.append("measure_phase2m_baselines_with_code_before_training")
    if not checks["phase2m_baselines_match_computed_predictions"]:
        blocked_actions.append("do_not_train_with_declared_or_stale_phase2m_baselines")
    if not checks["phase2m_repo_disjoint_holdout"]:
        blocked_actions.append("do_not_train_without_repo_disjoint_holdout")
    if not checks["phase2m_source_overlap_val_below_threshold"]:
        blocked_actions.append("do_not_train_when_source_overlap_solves_phase2m_val")
    if not checks["phase2m_native_head_only_val_below_threshold"]:
        blocked_actions.append("do_not_train_when_native_head_only_solves_phase2m_val")
    if not checks["phase2m_all_required_baselines_val_below_threshold"]:
        blocked_actions.append("do_not_train_when_any_required_baseline_solves_phase2m_val")
    if not checks["phase2m_command_slot_share_below_threshold"]:
        blocked_actions.append("do_not_train_with_phase2m_command_slot_imbalance")

    passed = all(checks.values())
    return {
        "audit_family": "phase2m_external_generalization_data_health",
        "passed": passed,
        "allowed_next_action": (
            "run_phase2m_smoke_training_only"
            if passed
            else "collect_or_revise_phase2m_readonly_traces_before_training"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "thresholds": {
            "min_train_rows": min_train_rows,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "max_source_overlap_val_accuracy": max_source_overlap_val_accuracy,
            "max_native_head_only_val_accuracy": max_native_head_only_val_accuracy,
            "required_evidence_densities": sorted(REQUIRED_EVIDENCE_DENSITIES),
            "required_candidate_counts": sorted(REQUIRED_CANDIDATE_COUNTS),
            "required_continuation_depths": sorted(REQUIRED_CONTINUATION_DEPTHS),
            "required_ambiguity_classes": sorted(REQUIRED_AMBIGUITY_CLASSES),
            "required_trace_types": sorted(REQUIRED_TRACE_TYPES),
            "required_baselines": sorted(REQUIRED_BASELINES),
            "baseline_methods": BASELINE_METHODS,
            "require_measured_baselines": require_measured_baselines,
            "require_computed_baselines_match": require_computed_baselines_match,
            "forbid_candidate_slot_markers": forbid_candidate_slot_markers,
            "require_all_baselines_below_threshold": require_all_baselines_below_threshold,
            "max_required_baseline_val_accuracy": max_required_baseline_val_accuracy,
            "max_command_slot_share": max_command_slot_share,
        },
        "rollups": {
            "train": _rollup(train_rows),
            "val": val_rollup,
            "holdout": _rollup(holdout_rows),
            "baselines": baselines,
            "command_slots": command_slots,
            "duplicate_repo_commit_trace_keys": [
                "|".join(key)
                for key, count in Counter(
                    (
                        str(row.get("repo_id")),
                        str(row.get("commit_hash")),
                        str(row.get("trace_hash")),
                    )
                    for row in all_rows
                ).items()
                if count > 1
            ][:16],
        },
        "effective_split_hashes": {
            "phase2m_train": _split_hash(train_rows),
            "phase2m_val": _split_hash(val_rows),
            "phase2m_holdout": _split_hash(holdout_rows),
        },
        "inputs": {
            "train_jsonl": str(Path(train_jsonl)),
            "val_jsonl": str(Path(val_jsonl)),
            "holdout_jsonl": str(Path(holdout_jsonl)),
        },
    }


def build_phase2m_pretrain_gate(
    *,
    data_health_json: str | Path,
    head_state_baseline_json: str | Path | None = None,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    head_state_baseline = _read_json(head_state_baseline_json) if head_state_baseline_json else {}
    split_hashes = data_health.get("effective_split_hashes", {})
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "effective_split_hashes_present": all(
            split_hashes.get(key)
            for key in ("phase2m_train", "phase2m_val", "phase2m_holdout")
        ),
        "source_overlap_below_threshold": data_health.get("checks", {}).get(
            "phase2m_source_overlap_val_below_threshold"
        )
        is True,
        "native_head_only_below_threshold": data_health.get("checks", {}).get(
            "phase2m_native_head_only_val_below_threshold"
        )
        is True,
        "repo_disjoint_holdout": data_health.get("checks", {}).get(
            "phase2m_repo_disjoint_holdout"
        )
        is True,
        "sealed_not_used": data_health.get("checks", {}).get(
            "phase2m_no_sealed_reference_anywhere"
        )
        is True,
        "head_state_baseline_passed": (
            True if not head_state_baseline else head_state_baseline.get("passed") is True
        ),
        "head_state_source_overlap_below_threshold": (
            True
            if not head_state_baseline
            else head_state_baseline.get("checks", {}).get(
                "phase2m_head_state_source_overlap_val_below_threshold"
            )
            is True
        ),
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_train_phase2m_until_pretrain_gate_passes")
    if not checks["head_state_source_overlap_below_threshold"]:
        blocked_actions.append("do_not_train_when_head_state_source_overlap_solves_phase2m_val")
    return {
        "audit_family": "phase2m_external_generalization_pretrain_gate",
        "passed": passed,
        "allowed_next_action": (
            "run_phase2m_smoke_training_only"
            if passed
            else "collect_or_revise_phase2m_readonly_traces_before_training"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "effective_split_hashes": split_hashes,
        "head_state_baseline": (
            {
                "passed": head_state_baseline.get("passed"),
                "val_source_overlap_accuracy": head_state_baseline.get("rollups", {})
                .get("source_overlap", {})
                .get("val", {})
                .get("accuracy"),
                "effective_split_hashes": head_state_baseline.get("effective_split_hashes"),
            }
            if head_state_baseline
            else None
        ),
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "head_state_baseline_json": (
                str(Path(head_state_baseline_json)) if head_state_baseline_json else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2M external-generalization trace data before training."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data-health")
    data.add_argument("--train-jsonl", required=True)
    data.add_argument("--val-jsonl", required=True)
    data.add_argument("--holdout-jsonl", required=True)
    data.add_argument("--output-json")
    data.add_argument("--min-train-rows", type=int, default=24)
    data.add_argument("--min-val-rows", type=int, default=24)
    data.add_argument("--min-holdout-rows", type=int, default=12)
    data.add_argument("--max-source-overlap-val-accuracy", type=float, default=0.75)
    data.add_argument("--max-native-head-only-val-accuracy", type=float, default=0.75)
    data.add_argument("--require-measured-baselines", action="store_true")
    data.add_argument("--require-computed-baselines-match", action="store_true")
    data.add_argument("--forbid-candidate-slot-markers", action="store_true")
    data.add_argument("--require-all-baselines-below-threshold", action="store_true")
    data.add_argument("--max-required-baseline-val-accuracy", type=float, default=0.75)
    data.add_argument("--max-command-slot-share", type=float)
    data.add_argument("--no-fail", action="store_true")
    head = sub.add_parser("head-state-baseline")
    head.add_argument("--head-train-jsonl", required=True)
    head.add_argument("--head-val-jsonl", required=True)
    head.add_argument("--output-json")
    head.add_argument("--max-val-source-overlap-accuracy", type=float, default=0.50)
    head.add_argument("--min-train-rows", type=int, default=24)
    head.add_argument("--min-val-rows", type=int, default=24)
    head.add_argument("--no-fail", action="store_true")
    gate = sub.add_parser("pretrain-gate")
    gate.add_argument("--data-health-json", required=True)
    gate.add_argument("--head-state-baseline-json")
    gate.add_argument("--output-json")
    gate.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    if args.command == "data-health":
        report = build_phase2m_data_health(
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            holdout_jsonl=args.holdout_jsonl,
            min_train_rows=args.min_train_rows,
            min_val_rows=args.min_val_rows,
            min_holdout_rows=args.min_holdout_rows,
            max_source_overlap_val_accuracy=args.max_source_overlap_val_accuracy,
            max_native_head_only_val_accuracy=args.max_native_head_only_val_accuracy,
            require_measured_baselines=args.require_measured_baselines,
            require_computed_baselines_match=args.require_computed_baselines_match,
            forbid_candidate_slot_markers=args.forbid_candidate_slot_markers,
            require_all_baselines_below_threshold=args.require_all_baselines_below_threshold,
            max_required_baseline_val_accuracy=args.max_required_baseline_val_accuracy,
            max_command_slot_share=args.max_command_slot_share,
        )
    elif args.command == "head-state-baseline":
        report = build_phase2m_head_state_baseline_audit(
            head_train_jsonl=args.head_train_jsonl,
            head_val_jsonl=args.head_val_jsonl,
            max_val_source_overlap_accuracy=args.max_val_source_overlap_accuracy,
            min_train_rows=args.min_train_rows,
            min_val_rows=args.min_val_rows,
        )
    else:
        report = build_phase2m_pretrain_gate(
            data_health_json=args.data_health_json,
            head_state_baseline_json=args.head_state_baseline_json,
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
