from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2d_gates import _metric, _trace_audit
from reflexlm.llm.candidate_features import source_overlap_command_slot_prediction
from reflexlm.llm.head_dataset import build_phase2c_head_state_prompt_from_state
from reflexlm.models.features import candidate_commands, serialize_state_as_text
from reflexlm.schema import ActionType, SystemStateFrame


REQUIRED_EVIDENCE_DENSITIES = {"low", "medium", "high"}
REQUIRED_CANDIDATE_COUNTS = {2, 3, 4}
REQUIRED_CONTINUATION_DEPTHS = {"one_step", "two_step", "stale_state_refresh"}
REQUIRED_AMBIGUITY_CLASSES = {"same_intent_command", "same_file_read", "stage_transition"}


def _load_json(path: str | Path | None) -> Any:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _action_type(row: dict[str, Any]) -> str | None:
    action = row.get("action")
    if not isinstance(action, dict):
        return None
    return str(action.get("type"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _group_by_episode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("episode_id"))].append(row)
    return {key: sorted(value, key=lambda item: int(item.get("t", 0))) for key, value in grouped.items()}


def _command_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if _action_type(row) == ActionType.RUN_COMMAND.value:
            return row
    return None


def _source_overlap_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    examples: list[dict[str, Any]] = []
    for row in rows:
        if _action_type(row) != ActionType.RUN_COMMAND.value:
            continue
        state = SystemStateFrame.model_validate(row["state"])
        candidates = candidate_commands(state)
        expected = row.get("action", {}).get("command")
        if expected not in candidates:
            continue
        total += 1
        prediction = source_overlap_command_slot_prediction(
            build_phase2c_head_state_prompt_from_state(state),
            candidates,
        )
        expected_slot = candidates.index(expected)
        correct += int(prediction == expected_slot)
        examples.append(
            {
                "episode_id": row.get("episode_id"),
                "candidate_count": len(candidates),
                "expected_slot": expected_slot,
                "source_overlap_prediction": prediction,
                "source_overlap_correct": prediction == expected_slot,
                "last_command_visible": bool(state.terminal.last_command),
            }
        )
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total if total else None),
        "examples": examples[:8],
    }


def _metadata_rollup(metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(metadata_rows),
        "pair_ids": sorted(
            {str(row.get("phase2l_pair_id")) for row in metadata_rows if row.get("phase2l_pair_id")}
        ),
        "evidence_densities": sorted(
            {
                str(row.get("phase2l_evidence_density"))
                for row in metadata_rows
                if row.get("phase2l_evidence_density")
            }
        ),
        "candidate_counts": sorted(
            {
                int(row.get("phase2l_candidate_count"))
                for row in metadata_rows
                if row.get("phase2l_candidate_count") is not None
            }
        ),
        "continuation_depths": sorted(
            {
                str(row.get("phase2l_continuation_depth"))
                for row in metadata_rows
                if row.get("phase2l_continuation_depth")
            }
        ),
        "ambiguity_classes": sorted(
            {
                str(row.get("phase2l_ambiguity_class"))
                for row in metadata_rows
                if row.get("phase2l_ambiguity_class")
            }
        ),
    }


def _pair_rollup(
    rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_episode = _group_by_episode(rows)
    pairs: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "members": defaultdict(list),
            "current_visible_hashes": set(),
            "prior_context_hashes": set(),
            "candidate_hashes": set(),
            "expected_commands": set(),
            "member_slots": defaultdict(set),
            "last_command_visible_at_command": [],
        }
    )
    for metadata in metadata_rows:
        if not metadata.get("phase2l_counterfactual_continuation"):
            continue
        episode_id = str(metadata.get("episode_id"))
        episode_rows = by_episode.get(episode_id, [])
        if not episode_rows:
            continue
        command = _command_row(episode_rows)
        if command is None:
            continue
        current_state = SystemStateFrame.model_validate(command["state"])
        prior_state = SystemStateFrame.model_validate(episode_rows[0]["state"])
        candidates = candidate_commands(current_state)
        expected = command.get("action", {}).get("command")
        if expected not in candidates:
            continue
        pair_id = str(metadata.get("phase2l_pair_id"))
        member = str(metadata.get("phase2l_pair_member"))
        current_hash = _sha256(build_phase2c_head_state_prompt_from_state(current_state))
        prior_hash = _sha256(build_phase2c_head_state_prompt_from_state(prior_state))
        candidate_hash = _sha256(json.dumps(candidates, sort_keys=True))
        pair = pairs[pair_id]
        pair["members"][member].append(episode_id)
        pair["current_visible_hashes"].add(current_hash)
        pair["prior_context_hashes"].add(prior_hash)
        pair["candidate_hashes"].add(candidate_hash)
        pair["expected_commands"].add(str(expected))
        pair["member_slots"][member].add(candidates.index(expected))
        pair["last_command_visible_at_command"].append(bool(current_state.terminal.last_command))

    details: dict[str, Any] = {}
    for pair_id, pair in pairs.items():
        members = {key: list(value) for key, value in pair["members"].items()}
        member_slots = {key: sorted(value) for key, value in pair["member_slots"].items()}
        details[pair_id] = {
            "members": members,
            "member_slots": member_slots,
            "current_visible_hash_count": len(pair["current_visible_hashes"]),
            "prior_context_hash_count": len(pair["prior_context_hashes"]),
            "candidate_hash_count": len(pair["candidate_hashes"]),
            "expected_command_count": len(pair["expected_commands"]),
            "last_command_visible_at_command": any(pair["last_command_visible_at_command"]),
            "passed": (
                set(members) == {"a", "b"}
                and len(pair["current_visible_hashes"]) == 1
                and len(pair["prior_context_hashes"]) >= 2
                and len(pair["candidate_hashes"]) == 1
                and len(pair["expected_commands"]) >= 2
                and all(len(slots) == 1 for slots in member_slots.values())
                and len({tuple(slots) for slots in member_slots.values()}) >= 2
                and not any(pair["last_command_visible_at_command"])
            ),
        }
    return {
        "pair_count": len(details),
        "passed_pair_count": sum(1 for item in details.values() if item["passed"]),
        "all_pairs_complete": all(item["passed"] for item in details.values()) if details else False,
        "details": details,
    }


def build_phase2l_data_health(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    train_metadata_json: str | Path,
    val_metadata_json: str | Path,
    max_source_overlap_val_accuracy: float = 0.60,
) -> dict[str, Any]:
    train_rows = _load_jsonl(train_jsonl)
    val_rows = _load_jsonl(val_jsonl)
    train_metadata = _load_json(train_metadata_json)
    val_metadata = _load_json(val_metadata_json)
    train_meta_rows = train_metadata if isinstance(train_metadata, list) else []
    val_meta_rows = val_metadata if isinstance(val_metadata, list) else []
    train_rollup = _metadata_rollup(train_meta_rows)
    val_rollup = _metadata_rollup(val_meta_rows)
    source_overlap = {
        "train": _source_overlap_rollup(train_rows),
        "val": _source_overlap_rollup(val_rows),
    }
    pair_rollup = {
        "train": _pair_rollup(train_rows, train_meta_rows),
        "val": _pair_rollup(val_rows, val_meta_rows),
    }
    serialized = "\n".join(
        serialize_state_as_text(SystemStateFrame.model_validate(row["state"]))
        for row in train_rows + val_rows
        if isinstance(row.get("state"), dict)
    )
    val_accuracy = source_overlap["val"]["accuracy"]
    checks = {
        "phase2l_train_rows_present": len(train_rows) > 0,
        "phase2l_val_rows_present": len(val_rows) > 0,
        "phase2l_train_metadata_present": len(train_meta_rows) > 0,
        "phase2l_val_metadata_present": len(val_meta_rows) > 0,
        "phase2l_profiles_nonsealed": "external_trace_v3_semantic_required" not in serialized,
        "phase2l_no_hidden_gold_or_sealed_visible": (
            "recovery_hint=" not in serialized
            and "correct_command" not in serialized
            and "scenario_template" not in serialized
            and "sealed" not in serialized.lower()
            and "gold" not in serialized.lower()
        ),
        "phase2l_train_counterfactual_pairs_complete": pair_rollup["train"]["all_pairs_complete"],
        "phase2l_val_counterfactual_pairs_complete": pair_rollup["val"]["all_pairs_complete"],
        "phase2l_current_visible_state_hash_equal": all(
            item["current_visible_hash_count"] == 1
            for item in pair_rollup["val"]["details"].values()
        ),
        "phase2l_prior_context_hash_different": all(
            item["prior_context_hash_count"] >= 2
            for item in pair_rollup["val"]["details"].values()
        ),
        "phase2l_correct_command_differs_by_pair": all(
            item["expected_command_count"] >= 2
            for item in pair_rollup["val"]["details"].values()
        ),
        "phase2l_command_state_last_command_cleared": all(
            not item["last_command_visible_at_command"]
            for item in pair_rollup["val"]["details"].values()
        ),
        "phase2l_evidence_density_coverage": REQUIRED_EVIDENCE_DENSITIES.issubset(
            set(val_rollup["evidence_densities"])
        ),
        "phase2l_candidate_count_coverage": REQUIRED_CANDIDATE_COUNTS.issubset(
            set(val_rollup["candidate_counts"])
        ),
        "phase2l_continuation_depth_coverage": REQUIRED_CONTINUATION_DEPTHS.issubset(
            set(val_rollup["continuation_depths"])
        ),
        "phase2l_ambiguity_class_coverage": REQUIRED_AMBIGUITY_CLASSES.issubset(
            set(val_rollup["ambiguity_classes"])
        ),
        "phase2l_source_overlap_baseline_recorded": source_overlap["val"]["total"] > 0,
        "phase2l_source_overlap_baseline_below_threshold": (
            isinstance(val_accuracy, float) and val_accuracy <= max_source_overlap_val_accuracy
        ),
    }
    blocked_actions: list[str] = []
    if not all(checks.values()):
        blocked_actions.append("do_not_train_phase2l_until_data_health_passes")
    if not checks["phase2l_source_overlap_baseline_below_threshold"]:
        blocked_actions.append("do_not_train_when_source_overlap_solves_phase2l_val")
    if not checks["phase2l_val_counterfactual_pairs_complete"]:
        blocked_actions.append("do_not_train_without_complete_counterfactual_pairs")
    return {
        "audit_family": "phase2l_counterfactual_continuation_data_health",
        "passed": all(checks.values()),
        "allowed_next_action": (
            "run_phase2l_smoke_training_only"
            if all(checks.values())
            else "revise_phase2l_data_before_training"
        ),
        "blocked_actions": blocked_actions,
        "checks": checks,
        "thresholds": {
            "max_source_overlap_val_accuracy": max_source_overlap_val_accuracy,
            "required_evidence_densities": sorted(REQUIRED_EVIDENCE_DENSITIES),
            "required_candidate_counts": sorted(REQUIRED_CANDIDATE_COUNTS),
            "required_continuation_depths": sorted(REQUIRED_CONTINUATION_DEPTHS),
            "required_ambiguity_classes": sorted(REQUIRED_AMBIGUITY_CLASSES),
        },
        "rollups": {
            "train": train_rollup,
            "val": val_rollup,
            "source_overlap": source_overlap,
            "pairs": pair_rollup,
        },
        "inputs": {
            "train_jsonl": str(Path(train_jsonl)),
            "val_jsonl": str(Path(val_jsonl)),
            "train_metadata_json": str(Path(train_metadata_json)),
            "val_metadata_json": str(Path(val_metadata_json)),
        },
    }


def build_phase2l_pretrain_gate(*, data_health_json: str | Path) -> dict[str, Any]:
    data_health = _load_json(data_health_json)
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "counterfactual_pairs_complete": (
            data_health.get("checks", {}).get("phase2l_val_counterfactual_pairs_complete")
            is True
        ),
        "source_overlap_below_threshold": (
            data_health.get("checks", {}).get("phase2l_source_overlap_baseline_below_threshold")
            is True
        ),
        "sealed_v3_not_used_for_pretrain_gate": "external_trace_v3_semantic_required"
        not in json.dumps(data_health.get("inputs", {}), sort_keys=True),
    }
    passed = all(checks.values())
    return {
        "audit_family": "phase2l_counterfactual_continuation_pretrain_gate",
        "passed": passed,
        "allowed_next_action": (
            "run_phase2l_smoke_training_only"
            if passed
            else "revise_phase2l_data_before_training"
        ),
        "blocked_actions": [] if passed else ["do_not_train_phase2l_until_pretrain_gate_passes"],
        "checks": checks,
        "inputs": {"data_health_json": str(Path(data_health_json))},
    }


def _control(payload: dict[str, Any], name: str) -> bool:
    policy = payload.get("policy", {}) if isinstance(payload, dict) else {}
    if name == "wrong_cache":
        return policy.get("continuation_control") == "wrong_cache"
    if name == "cache_erased":
        return (
            policy.get("continuation_control") == "cache_erased"
            or policy.get("continuation_cache_enabled") is False
        )
    return False


def build_phase2l_postflight(
    *,
    data_health_json: str | Path,
    full_eval_json: str | Path,
    native_head_only_eval_json: str | Path,
    wrong_cache_eval_json: str | Path,
    cache_erased_eval_json: str | Path,
    min_full_completion: float = 0.85,
    min_full_minus_native_head_only: float = 0.15,
    min_full_minus_wrong_cache: float = 0.25,
    min_full_minus_cache_erased: float = 0.25,
    postflight_stage: str = "full",
) -> dict[str, Any]:
    if postflight_stage not in {"smoke", "local_full512", "full"}:
        raise ValueError("postflight_stage must be 'smoke', 'local_full512', or 'full'")
    data_health = _load_json(data_health_json)
    full = _load_json(full_eval_json)
    native = _load_json(native_head_only_eval_json)
    wrong_cache = _load_json(wrong_cache_eval_json)
    cache_erased = _load_json(cache_erased_eval_json)
    full_completion = _metric(full, "task_completion_rate")
    native_completion = _metric(native, "task_completion_rate")
    wrong_cache_completion = _metric(wrong_cache, "task_completion_rate")
    cache_erased_completion = _metric(cache_erased, "task_completion_rate")

    def _delta(other: float | None) -> float | None:
        if isinstance(full_completion, float) and isinstance(other, float):
            return full_completion - other
        return None

    full_minus_native = _delta(native_completion)
    full_minus_wrong_cache = _delta(wrong_cache_completion)
    full_minus_cache_erased = _delta(cache_erased_completion)
    full_trace = _trace_audit(full)
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "full_completion_gate_passed": (
            isinstance(full_completion, float) and full_completion >= min_full_completion
        ),
        "native_head_only_baseline_measured": isinstance(native_completion, float),
        "wrong_cache_control_recorded": _control(wrong_cache, "wrong_cache"),
        "cache_erased_control_recorded": _control(cache_erased, "cache_erased"),
        "full_beats_native_head_only_by_required_delta": (
            isinstance(full_minus_native, float)
            and full_minus_native >= min_full_minus_native_head_only
        ),
        "full_beats_wrong_cache_by_required_delta": (
            isinstance(full_minus_wrong_cache, float)
            and full_minus_wrong_cache >= min_full_minus_wrong_cache
        ),
        "full_beats_cache_erased_by_required_delta": (
            isinstance(full_minus_cache_erased, float)
            and full_minus_cache_erased >= min_full_minus_cache_erased
        ),
        "full_low_level_qwen_calls_zero": full_trace["low_level_qwen_calls"] == 0,
        "source_overlap_baseline_below_threshold": data_health.get("checks", {}).get(
            "phase2l_source_overlap_baseline_below_threshold"
        )
        is True,
        "sealed_v3_not_used_for_postflight": "external_trace_v3_semantic_required"
        not in json.dumps(
            {
                "data_health": data_health.get("inputs", {}),
                "full": full.get("dataset", {}),
                "native": native.get("dataset", {}),
                "wrong_cache": wrong_cache.get("dataset", {}),
                "cache_erased": cache_erased.get("dataset", {}),
            },
            sort_keys=True,
        ),
    }
    blocked_actions: list[str] = []
    if not checks["full_completion_gate_passed"]:
        blocked_actions.append("do_not_package_until_phase2l_val_gate_passes")
    if not checks["full_beats_native_head_only_by_required_delta"]:
        blocked_actions.append("do_not_package_without_full_beating_native_head_only")
    if not checks["full_beats_wrong_cache_by_required_delta"]:
        blocked_actions.append("do_not_package_without_full_beating_wrong_cache")
    if not checks["full_beats_cache_erased_by_required_delta"]:
        blocked_actions.append("do_not_package_without_full_beating_cache_erased")
    if not checks["data_health_passed"]:
        blocked_actions.append("do_not_package_until_phase2l_data_health_passes")
    passed = all(checks.values())
    if passed and postflight_stage == "smoke":
        allowed_next_action = "run_phase2l_full_nonsealed_training_only"
    elif passed and postflight_stage == "local_full512":
        allowed_next_action = "run_phase2l_full1024_on_larger_gpu_or_repeat_local_full512_seed"
    elif passed and postflight_stage == "full":
        allowed_next_action = "run_phase2l_package_only"
    else:
        allowed_next_action = "revise_phase2l_before_next_stage"
    return {
        "audit_family": "phase2l_counterfactual_continuation_postflight",
        "postflight_stage": postflight_stage,
        "passed": passed,
        "ready_for_full_train": passed and postflight_stage == "smoke",
        "ready_for_larger_gpu_full1024": passed and postflight_stage == "local_full512",
        "ready_for_package": passed and postflight_stage == "full",
        "ready_for_sealed_eval": False,
        "allowed_next_action": allowed_next_action,
        "blocked_actions": blocked_actions,
        "checks": checks,
        "metrics": {
            "full_completion": full_completion,
            "native_head_only_completion": native_completion,
            "wrong_cache_completion": wrong_cache_completion,
            "cache_erased_completion": cache_erased_completion,
            "full_minus_native_head_only": full_minus_native,
            "full_minus_wrong_cache": full_minus_wrong_cache,
            "full_minus_cache_erased": full_minus_cache_erased,
            "full_trace_audit": full_trace,
        },
        "thresholds": {
            "min_full_completion": min_full_completion,
            "min_full_minus_native_head_only": min_full_minus_native_head_only,
            "min_full_minus_wrong_cache": min_full_minus_wrong_cache,
            "min_full_minus_cache_erased": min_full_minus_cache_erased,
        },
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "full_eval_json": str(Path(full_eval_json)),
            "native_head_only_eval_json": str(Path(native_head_only_eval_json)),
            "wrong_cache_eval_json": str(Path(wrong_cache_eval_json)),
            "cache_erased_eval_json": str(Path(cache_erased_eval_json)),
        },
    }


def build_phase2l_sealed_gate(
    *,
    full_eval_json: str | Path,
    no_nsi_eval_json: str | Path,
    native_head_only_eval_json: str | Path,
    continuation_only_eval_json: str | Path,
    prompt_only_eval_json: str | Path,
    react_eval_json: str | Path,
    min_full_completion: float = 0.85,
    min_full_minus_no_nsi: float = 0.15,
    min_full_minus_native_head_only: float = 0.15,
    min_full_minus_continuation_only: float = 0.25,
) -> dict[str, Any]:
    evals = {
        "full": _load_json(full_eval_json),
        "no_nsi": _load_json(no_nsi_eval_json),
        "native_head_only": _load_json(native_head_only_eval_json),
        "continuation_only": _load_json(continuation_only_eval_json),
        "prompt_only": _load_json(prompt_only_eval_json),
        "react": _load_json(react_eval_json),
    }
    table: dict[str, dict[str, float | str | None]] = {}
    for name, payload in evals.items():
        table[name] = {
            "policy_label": payload.get("policy", {}).get("policy_label"),
            "task_completion_rate": _metric(payload, "task_completion_rate"),
            "command_decision_accuracy": _metric(payload, "command_decision_accuracy"),
            "model_calls": _metric(payload, "model_calls"),
            "state_hallucination_rate": _metric(payload, "state_hallucination_rate"),
            "dataset_path": payload.get("dataset", {}).get("dataset_path"),
        }

    full_completion = table["full"]["task_completion_rate"]

    def _delta(other: float | str | None) -> float | None:
        if isinstance(full_completion, float) and isinstance(other, float):
            return full_completion - other
        return None

    deltas = {
        "full_minus_no_nsi": _delta(table["no_nsi"]["task_completion_rate"]),
        "full_minus_native_head_only": _delta(
            table["native_head_only"]["task_completion_rate"]
        ),
        "full_minus_continuation_only": _delta(
            table["continuation_only"]["task_completion_rate"]
        ),
    }
    dataset_paths = {
        name: row["dataset_path"] for name, row in table.items()
    }
    checks = {
        "sealed_v3_inputs_only": all(
            isinstance(path, str)
            and "external_trace_v3_semantic_required" in path
            for path in dataset_paths.values()
        ),
        "full_completion_gate_passed": (
            isinstance(full_completion, float)
            and full_completion >= min_full_completion
        ),
        "full_beats_no_nsi_by_required_delta": (
            isinstance(deltas["full_minus_no_nsi"], float)
            and deltas["full_minus_no_nsi"] >= min_full_minus_no_nsi
        ),
        "full_beats_native_head_only_by_required_delta": (
            isinstance(deltas["full_minus_native_head_only"], float)
            and deltas["full_minus_native_head_only"]
            >= min_full_minus_native_head_only
        ),
        "full_beats_continuation_only_by_required_delta": (
            isinstance(deltas["full_minus_continuation_only"], float)
            and deltas["full_minus_continuation_only"]
            >= min_full_minus_continuation_only
        ),
        "allowlist_hallucination_zero": all(
            row["state_hallucination_rate"] == 0.0 for row in table.values()
        ),
        "full_low_level_qwen_calls_zero": table["full"]["model_calls"] == 0.0,
    }
    passed = all(checks.values())
    return {
        "audit_family": "phase2l_counterfactual_continuation_sealed_v3_gate",
        "passed": passed,
        "claim_boundary": (
            "sealed_v3_supports_counterfactual_continuation_memory_necessity"
            if passed
            else "bounded_claim_only_do_not_upgrade_continuation_memory_necessity"
        ),
        "checks": checks,
        "metrics": {
            "table": table,
            "deltas": deltas,
        },
        "thresholds": {
            "min_full_completion": min_full_completion,
            "min_full_minus_no_nsi": min_full_minus_no_nsi,
            "min_full_minus_native_head_only": min_full_minus_native_head_only,
            "min_full_minus_continuation_only": min_full_minus_continuation_only,
        },
        "inputs": {
            "full_eval_json": str(Path(full_eval_json)),
            "no_nsi_eval_json": str(Path(no_nsi_eval_json)),
            "native_head_only_eval_json": str(Path(native_head_only_eval_json)),
            "continuation_only_eval_json": str(Path(continuation_only_eval_json)),
            "prompt_only_eval_json": str(Path(prompt_only_eval_json)),
            "react_eval_json": str(Path(react_eval_json)),
        },
    }


def render_phase2l_sealed_gate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase2L Sealed v3 Gate",
        "",
        f"- Passed: `{str(report['passed']).lower()}`",
        f"- Claim boundary: `{report['claim_boundary']}`",
        "",
        "| mechanism | completion | command_accuracy | model_calls | hallucination |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["metrics"]["table"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["task_completion_rate"]),
                    str(row["command_decision_accuracy"]),
                    str(row["model_calls"]),
                    str(row["state_hallucination_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| delta | value | threshold | passed |",
            "| --- | ---: | ---: | --- |",
            (
                f"| full_minus_no_nsi | {report['metrics']['deltas']['full_minus_no_nsi']} | "
                f"{report['thresholds']['min_full_minus_no_nsi']} | "
                f"{report['checks']['full_beats_no_nsi_by_required_delta']} |"
            ),
            (
                f"| full_minus_native_head_only | "
                f"{report['metrics']['deltas']['full_minus_native_head_only']} | "
                f"{report['thresholds']['min_full_minus_native_head_only']} | "
                f"{report['checks']['full_beats_native_head_only_by_required_delta']} |"
            ),
            (
                f"| full_minus_continuation_only | "
                f"{report['metrics']['deltas']['full_minus_continuation_only']} | "
                f"{report['thresholds']['min_full_minus_continuation_only']} | "
                f"{report['checks']['full_beats_continuation_only_by_required_delta']} |"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2L counterfactual-continuation gates.")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data-health")
    data.add_argument("--train-jsonl", required=True)
    data.add_argument("--val-jsonl", required=True)
    data.add_argument("--train-metadata-json", required=True)
    data.add_argument("--val-metadata-json", required=True)
    data.add_argument("--output-json")
    data.add_argument("--max-source-overlap-val-accuracy", type=float, default=0.60)
    data.add_argument("--no-fail", action="store_true")

    pretrain = sub.add_parser("pretrain-gate")
    pretrain.add_argument("--data-health-json", required=True)
    pretrain.add_argument("--output-json")
    pretrain.add_argument("--no-fail", action="store_true")

    post = sub.add_parser("postflight")
    post.add_argument("--data-health-json", required=True)
    post.add_argument("--full-eval-json", required=True)
    post.add_argument("--native-head-only-eval-json", required=True)
    post.add_argument("--wrong-cache-eval-json", required=True)
    post.add_argument("--cache-erased-eval-json", required=True)
    post.add_argument("--output-json")
    post.add_argument("--min-full-completion", type=float, default=0.85)
    post.add_argument("--min-full-minus-native-head-only", type=float, default=0.15)
    post.add_argument("--min-full-minus-wrong-cache", type=float, default=0.25)
    post.add_argument("--min-full-minus-cache-erased", type=float, default=0.25)
    post.add_argument("--stage", choices=("smoke", "local_full512", "full"), default="full")
    post.add_argument("--no-fail", action="store_true")

    sealed = sub.add_parser("sealed-gate")
    sealed.add_argument("--full-eval-json", required=True)
    sealed.add_argument("--no-nsi-eval-json", required=True)
    sealed.add_argument("--native-head-only-eval-json", required=True)
    sealed.add_argument("--continuation-only-eval-json", required=True)
    sealed.add_argument("--prompt-only-eval-json", required=True)
    sealed.add_argument("--react-eval-json", required=True)
    sealed.add_argument("--output-json")
    sealed.add_argument("--output-md")
    sealed.add_argument("--min-full-completion", type=float, default=0.85)
    sealed.add_argument("--min-full-minus-no-nsi", type=float, default=0.15)
    sealed.add_argument("--min-full-minus-native-head-only", type=float, default=0.15)
    sealed.add_argument("--min-full-minus-continuation-only", type=float, default=0.25)
    sealed.add_argument("--no-fail", action="store_true")

    args = parser.parse_args()
    if args.command == "data-health":
        report = build_phase2l_data_health(
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            train_metadata_json=args.train_metadata_json,
            val_metadata_json=args.val_metadata_json,
            max_source_overlap_val_accuracy=args.max_source_overlap_val_accuracy,
        )
    elif args.command == "pretrain-gate":
        report = build_phase2l_pretrain_gate(data_health_json=args.data_health_json)
    elif args.command == "postflight":
        report = build_phase2l_postflight(
            data_health_json=args.data_health_json,
            full_eval_json=args.full_eval_json,
            native_head_only_eval_json=args.native_head_only_eval_json,
            wrong_cache_eval_json=args.wrong_cache_eval_json,
            cache_erased_eval_json=args.cache_erased_eval_json,
            min_full_completion=args.min_full_completion,
            min_full_minus_native_head_only=args.min_full_minus_native_head_only,
            min_full_minus_wrong_cache=args.min_full_minus_wrong_cache,
            min_full_minus_cache_erased=args.min_full_minus_cache_erased,
            postflight_stage=args.stage,
        )
    else:
        report = build_phase2l_sealed_gate(
            full_eval_json=args.full_eval_json,
            no_nsi_eval_json=args.no_nsi_eval_json,
            native_head_only_eval_json=args.native_head_only_eval_json,
            continuation_only_eval_json=args.continuation_only_eval_json,
            prompt_only_eval_json=args.prompt_only_eval_json,
            react_eval_json=args.react_eval_json,
            min_full_completion=args.min_full_completion,
            min_full_minus_no_nsi=args.min_full_minus_no_nsi,
            min_full_minus_native_head_only=args.min_full_minus_native_head_only,
            min_full_minus_continuation_only=args.min_full_minus_continuation_only,
        )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if getattr(args, "output_md", None):
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_phase2l_sealed_gate_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
