from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2au_policy_required_head_dataset import (
    _command_identity_reference,
    _commands,
    _command_slot,
    _descriptor_labels,
    _policy_int,
    _progress_label,
    _scope_paths,
)
from reflexlm.llm.candidate_features import command_intent_for_text
from reflexlm.models.features import ACTION_ORDER, ROUTE_ORDER
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER
from reflexlm.schema import ActionType, InternalTarget, RouteName, TaskType


DATASET_FAMILY = "phase2av_graded_descriptor_runtime_head_dataset"
CLAIM_BOUNDARY = "phase2av_head_training_rows_not_runtime_delta_evidence"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _descriptor_target(task: dict[str, Any]) -> dict[str, Any]:
    target = task.get("learned_patch_descriptor_target")
    return target if isinstance(target, dict) else {}


def _neutralize_generated_test_identifiers(text: str) -> str:
    # Generated test names are synthetic scaffolding, not runtime repair evidence.
    return re.sub(
        r"test_([A-Za-z0-9_]*?)_(?:import|attribute|symbol|literal|guard)_restored_\d+",
        r"test_\1_repair_case",
        text,
    )


def _runtime_visible_block(task: dict[str, Any], key: str) -> str:
    value = task.get(key)
    if not isinstance(value, dict):
        return ""
    return _neutralize_generated_test_identifiers(
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    )


def _runtime_visible_evidence(task: dict[str, Any]) -> dict[str, Any]:
    evidence = task.get("runtime_visible_evidence")
    return evidence if isinstance(evidence, dict) else {}


def _pytest_text(evidence: dict[str, Any]) -> str:
    before_patch = evidence.get("pytest_before_patch")
    if not isinstance(before_patch, dict):
        return json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    return "\n".join(
        str(before_patch.get(key) or "")
        for key in ("stdout_excerpt", "stderr_excerpt")
    )


def _traceback_symbols(text: str) -> list[str]:
    symbols = set(re.findall(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning)\b", text))
    if "has no attribute" in text.lower():
        symbols.add("AttributeError")
    if "is not defined" in text.lower():
        symbols.add("NameError")
    return sorted(symbols)


def _descriptor_failure_family(symbols: list[str], text: str) -> str:
    lowered = text.lower()
    symbol_set = {symbol.lower() for symbol in symbols}
    if "attributeerror" in symbol_set or "has no attribute" in lowered:
        return "attribute_missing_runtime"
    if (
        "nameerror" in symbol_set
        or "importerror" in symbol_set
        or "modulenotfounderror" in symbol_set
        or "is not defined" in lowered
        or "no module named" in lowered
    ):
        return "missing_import_or_symbol_runtime"
    if (
        "syntaxerror" in symbol_set
        or "indentationerror" in symbol_set
        or "unexpected indent" in lowered
        or "invalid syntax" in lowered
    ):
        return "syntax_load_failure_runtime"
    if "assertionerror" in symbol_set:
        return "assertion_behavior_mismatch_runtime"
    return "unknown_runtime_failure"


def _compact_runtime_repair_evidence(task: dict[str, Any]) -> dict[str, Any]:
    evidence = _runtime_visible_evidence(task)
    pytest_text = _pytest_text(evidence)
    symbols = _traceback_symbols(pytest_text)
    compact = {
        "changed_files": evidence.get("changed_files") or [],
        "watched_files": evidence.get("watched_files") or [],
        "structural_probe_hashes": evidence.get("structural_probe_hashes") or [],
        "traceback_symbols": symbols,
        "descriptor_failure_family": _descriptor_failure_family(symbols, pytest_text),
        "pytest_exit_code": (
            evidence.get("pytest_before_patch", {}).get("exit_code")
            if isinstance(evidence.get("pytest_before_patch"), dict)
            else None
        ),
    }
    return {key: value for key, value in compact.items() if value not in (None, [], "")}


def _command_identity_tokens(task: dict[str, Any]) -> str:
    evidence = _compact_runtime_repair_evidence(task)
    tokens: list[str] = []
    for key in ("structural_probe_hashes", "traceback_symbols"):
        value = evidence.get(key)
        if isinstance(value, list):
            tokens.extend(str(item) for item in value if str(item).strip())
    return " ".join(tokens)


def _candidate_summary(task: dict[str, Any]) -> list[str]:
    candidates = task.get("repair_candidates")
    if not isinstance(candidates, list):
        return []
    lines = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        visible = {
            key: candidate.get(key)
            for key in (
                "description",
                "edit_scope",
                "intent",
                "repair_action",
                "structural_probe_hash",
                "target_symbol",
            )
            if key in candidate
        }
        lines.append(json.dumps(visible, ensure_ascii=False, sort_keys=True))
    return lines


def _state_prompt(task: dict[str, Any], commands: list[str]) -> str:
    compact_evidence = _compact_runtime_repair_evidence(task)
    identity_tokens = _command_identity_tokens(task)
    lines = [
        "Phase2AV graded descriptor-runtime native-head input.",
        "Use public, non-sealed, runtime-visible evidence only.",
        "Predict bounded command and descriptor heads; do not generate freeform patches.",
        "",
        "Runtime-visible repair evidence:",
        json.dumps(compact_evidence, ensure_ascii=False, sort_keys=True),
        f"command_identity_tokens={identity_tokens}",
        "",
        f"task_id={task.get('task_id')}",
        f"repo_origin={task.get('repo_origin')}",
        f"repo_commit={task.get('repo_commit')}",
        f"problem={task.get('problem_statement')}",
        f"current_visible_text={task.get('current_visible_text')}",
        f"allowed_write_scope={','.join(_scope_paths(task))}",
        f"difficulty_axes={','.join(str(axis) for axis in task.get('difficulty_axes', []))}",
        "",
        "Runtime-visible contract:",
        _runtime_visible_block(task, "runtime_visible_contract"),
        "",
        "Runtime-visible evidence:",
        _runtime_visible_block(task, "runtime_visible_evidence"),
        "",
        "Repair candidates:",
    ]
    lines.extend(f"- {line}" for line in _candidate_summary(task))
    lines.append("")
    lines.append("Candidate verification commands:")
    lines.extend(f"- {command}" for command in commands)
    return "\n".join(lines)


def _head_row(task: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    commands = _commands(task)
    if not commands:
        raise ValueError(f"Phase2AV task lacks candidate policy command: {task.get('task_id')}")
    command_slot = _command_slot(task, commands)
    action = ActionType.RUN_COMMAND
    target = InternalTarget.ESCALATE_TO_DEBUG_CORTEX
    route = RouteName.DEBUG
    scope_paths = _scope_paths(task)
    policy = task.get("expected_policy") if isinstance(task.get("expected_policy"), dict) else {}
    descriptor_target = dict(_descriptor_target(task))
    descriptor_labels = _descriptor_labels(descriptor_target)
    state_prompt = _state_prompt(task, commands)
    command_identity_reference = _command_identity_reference(state_prompt, commands)
    compact_runtime_evidence = _compact_runtime_repair_evidence(task)
    descriptor_target["verification_command_slot"] = command_slot
    return {
        "example_id": f"{task.get('task_id')}:{split}:phase2av_descriptor_runtime",
        "episode_id": task.get("task_id"),
        "t": index,
        "task_type": TaskType.TEST_FAILURE.value,
        "prompt_style": "phase2av_graded_descriptor_runtime_head_v1",
        "state_prompt": state_prompt,
        "head_scope": "debug_cortex_phase2av_descriptor_runtime",
        "internal_target": target.value,
        "internal_target_index": INTERNAL_TARGET_ORDER.index(target),
        "route_name": route.value,
        "route_index": ROUTE_ORDER.index(route),
        "action_type": action.value,
        "action_index": ACTION_ORDER.index(action),
        "command_intent": command_intent_for_text(commands[command_slot]),
        "command": commands[command_slot],
        "command_slot": command_slot,
        "candidate_commands": commands,
        "file_target": None,
        "file_slot": -100,
        "candidate_files": scope_paths,
        "legal_action_mask": {
            item.value: int(item in {ActionType.RUN_COMMAND, ActionType.DONE})
            for item in ACTION_ORDER
        },
        "nsi_reference": {
            "reflex_action": action.value,
            "route_name": route.value,
            "receptor_failure_signal": "phase2av_runtime_visible_generated_test_failure",
            "debug_action_stage": "descriptor_runtime_apply_candidate_and_verify",
            "descriptor_failure_family": compact_runtime_evidence.get(
                "descriptor_failure_family",
                "other",
            ),
            **command_identity_reference,
            "confidence": 0.85,
            "risk": 0.35,
            "salience": 0.8,
            "prediction_error": 0.25,
        },
        "confidence_target": 0.85,
        "inhibition_target": 0.0,
        "salience_target": 0.8,
        "risk_target": 0.35,
        "urgency_target": 0.6,
        "prediction_error_target": 0.25,
        "patch_proposal_label": _policy_int(task, "patch_proposal", 1),
        "test_selection_slot": command_slot,
        "rollback_safety_label": _policy_int(task, "rollback_safety", 1),
        "stop_condition_label": _policy_int(task, "stop_condition", 0),
        "bounded_edit_scope_label": _policy_int(task, "bounded_edit_scope", int(bool(scope_paths))),
        "progress_monitor_label": _progress_label(task),
        "verification_state_label": _policy_int(task, "verification_state", 0),
        **descriptor_labels,
        "open_repair_control_label_scope": "phase2av_graded_descriptor_runtime_pretrain_gate",
        "runtime_overrides": [
            "debug_cortex_escalation",
            "phase2av_descriptor_runtime_candidate_application",
            "no_json_motor_output",
            "no_sealed_feedback",
            "no_freeform_patch_generation",
        ],
        "learned_patch_candidate_target": descriptor_target,
        "learned_patch_policy_target": {
            "patch_operation": descriptor_target.get("operation"),
            "patch_template": descriptor_target.get("after_fragment_template_id"),
            "target_source": descriptor_target.get("target_source"),
            "recorded_patch_text_as_target": False,
            "symbolic_generator_as_target": False,
        },
        "source_task_manifest": {
            "task_id": task.get("task_id"),
            "task_spec_sha256": task.get("task_spec_sha256"),
            "repo_origin": task.get("repo_origin"),
            "repo_commit": task.get("repo_commit"),
            "split": task.get("split"),
            "sealed_feedback_used": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "expected_repair_action": task.get("expected_repair_action"),
            "source_benchmark_family": (
                task.get("source", {}).get("source_benchmark_family")
                if isinstance(task.get("source"), dict)
                else None
            ),
        },
    }


def _reordered_task(task: dict[str, Any], order: list[int], *, target_slot: int) -> dict[str, Any]:
    augmented = dict(task)
    commands = _commands(task)
    augmented["candidate_policy_commands"] = [commands[index] for index in order]
    candidates = task.get("repair_candidates")
    if isinstance(candidates, list) and len(candidates) >= len(order):
        augmented["repair_candidates"] = [candidates[index] for index in order]
    augmented["task_id"] = f"{task.get('task_id')}:candidate_order_slot{target_slot}"
    augmented["candidate_order_augmentation"] = {
        "enabled": True,
        "target_correct_slot": target_slot,
        "source_task_id": task.get("task_id"),
        "order": order,
    }
    return augmented


def _augment_candidate_order(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for task in tasks:
        commands = _commands(task)
        if len(commands) <= 1:
            augmented.append(task)
            continue
        expected_slot = _command_slot(task, commands)
        if expected_slot < 0 or expected_slot >= len(commands):
            augmented.append(task)
            continue
        for target_slot in range(len(commands)):
            remaining = [index for index in range(len(commands)) if index != expected_slot]
            order = list(remaining)
            order.insert(target_slot, expected_slot)
            augmented.append(_reordered_task(task, order, target_slot=target_slot))
    return augmented


def build_phase2av_descriptor_runtime_head_dataset(
    *,
    train_tasks_jsonl: str | Path,
    val_tasks_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
    augment_train_candidate_order: bool = False,
) -> dict[str, Any]:
    train_tasks = _read_jsonl(train_tasks_jsonl)
    val_tasks = _read_jsonl(val_tasks_jsonl)
    effective_train_tasks = _augment_candidate_order(train_tasks) if augment_train_candidate_order else train_tasks
    train_rows = [_head_row(task, split="train", index=index) for index, task in enumerate(effective_train_tasks)]
    val_rows = [_head_row(task, split="val", index=index) for index, task in enumerate(val_tasks)]
    output = Path(output_dir)
    _write_jsonl(output / "train.jsonl", train_rows)
    _write_jsonl(output / "val.jsonl", val_rows)
    manifest = {
        "dataset_family": DATASET_FAMILY,
        "passed": bool(train_rows and val_rows),
        "claim_boundary": CLAIM_BOUNDARY,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "candidate_order_augmentation": {
            "train_enabled": augment_train_candidate_order,
            "source_train_rows": len(train_tasks),
            "effective_train_rows": len(effective_train_tasks),
            "val_enabled": False,
        },
        "effective_split_hashes": {
            "phase2av_head_train": _rows_sha256(train_rows),
            "phase2av_head_val": _rows_sha256(val_rows),
        },
        "source_task_hashes": {
            "phase2av_task_train": _rows_sha256(train_tasks),
            "phase2av_effective_task_train": _rows_sha256(effective_train_tasks),
            "phase2av_task_val": _rows_sha256(val_tasks),
        },
        "smoke_training_allowed": bool(train_rows and val_rows),
        "full_training_allowed": False,
        "package_allowed": False,
        "sealed_eval_allowed": False,
        "unsupported_claims": [
            "learned_descriptor_runtime_delta_before_postflight",
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "outputs": {
            "train_jsonl": str(output / "train.jsonl"),
            "val_jsonl": str(output / "val.jsonl"),
        },
        "inputs": {
            "train_tasks_jsonl": str(Path(train_tasks_jsonl)),
            "val_tasks_jsonl": str(Path(val_tasks_jsonl)),
        },
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AV descriptor-runtime native-head training rows."
    )
    parser.add_argument("--train-tasks-jsonl", required=True)
    parser.add_argument("--val-tasks-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--augment-train-candidate-order", action="store_true")
    args = parser.parse_args()
    report = build_phase2av_descriptor_runtime_head_dataset(
        train_tasks_jsonl=args.train_tasks_jsonl,
        val_tasks_jsonl=args.val_tasks_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        augment_train_candidate_order=args.augment_train_candidate_order,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
