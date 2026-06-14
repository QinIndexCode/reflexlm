from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import command_intent_for_text, structured_command_identity_rows
from reflexlm.llm.native_cortex import PATCH_OPERATION_ORDER, PATCH_TEMPLATE_ORDER
from reflexlm.models.features import ACTION_ORDER, MAX_CANDIDATE_SLOTS, ROUTE_ORDER
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER
from reflexlm.schema import ActionType, InternalTarget, RouteName, TaskType


DATASET_FAMILY = "phase2au_policy_required_runtime_delta_head_dataset"
CLAIM_BOUNDARY = "phase2au_head_training_rows_not_runtime_delta_evidence"


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


def _commands(task: dict[str, Any]) -> list[str]:
    policy_commands = task.get("candidate_policy_commands")
    if isinstance(policy_commands, list) and policy_commands:
        return [str(item).strip() for item in policy_commands if str(item).strip()]
    values = task.get("evaluation_commands")
    commands = [str(item).strip() for item in values] if isinstance(values, list) else []
    return [command for command in commands if command]


def _command_slot(task: dict[str, Any], commands: list[str]) -> int:
    expected = str(task.get("expected_repair_action") or "").strip()
    if expected:
        for index, command in enumerate(commands):
            if expected in command:
                return index
    return 0


def _scope_paths(task: dict[str, Any]) -> list[str]:
    scope = task.get("allowed_write_scope")
    if isinstance(scope, list):
        return [str(item).replace("\\", "/") for item in scope if str(item).strip()]
    if isinstance(scope, str) and scope.strip():
        return [scope.replace("\\", "/")]
    return []


def _policy_int(task: dict[str, Any], key: str, default: int = 0) -> int:
    policy = task.get("expected_policy")
    if isinstance(policy, dict) and key in policy:
        return int(bool(policy[key]))
    return default


def _progress_label(task: dict[str, Any]) -> int:
    axes = {str(axis) for axis in task.get("difficulty_axes", []) if str(axis).strip()}
    if "stateful_verification" in axes:
        return 2
    if "multi_file_interaction" in axes:
        return 1
    return 0


def _descriptor_vocab(policy: dict[str, Any]) -> tuple[str, str]:
    operation = str(policy.get("patch_operation") or "")
    template = str(policy.get("patch_template") or "")
    if operation in PATCH_OPERATION_ORDER and template in PATCH_TEMPLATE_ORDER:
        return operation, template
    if "import" in template or "import" in operation:
        return "insert_import", "import_restoration"
    if "attribute" in template or "attr" in template:
        return "replace_attribute", "call_attribute_restoration"
    if "literal" in template:
        return "replace_literal", "literal_restoration"
    if "guard" in template:
        return "insert_guard", "guard_restoration"
    return "replace_symbol", "symbol_reference_restoration"


def _descriptor_target(task: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    operation, template = _descriptor_vocab(policy)
    scope_paths = _scope_paths(task)
    target_path = scope_paths[0] if scope_paths else ""
    return {
        "schema_version": "phase2au.policy_required_patch_descriptor.v1",
        "target_source": "runtime_visible_policy_descriptor_not_patch_diff",
        "target_path": target_path,
        "operation": operation,
        "anchor": {
            "kind": "phase2au_runtime_visible_task",
            "task_spec_sha256": task.get("task_spec_sha256"),
        },
        "before_fragment_hash": str(task.get("task_spec_sha256") or "")[:16],
        "after_fragment_template_id": template,
        "literal_or_symbol_payload": {
            "original_policy_operation": policy.get("patch_operation"),
            "original_policy_template": policy.get("patch_template"),
            "difficulty_axes": task.get("difficulty_axes", []),
        },
        "safety_constraints": {
            "max_changed_files": max(1, len(scope_paths)),
            "allowed_paths": scope_paths,
            "forbid_unbounded_diff_text": True,
            "require_anchor_match": True,
            "require_rollback_verification": True,
        },
        "verification_command_slot": 0,
    }


def _descriptor_labels(target: dict[str, Any]) -> dict[str, int]:
    operation = str(target.get("operation") or "")
    template = str(target.get("after_fragment_template_id") or "")
    safety = target.get("safety_constraints") if isinstance(target.get("safety_constraints"), dict) else {}
    allowed_paths = [
        str(item).replace("\\", "/")
        for item in safety.get("allowed_paths", [])
        if str(item).strip()
    ]
    target_path = str(target.get("target_path") or "").replace("\\", "/")
    return {
        "patch_operation_label": (
            PATCH_OPERATION_ORDER.index(operation) if operation in PATCH_OPERATION_ORDER else -100
        ),
        "patch_target_file_slot": (
            min(allowed_paths.index(target_path), 3) if target_path in allowed_paths else 0
        ),
        "patch_template_slot": (
            PATCH_TEMPLATE_ORDER.index(template)
            if template in PATCH_TEMPLATE_ORDER[:4]
            else -100
        ),
    }


def _state_prompt(task: dict[str, Any], commands: list[str]) -> str:
    policy = task.get("expected_policy") if isinstance(task.get("expected_policy"), dict) else {}
    contract = (
        task.get("runtime_visible_contract")
        if isinstance(task.get("runtime_visible_contract"), dict)
        else {}
    )
    lines = [
        "Phase2AU policy-required runtime-delta native-head input.",
        "This row trains bounded repair-control heads only; it is not runtime-delta evidence.",
        "Use public, non-sealed, runtime-visible task fields only.",
        "",
        f"task_id={task.get('task_id')}",
        f"repo_origin={task.get('repo_origin')}",
        f"repo_commit={task.get('repo_commit')}",
        f"problem={task.get('problem_statement')}",
        f"allowed_write_scope={','.join(_scope_paths(task))}",
        f"difficulty_axes={','.join(str(axis) for axis in task.get('difficulty_axes', []))}",
        "",
        "Runtime-visible contract:",
    ]
    lines.extend(f"- {key}={value}" for key, value in sorted(contract.items()))
    lines.append("")
    identity = (
        task.get("runtime_visible_identity")
        if isinstance(task.get("runtime_visible_identity"), dict)
        else {}
    )
    tokens = identity.get("command_identity_tokens")
    token_text = " ".join(str(item) for item in tokens) if isinstance(tokens, list) else ""
    lines.append(f"command_identity_tokens = {token_text}")
    lines.append("")
    lines.append("Expected bounded policy heads:")
    lines.extend(f"- {key}={value}" for key, value in sorted(policy.items()))
    lines.append("")
    lines.append("Candidate verification commands:")
    lines.extend(f"- {command}" for command in commands)
    return "\n".join(lines)


def _command_identity_reference(state_prompt: str, commands: list[str]) -> dict[str, float]:
    rows = structured_command_identity_rows(state_prompt, commands)
    slot_scores = [float(row[1]) if index < len(commands) else 0.0 for index, row in enumerate(rows)]
    sorted_scores = sorted(slot_scores, reverse=True)
    best = sorted_scores[0] if sorted_scores else 0.0
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    unique_best = best > 0.0 and slot_scores.count(best) == 1
    reference: dict[str, float] = {
        f"command_identity_slot:{index}": (
            slot_scores[index] if index < len(slot_scores) else 0.0
        )
        for index in range(MAX_CANDIDATE_SLOTS)
    }
    reference["command_identity_margin"] = float(best - second if unique_best else 0.0)
    reference["command_identity_confidence"] = float(best if unique_best else 0.0)
    return reference


def _head_row(task: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    commands = _commands(task)
    if not commands:
        raise ValueError(f"Phase2AU task lacks evaluation command: {task.get('task_id')}")
    command_slot = _command_slot(task, commands)
    action = ActionType.RUN_COMMAND
    target = InternalTarget.ESCALATE_TO_DEBUG_CORTEX
    route = RouteName.DEBUG
    scope_paths = _scope_paths(task)
    policy = task.get("expected_policy") if isinstance(task.get("expected_policy"), dict) else {}
    descriptor_target = _descriptor_target(task, policy)
    descriptor_labels = _descriptor_labels(descriptor_target)
    state_prompt = _state_prompt(task, commands)
    command_identity_reference = _command_identity_reference(state_prompt, commands)
    return {
        "example_id": f"{task.get('task_id')}:{split}:phase2au_policy_required",
        "episode_id": task.get("task_id"),
        "t": index,
        "task_type": TaskType.TEST_FAILURE.value,
        "prompt_style": "phase2au_policy_required_runtime_delta_control_head_v1",
        "state_prompt": state_prompt,
        "head_scope": "debug_cortex_policy_required_repair_control",
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
            "receptor_failure_signal": "phase2au_policy_required_runtime_task",
            "debug_action_stage": "policy_required_patch_and_verify",
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
        "stop_condition_label": 0,
        "bounded_edit_scope_label": _policy_int(task, "bounded_edit_scope", int(bool(scope_paths))),
        "progress_monitor_label": _progress_label(task),
        "verification_state_label": 0,
        **descriptor_labels,
        "open_repair_control_label_scope": "phase2au_policy_required_runtime_task_gate",
        "runtime_overrides": [
            "debug_cortex_escalation",
            "phase2au_policy_required_runtime_delta_control",
            "no_json_motor_output",
            "no_sealed_feedback",
        ],
        "learned_patch_candidate_target": descriptor_target,
        "learned_patch_policy_target": {
            "patch_operation": policy.get("patch_operation"),
            "patch_template": policy.get("patch_template"),
            "normalized_patch_operation": descriptor_target["operation"],
            "normalized_patch_template": descriptor_target["after_fragment_template_id"],
            "target_source": "runtime_visible_policy_descriptor_not_patch_diff",
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
            "runtime_visible_identity_source": (
                task.get("runtime_visible_identity", {}).get("identity_source")
                if isinstance(task.get("runtime_visible_identity"), dict)
                else None
            ),
        },
    }


def build_phase2au_policy_required_head_dataset(
    *,
    train_tasks_jsonl: str | Path,
    val_tasks_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
) -> dict[str, Any]:
    train_tasks = _read_jsonl(train_tasks_jsonl)
    val_tasks = _read_jsonl(val_tasks_jsonl)
    train_rows = [_head_row(task, split="train", index=index) for index, task in enumerate(train_tasks)]
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
        "effective_split_hashes": {
            "phase2au_head_train": _rows_sha256(train_rows),
            "phase2au_head_val": _rows_sha256(val_rows),
        },
        "source_task_hashes": {
            "phase2au_task_train": _rows_sha256(train_tasks),
            "phase2au_task_val": _rows_sha256(val_tasks),
        },
        "runtime_delta_supported": False,
        "package_allowed": False,
        "sealed_eval_allowed": False,
        "unsupported_claims": [
            "learned_runtime_delta_before_phase2au_execution_postflight",
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
        description="Build Phase2AU policy-required native-head training rows."
    )
    parser.add_argument("--train-tasks-jsonl", required=True)
    parser.add_argument("--val-tasks-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()
    report = build_phase2au_policy_required_head_dataset(
        train_tasks_jsonl=args.train_tasks_jsonl,
        val_tasks_jsonl=args.val_tasks_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
