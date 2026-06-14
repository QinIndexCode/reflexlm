from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2ax_runtime_pretrain_gate import _prior_tokens
from reflexlm.cli.build_phase2au_policy_required_head_dataset import (
    _commands,
    _command_slot,
    _descriptor_labels,
    _policy_int,
    _progress_label,
)
from reflexlm.llm.candidate_features import command_intent_for_text
from reflexlm.models.features import ACTION_ORDER, MAX_CANDIDATE_SLOTS, ROUTE_ORDER
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER
from reflexlm.schema import ActionType, InternalTarget, RouteName, TaskType


DATASET_FAMILY = "phase2ax_package_loaded_counterfactual_repair_head_dataset"
CLAIM_BOUNDARY = "phase2ax_head_training_rows_not_runtime_or_claim_evidence"


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    file = Path(path)
    if not file.exists():
        return {}
    payload = json.loads(file.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
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


def _sha256_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_prior_evidence(task: dict[str, Any]) -> dict[str, Any]:
    prior = task.get("phase2ax_prior_runtime_evidence")
    prior = prior if isinstance(prior, dict) else {}
    return {
        key: prior.get(key)
        for key in (
            "changed_files",
            "watched_files",
            "structural_probe_hashes",
            "repair_modes",
            "descriptor_operation",
            "descriptor_template",
            "target_path_hash",
            "target_symbol_hash",
            "version",
        )
        if prior.get(key) not in (None, "", [])
    }


def _candidate_lines(task: dict[str, Any]) -> list[str]:
    candidates = task.get("repair_candidates")
    candidates = candidates if isinstance(candidates, list) else []
    lines: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        visible = {
            key: candidate.get(key)
            for key in ("repair_action", "intent", "edit_scope", "description")
            if candidate.get(key) not in (None, "")
        }
        lines.append(json.dumps(visible, ensure_ascii=False, sort_keys=True))
    return lines


def _state_prompt(task: dict[str, Any], commands: list[str]) -> str:
    surface = task.get("phase2ax_current_repair_surface")
    surface = surface if isinstance(surface, dict) else {}
    contract = task.get("runtime_visible_contract")
    contract = contract if isinstance(contract, dict) else {}
    lines = [
        "Phase2AX package-loaded counterfactual repair native-head input.",
        "Use only masked current repair surface plus prior runtime evidence.",
        "The current repair surface is identical within each counterfactual pair.",
        "Do not use sealed feedback, gold labels, candidate slot markers, pre-test stdout, or freeform patch text.",
        "",
        f"task_family={task.get('benchmark_family')}",
        f"repo_origin={task.get('repo_origin')}",
        f"repo_commit={task.get('repo_commit')}",
        f"pair_id={task.get('phase2ax_pair_id')}",
        "",
        "Masked current repair surface:",
        json.dumps(surface, ensure_ascii=False, sort_keys=True),
        "",
        "Prior runtime evidence:",
        json.dumps(_safe_prior_evidence(task), ensure_ascii=False, sort_keys=True),
        "",
        "Runtime-visible contract:",
        json.dumps(contract, ensure_ascii=False, sort_keys=True),
        "",
        "Bounded repair candidates:",
    ]
    lines.extend(f"- {line}" for line in _candidate_lines(task))
    lines.append("")
    lines.append("Candidate commands:")
    lines.extend(f"- {command}" for command in commands)
    return "\n".join(lines)


def _scope_paths(task: dict[str, Any]) -> list[str]:
    target = task.get("learned_patch_descriptor_target")
    target = target if isinstance(target, dict) else {}
    safety = target.get("safety_constraints")
    safety = safety if isinstance(safety, dict) else {}
    paths = safety.get("allowed_paths")
    if isinstance(paths, list):
        return [str(path).replace("\\", "/") for path in paths if str(path).strip()]
    target_path = str(target.get("target_path") or "").replace("\\", "/")
    return [target_path] if target_path else []


def _command_identity_reference(task: dict[str, Any], commands: list[str]) -> dict[str, float]:
    tokens = _prior_tokens(_safe_prior_evidence(task))
    scores: list[float] = []
    for command in commands[:MAX_CANDIDATE_SLOTS]:
        lowered = command.lower()
        score = 0.0
        for token in tokens:
            if token and token in lowered:
                score += 1.0
        scores.append(score)
    while len(scores) < MAX_CANDIDATE_SLOTS:
        scores.append(0.0)
    best = max(scores) if scores else 0.0
    sorted_scores = sorted(scores, reverse=True)
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    unique_best = best > 0.0 and scores.count(best) == 1
    payload = {
        f"command_identity_slot:{index}": float(scores[index])
        for index in range(MAX_CANDIDATE_SLOTS)
    }
    payload["command_identity_margin"] = float(best - second if unique_best else 0.0)
    payload["command_identity_confidence"] = float(best if unique_best else 0.0)
    return payload


def _descriptor_target(task: dict[str, Any], command_slot: int) -> dict[str, Any]:
    target = task.get("learned_patch_descriptor_target")
    target = dict(target) if isinstance(target, dict) else {}
    target["verification_command_slot"] = command_slot
    return target


def _head_row(task: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    commands = _commands(task)
    if not commands:
        raise ValueError(f"Phase2AX task lacks candidate policy commands: {task.get('task_id')}")
    command_slot = _command_slot(task, commands)
    action = ActionType.RUN_COMMAND
    target = InternalTarget.ESCALATE_TO_DEBUG_CORTEX
    route = RouteName.DEBUG
    descriptor_target = _descriptor_target(task, command_slot)
    state_prompt = _state_prompt(task, commands)
    scope_paths = _scope_paths(task)
    policy = task.get("expected_policy") if isinstance(task.get("expected_policy"), dict) else {}
    return {
        "example_id": f"{task.get('task_id')}:{split}:phase2ax_counterfactual_repair",
        "episode_id": task.get("task_id"),
        "t": index,
        "task_type": TaskType.TEST_FAILURE.value,
        "prompt_style": "phase2ax_package_loaded_counterfactual_repair_head_v1",
        "state_prompt": state_prompt,
        "head_scope": "debug_cortex_phase2ax_counterfactual_repair",
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
            "receptor_failure_signal": "phase2ax_prior_runtime_counterfactual_repair",
            "debug_action_stage": "package_loaded_prior_conditioned_repair_selection",
            "descriptor_failure_family": "counterfactual_prior_required",
            **_command_identity_reference(task, commands),
            "confidence": 0.9,
            "risk": 0.35,
            "salience": 0.85,
            "prediction_error": 0.2,
        },
        "confidence_target": 0.9,
        "inhibition_target": 0.0,
        "salience_target": 0.85,
        "risk_target": 0.35,
        "urgency_target": 0.6,
        "prediction_error_target": 0.2,
        "patch_proposal_label": _policy_int(task, "patch_proposal", 1),
        "test_selection_slot": command_slot,
        "rollback_safety_label": _policy_int(task, "rollback_safety", 1),
        "stop_condition_label": _policy_int(task, "stop_condition", 1),
        "bounded_edit_scope_label": _policy_int(task, "bounded_edit_scope", int(bool(scope_paths))),
        "progress_monitor_label": _progress_label(task),
        "verification_state_label": _policy_int(task, "verification_state", 1),
        **_descriptor_labels(descriptor_target),
        "open_repair_control_label_scope": "phase2ax_runtime_pretrain_gate_only",
        "runtime_overrides": [
            "debug_cortex_escalation",
            "phase2ax_package_loaded_counterfactual_repair",
            "prior_runtime_evidence_required",
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
            "pair_id": task.get("phase2ax_pair_id"),
            "pair_member": task.get("phase2ax_pair_member"),
            "repo_origin": task.get("repo_origin"),
            "repo_commit": task.get("repo_commit"),
            "sealed_feedback_used": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "expected_repair_action": task.get("expected_repair_action"),
        },
    }


def _split_by_pair(rows: list[dict[str, Any]], *, train_pair_count: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row.get("phase2ax_pair_id"))].append(row)
    pair_ids = sorted(pair_id for pair_id, pair_rows in by_pair.items() if len(pair_rows) == 2)
    if not pair_ids:
        return [], []
    if train_pair_count is None:
        train_pair_count = max(1, len(pair_ids) // 2)
    train_ids = set(pair_ids[: max(1, min(train_pair_count, len(pair_ids) - 1))])
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        target = train_rows if pair_id in train_ids else val_rows
        target.extend(sorted(by_pair[pair_id], key=lambda item: str(item.get("phase2ax_pair_member"))))
    return train_rows, val_rows


def _slot_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("command_slot")) for row in rows).items()))


def build_phase2ax_head_dataset(
    *,
    tasks_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
    data_health_json: str | Path | None = None,
    pretrain_gate_json: str | Path | None = None,
    train_pair_count: int | None = None,
) -> dict[str, Any]:
    raw_rows = _read_jsonl(tasks_jsonl)
    train_tasks, val_tasks = _split_by_pair(raw_rows, train_pair_count=train_pair_count)
    train_rows = [_head_row(row, split="train", index=index) for index, row in enumerate(train_tasks)]
    val_rows = [_head_row(row, split="val", index=index) for index, row in enumerate(val_tasks)]
    output = Path(output_dir)
    _write_jsonl(output / "train.jsonl", train_rows)
    _write_jsonl(output / "val.jsonl", val_rows)
    data_health = _read_json(data_health_json)
    pretrain_gate = _read_json(pretrain_gate_json)
    manifest = {
        "dataset_family": DATASET_FAMILY,
        "passed": bool(train_rows and val_rows)
        and (not data_health or data_health.get("passed") is True)
        and (not pretrain_gate or pretrain_gate.get("passed") is True),
        "claim_boundary": CLAIM_BOUNDARY,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "source_rows": len(raw_rows),
        "source_data_health_passed": data_health.get("passed") if data_health else None,
        "source_pretrain_gate_passed": pretrain_gate.get("passed") if pretrain_gate else None,
        "command_slot_distribution": {
            "train": _slot_counts(train_rows),
            "val": _slot_counts(val_rows),
        },
        "effective_split_hashes": {
            "phase2ax_head_train": _sha256_rows(train_rows),
            "phase2ax_head_val": _sha256_rows(val_rows),
        },
        "source_task_hashes": {
            "phase2ax_tasks": _sha256_rows(raw_rows),
            "phase2ax_task_train": _sha256_rows(train_tasks),
            "phase2ax_task_val": _sha256_rows(val_tasks),
        },
        "smoke_training_allowed": bool(train_rows and val_rows)
        and pretrain_gate.get("passed") is True,
        "full_training_allowed": False,
        "package_allowed": False,
        "sealed_eval_allowed": False,
        "unsupported_claims": [
            "phase2ax_model_delta_before_smoke_postflight",
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
            "tasks_jsonl": str(Path(tasks_jsonl)),
            "data_health_json": str(Path(data_health_json)) if data_health_json else None,
            "pretrain_gate_json": str(Path(pretrain_gate_json)) if pretrain_gate_json else None,
        },
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AX native-head rows from counterfactual repair tasks."
    )
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--data-health-json")
    parser.add_argument("--pretrain-gate-json")
    parser.add_argument("--train-pair-count", type=int)
    args = parser.parse_args()
    manifest = build_phase2ax_head_dataset(
        tasks_jsonl=args.tasks_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        data_health_json=args.data_health_json,
        pretrain_gate_json=args.pretrain_gate_json,
        train_pair_count=args.train_pair_count,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
