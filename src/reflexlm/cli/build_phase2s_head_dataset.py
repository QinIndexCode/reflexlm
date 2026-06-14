from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import command_intent_for_text
from reflexlm.llm.native_cortex import PATCH_OPERATION_ORDER, PATCH_TEMPLATE_ORDER
from reflexlm.llm.receptor_latent import COMMAND_IDENTITY_LATENT_FIELDS
from reflexlm.models.features import ACTION_ORDER, MAX_CANDIDATE_SLOTS, ROUTE_ORDER
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER
from reflexlm.schema import ActionType, InternalTarget, RouteName, TaskType


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _sha256(value: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(value, list):
        for row in value:
            digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()
    digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _get_dict(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def _get_list(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    return value if isinstance(value, list) else []


def _repair_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in _get_list(row, "repair_candidates") if isinstance(item, dict)][
        :MAX_CANDIDATE_SLOTS
    ]


def _candidate_metadata_fields(candidate: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for key in (
        "edit_scope",
        "target_symbol",
        "target_literal_hash",
        "structural_probe_hash",
        "target_line",
        "target_col",
    ):
        value = candidate.get(key)
        if value is None or value == "":
            continue
        fields.append(f"{key}={value}")
    return fields


def _candidate_commands(row: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for candidate in _repair_candidates(row):
        action = candidate.get("repair_action")
        if action is not None:
            structural_probe_hash = str(candidate.get("structural_probe_hash") or "")
            commands.append(
                " ".join(
                    item
                    for item in [
                        str(action),
                        f"command_identity_tokens={structural_probe_hash}"
                        if structural_probe_hash
                        else "",
                        *_candidate_metadata_fields(candidate),
                    ]
                    if item
                )
            )
    return commands


def _string_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item).replace("\\", "/").lower() for item in value if item is not None}
    if value is None:
        return set()
    return {str(value).replace("\\", "/").lower()}


def _contains_path_or_symbol(haystacks: set[str], needle: str) -> bool:
    normalized = needle.replace("\\", "/").lower().strip()
    if not normalized:
        return False
    return any(normalized == item or normalized in item or item in normalized for item in haystacks)


def _contains_symbol(haystacks: set[str], needle: str) -> bool:
    normalized = needle.replace("\\", "/").lower().strip()
    if not normalized:
        return False
    return any(
        normalized == item
        or normalized.endswith("." + item)
        or item.endswith("." + normalized)
        for item in haystacks
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _command_identity_signal(row: dict[str, Any], commands: list[str]) -> dict[str, float]:
    runtime = _get_dict(row, "runtime_visible_evidence")
    traceback_symbols = _string_set(runtime.get("traceback_symbols"))
    changed_files = _string_set(runtime.get("changed_files"))
    watched_files = _string_set(runtime.get("watched_files"))
    failing_test_target = _string_set(runtime.get("failing_test_target"))
    expected_literal_hash = str(runtime.get("expected_literal_hash") or "").lower()
    structural_probe_hashes = {
        str(item).lower()
        for item in _get_list(runtime, "structural_probe_hashes")
        if item is not None
    }
    raw_target_location = runtime.get("target_location")
    target_location = raw_target_location if isinstance(raw_target_location, dict) else {}
    target_line = _int_or_none(target_location.get("line"))
    target_col = _int_or_none(target_location.get("col"))
    target_path = str(target_location.get("path") or "").replace("\\", "/").lower()
    candidates = _repair_candidates(row)
    scores: list[float] = []
    for index, _command in enumerate(commands):
        candidate = candidates[index] if index < len(candidates) else {}
        target_symbol = str(candidate.get("target_symbol") or "")
        edit_scope = str(candidate.get("edit_scope") or "")
        candidate_literal_hash = str(candidate.get("target_literal_hash") or "").lower()
        candidate_structural_probe_hash = str(
            candidate.get("structural_probe_hash") or ""
        ).lower()
        candidate_line = _int_or_none(candidate.get("target_line"))
        candidate_col = _int_or_none(candidate.get("target_col"))
        score = 0.0
        if (
            candidate_structural_probe_hash
            and candidate_structural_probe_hash in structural_probe_hashes
        ):
            score += 6.0
        if (
            expected_literal_hash
            and candidate_literal_hash
            and candidate_literal_hash == expected_literal_hash
        ):
            score += 6.0
            if (
                target_line is not None
                and target_col is not None
                and candidate_line == target_line
                and candidate_col == target_col
            ):
                score += 3.0
        if (
            target_path
            and edit_scope
            and _contains_path_or_symbol({target_path}, edit_scope)
            and target_line is not None
            and target_col is not None
            and candidate_line == target_line
            and candidate_col == target_col
        ):
            score += 3.0
        if target_symbol and _contains_symbol(traceback_symbols, target_symbol):
            score += 4.0
        if edit_scope and _contains_path_or_symbol(changed_files, edit_scope):
            score += 2.0
        if edit_scope and _contains_path_or_symbol(watched_files | failing_test_target, edit_scope):
            score += 1.0
        scores.append(score)
    while len(scores) < MAX_CANDIDATE_SLOTS:
        scores.append(0.0)
    sorted_scores = sorted(scores, reverse=True)
    best = sorted_scores[0] if sorted_scores else 0.0
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    unique_best = best > 0.0 and scores.count(best) == 1
    payload = {
        field: float(scores[index])
        for index, field in enumerate(COMMAND_IDENTITY_LATENT_FIELDS)
        if field.startswith("command_identity_slot:")
    }
    payload["command_identity_margin"] = float(best - second if unique_best else 0.0)
    payload["command_identity_confidence"] = float(best if unique_best else 0.0)
    return payload


def _runtime_evidence_text(row: dict[str, Any]) -> str:
    return json.dumps(
        row.get("runtime_visible_evidence") or {},
        ensure_ascii=False,
        sort_keys=True,
    )


def _candidate_lines(row: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for candidate in _repair_candidates(row):
        action = str(candidate.get("repair_action") or "")
        fields = [
            f"repair_action={action}",
            f"intent={candidate.get('intent')}",
            *_candidate_metadata_fields(candidate),
            f"verification_command={candidate.get('verification_command')}",
            f"description={candidate.get('description')}",
        ]
        lines.append("- " + "; ".join(fields))
    return lines


def _state_prompt(row: dict[str, Any], commands: list[str]) -> str:
    difficulty = _get_dict(row, "difficulty")
    lines = [
        "Phase2S public repository repair native-head state input.",
        "Use only runtime-visible repository evidence and bounded repair actions.",
        "Text generation is not a motor channel for this training row.",
        "",
        "Current visible state:",
        str(row.get("current_visible_text") or ""),
        "",
        "Runtime-visible repair evidence:",
        _runtime_evidence_text(row),
        "",
        "Difficulty:",
        f"task_family={difficulty.get('task_family')}",
        f"evidence_density={difficulty.get('evidence_density')}",
        f"candidate_count={difficulty.get('candidate_count')}",
        f"repair_depth={difficulty.get('repair_depth')}",
        f"failure_observability={difficulty.get('failure_observability')}",
        f"ambiguity_class={difficulty.get('ambiguity_class')}",
        "",
        "Candidate repair actions:",
    ]
    lines.extend(_candidate_lines(row) or ["- <none>"])
    lines.extend(
        [
            "",
            "Candidate commands:",
            *[f"- {command}" for command in commands],
            "",
            "Head constraints:",
            "- RUN_COMMAND must select one repair action command slot.",
            "- Selection must use public-repo runtime evidence, sandbox repair artifacts, and bounded edit scope.",
            "- Do not infer from hidden labels, sealed failures, or candidate slot markers.",
        ]
    )
    return "\n".join(lines)


def _phase2at_descriptor_labels(row: dict[str, Any]) -> dict[str, int]:
    target = row.get("learned_patch_candidate_target")
    if not isinstance(target, dict):
        return {
            "patch_operation_label": -100,
            "patch_target_file_slot": -100,
            "patch_template_slot": -100,
        }
    operation = str(target.get("operation") or "")
    operation_label = (
        PATCH_OPERATION_ORDER.index(operation)
        if operation in PATCH_OPERATION_ORDER
        else -100
    )
    safety = target.get("safety_constraints") if isinstance(target.get("safety_constraints"), dict) else {}
    allowed_paths = [
        str(item).replace("\\", "/")
        for item in safety.get("allowed_paths", [])
        if str(item).strip()
    ]
    target_path = str(target.get("target_path") or "").replace("\\", "/")
    target_file_slot = allowed_paths.index(target_path) if target_path in allowed_paths else 0
    template_id = str(target.get("after_fragment_template_id") or "")
    patch_template_slot = (
        PATCH_TEMPLATE_ORDER.index(template_id)
        if template_id in PATCH_TEMPLATE_ORDER[:MAX_CANDIDATE_SLOTS]
        else -100
    )
    return {
        "patch_operation_label": operation_label,
        "patch_target_file_slot": min(target_file_slot, MAX_CANDIDATE_SLOTS - 1),
        "patch_template_slot": patch_template_slot,
    }


def phase2s_repair_trace_to_head_row(row: dict[str, Any]) -> dict[str, Any]:
    commands = _candidate_commands(row)
    expected = str(row.get("expected_repair_action") or "")
    matching_slots = [
        index
        for index, command in enumerate(commands)
        if command == expected or command.startswith(expected + " ")
    ]
    if not matching_slots:
        raise ValueError(
            f"expected_repair_action not present in repair_candidates: {row.get('trace_id')}"
        )
    command_slot = matching_slots[0]
    target = InternalTarget.ESCALATE_TO_DEBUG_CORTEX
    route = RouteName.DEBUG
    action = ActionType.RUN_COMMAND
    nsi_reference = {
        "salience": 0.85,
        "risk": 0.12,
        "prediction_error": 0.25,
        "confidence": 0.9,
        "reflex_action": action.value,
        "route_name": route.value,
        "receptor_failure_signal": "source_inspected",
        "debug_action_stage": "source_inspected",
    }
    nsi_reference.update(_command_identity_signal(row, commands))
    repair_runtime = _get_dict(row, "repair_runtime")
    patch_recorded = bool(repair_runtime.get("patch_application_recorded"))
    bounded_scope = bool(repair_runtime.get("bounded_edit_scope_observed"))
    rollback_recorded = bool(repair_runtime.get("rollback_recorded"))
    post_patch_tests = bool(repair_runtime.get("post_patch_tests_recorded"))
    phase2at_labels = _phase2at_descriptor_labels(row)
    return {
        "example_id": str(row.get("trace_id")),
        "episode_id": str(row.get("trace_id")),
        "t": 0,
        "task_type": TaskType.TEST_FAILURE.value,
        "prompt_style": "phase2s_public_repair_head_v1",
        "state_prompt": _state_prompt(row, commands),
        "head_scope": "debug_cortex",
        "internal_target": target.value,
        "internal_target_index": INTERNAL_TARGET_ORDER.index(target),
        "route_name": route.value,
        "route_index": ROUTE_ORDER.index(route),
        "action_type": action.value,
        "action_index": ACTION_ORDER.index(action),
        "command_intent": command_intent_for_text(expected),
        "command": expected,
        "file_target": None,
        "command_slot": command_slot,
        "file_slot": -100,
        "confidence_target": 1.0,
        "inhibition_target": 0.0,
        "salience_target": 0.85,
        "risk_target": 0.12,
        "urgency_target": 0.6,
        "prediction_error_target": 0.25,
        "legal_action_mask": {item.value: int(item == ActionType.RUN_COMMAND) for item in ACTION_ORDER},
        "candidate_commands": commands,
        "candidate_files": [],
        "patch_proposal_label": int(patch_recorded and bounded_scope),
        "test_selection_slot": 0,
        "rollback_safety_label": int(rollback_recorded),
        "stop_condition_label": 0,
        "bounded_edit_scope_label": int(bounded_scope),
        "progress_monitor_label": 1,
        "verification_state_label": 0 if not post_patch_tests else 1,
        **phase2at_labels,
        "nsi_reference": nsi_reference,
        "runtime_overrides": [
            "debug_cortex_escalation",
            "phase2s_public_repair",
            "sandboxed_repair_evidence",
        ],
        "source_trace": {
            "repo_id": row.get("repo_id"),
            "repo_url_or_origin": row.get("repo_url_or_origin"),
            "commit_hash": row.get("commit_hash"),
            "trace_hash": row.get("trace_hash"),
            "source_kind": row.get("source_kind"),
            "sealed_v3_used": False,
        },
    }


def _build_manifest(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: Path,
    train_raw_rows: list[dict[str, Any]],
    val_raw_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    data_health_json: str | Path | None,
    pretrain_gate_json: str | Path | None,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    pretrain_gate = _read_json(pretrain_gate_json)
    effective_split_hashes = None
    if data_health:
        value = data_health.get("effective_split_hashes")
        if isinstance(value, dict) and value:
            effective_split_hashes = value
    if effective_split_hashes is None:
        effective_split_hashes = {
            "phase2s_train": _sha256(train_raw_rows),
            "phase2s_val": _sha256(val_raw_rows),
        }
    manifest = {
        "dataset_family": "phase2s_public_repair_head_dataset",
        "json_text_target": False,
        "sealed_v3_used": False,
        "claim_bearing_training_candidate": True,
        "source_data_health_passed": data_health.get("passed") if data_health else None,
        "source_pretrain_gate_passed": pretrain_gate.get("passed") if pretrain_gate else None,
        "effective_split_hashes": effective_split_hashes,
        "splits": {
            "train": {
                "source_jsonl": str(Path(train_jsonl)),
                "path": str(output_dir / "train.jsonl"),
                "source_rows": len(train_raw_rows),
                "rows": len(train_rows),
                "source_sha256": _sha256(train_raw_rows),
                "sha256": _sha256(train_rows),
            },
            "val": {
                "source_jsonl": str(Path(val_jsonl)),
                "path": str(output_dir / "val.jsonl"),
                "source_rows": len(val_raw_rows),
                "rows": len(val_rows),
                "source_sha256": _sha256(val_raw_rows),
                "sha256": _sha256(val_rows),
            },
        },
        "inputs": {
            "data_health_json": str(Path(data_health_json)) if data_health_json else None,
            "pretrain_gate_json": str(Path(pretrain_gate_json)) if pretrain_gate_json else None,
        },
    }
    return manifest


def build_phase2s_head_dataset(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: str | Path,
    data_health_json: str | Path | None = None,
    pretrain_gate_json: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    train_raw_rows = _read_jsonl(train_jsonl)
    val_raw_rows = _read_jsonl(val_jsonl)
    train_rows = [phase2s_repair_trace_to_head_row(row) for row in train_raw_rows]
    val_rows = [phase2s_repair_trace_to_head_row(row) for row in val_raw_rows]
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "train.jsonl", train_rows)
    _write_jsonl(output / "val.jsonl", val_rows)
    manifest = _build_manifest(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        output_dir=output,
        train_raw_rows=train_raw_rows,
        val_raw_rows=val_raw_rows,
        train_rows=train_rows,
        val_rows=val_rows,
        data_health_json=data_health_json,
        pretrain_gate_json=pretrain_gate_json,
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build native-head training rows from Phase2S public repair traces."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-health-json")
    parser.add_argument("--pretrain-gate-json")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    manifest = build_phase2s_head_dataset(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        data_health_json=args.data_health_json,
        pretrain_gate_json=args.pretrain_gate_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
