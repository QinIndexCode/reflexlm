from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import command_intent_for_text
from reflexlm.llm.receptor_latent import COMMAND_IDENTITY_LATENT_FIELDS
from reflexlm.models.features import ACTION_ORDER, MAX_CANDIDATE_SLOTS, ROUTE_ORDER
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER
from reflexlm.schema import ActionType, InternalTarget, RouteName, TaskType


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
    if isinstance(value, list):
        digest = hashlib.sha256()
        for row in value:
            digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if token}


def _candidate_commands(row: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for candidate in row.get("command_candidates", []):
        if isinstance(candidate, str):
            commands.append(candidate)
        elif isinstance(candidate, dict) and candidate.get("command") is not None:
            commands.append(str(candidate["command"]))
    return commands[:MAX_CANDIDATE_SLOTS]


def _candidate_entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for candidate in row.get("command_candidates", []):
        if isinstance(candidate, dict):
            entries.append(candidate)
        elif isinstance(candidate, str):
            entries.append({"command": candidate})
    return entries[:MAX_CANDIDATE_SLOTS]


def _runtime_evidence_text(row: dict[str, Any]) -> str:
    return json.dumps(
        row.get("runtime_visible_evidence") or {},
        ensure_ascii=False,
        sort_keys=True,
    )


def _command_identity_signal(row: dict[str, Any], commands: list[str]) -> dict[str, float]:
    evidence_tokens = _tokens(_runtime_evidence_text(row))
    evidence = row.get("runtime_visible_evidence") if isinstance(row.get("runtime_visible_evidence"), dict) else {}
    active_watch_key = str(evidence.get("active_watch_key") or "")
    candidates = _candidate_entries(row)
    scores = []
    for index, command in enumerate(commands):
        candidate = candidates[index] if index < len(candidates) else {}
        candidate_watch_key = str(candidate.get("watch_key") or "")
        if active_watch_key and candidate_watch_key:
            score = 4.0 if candidate_watch_key == active_watch_key else 0.0
        else:
            score = float(len(evidence_tokens & _tokens(command)))
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


def _state_prompt(row: dict[str, Any], commands: list[str]) -> str:
    difficulty = row.get("difficulty") if isinstance(row.get("difficulty"), dict) else {}
    lines = [
        "Phase2M external-generalization native-head state input.",
        "Use only runtime-visible repository evidence and bounded candidate commands.",
        "Text generation is not a motor channel for this training row.",
        "",
        "Current visible state:",
        str(row.get("current_visible_text") or ""),
        "",
        "Runtime-visible evidence:",
        _runtime_evidence_text(row),
        "",
        "Difficulty:",
        f"evidence_density={difficulty.get('evidence_density')}",
        f"candidate_count={difficulty.get('candidate_count')}",
        f"continuation_depth={difficulty.get('continuation_depth')}",
        f"ambiguity_class={difficulty.get('ambiguity_class')}",
        f"trace_type={difficulty.get('trace_type')}",
        "",
        "Candidate commands:",
    ]
    lines.extend(f"- {command}" for command in commands)
    lines.extend(
        [
            "",
            "Head constraints:",
            "- RUN_COMMAND must select a command slot.",
            "- Candidate choice must use only the runtime-visible evidence above.",
        ]
    )
    return "\n".join(lines)


def phase2m_trace_to_head_row(row: dict[str, Any]) -> dict[str, Any]:
    commands = _candidate_commands(row)
    expected = str(row.get("expected_command") or "")
    if expected not in commands:
        raise ValueError(f"expected_command not present in command_candidates: {row.get('trace_id')}")
    command_slot = commands.index(expected)
    command_intent = command_intent_for_text(expected)
    target = InternalTarget.ESCALATE_TO_DEBUG_CORTEX
    route = RouteName.DEBUG
    action = ActionType.RUN_COMMAND
    nsi_reference = {
        "salience": 0.8,
        "risk": 0.1,
        "prediction_error": 0.2,
        "confidence": 0.9,
        "reflex_action": action.value,
        "route_name": route.value,
        "receptor_failure_signal": "source_inspected",
        "debug_action_stage": "source_inspected",
    }
    nsi_reference.update(_command_identity_signal(row, commands))
    return {
        "example_id": str(row.get("trace_id")),
        "episode_id": str(row.get("trace_id")),
        "t": 0,
        "task_type": TaskType.TEST_FAILURE.value,
        "prompt_style": "phase2m_external_generalization_head_v1",
        "state_prompt": _state_prompt(row, commands),
        "head_scope": "debug_cortex",
        "internal_target": target.value,
        "internal_target_index": INTERNAL_TARGET_ORDER.index(target),
        "route_name": route.value,
        "route_index": ROUTE_ORDER.index(route),
        "action_type": action.value,
        "action_index": ACTION_ORDER.index(action),
        "command_intent": command_intent,
        "command": expected,
        "file_target": None,
        "command_slot": command_slot,
        "file_slot": -100,
        "confidence_target": 1.0,
        "inhibition_target": 0.0,
        "salience_target": 0.8,
        "risk_target": 0.1,
        "urgency_target": 0.5,
        "prediction_error_target": 0.2,
        "legal_action_mask": {item.value: int(item == ActionType.RUN_COMMAND) for item in ACTION_ORDER},
        "candidate_commands": commands,
        "candidate_files": [],
        "nsi_reference": nsi_reference,
        "runtime_overrides": ["debug_cortex_escalation", "phase2m_external_generalization"],
    }


def build_phase2m_head_dataset(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    train_rows = [phase2m_trace_to_head_row(row) for row in _read_jsonl(train_jsonl)]
    val_rows = [phase2m_trace_to_head_row(row) for row in _read_jsonl(val_jsonl)]
    _write_jsonl(output / "train.jsonl", train_rows)
    _write_jsonl(output / "val.jsonl", val_rows)
    manifest = {
        "dataset_family": "phase2m_external_generalization_head_dataset",
        "json_text_target": False,
        "sealed_v3_used": False,
        "splits": {
            "train": {
                "source_jsonl": str(Path(train_jsonl)),
                "path": str(output / "train.jsonl"),
                "rows": len(train_rows),
                "sha256": _sha256(train_rows),
            },
            "val": {
                "source_jsonl": str(Path(val_jsonl)),
                "path": str(output / "val.jsonl"),
                "rows": len(val_rows),
                "sha256": _sha256(val_rows),
            },
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build native-head training rows from normalized Phase2M traces."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    manifest = build_phase2m_head_dataset(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
