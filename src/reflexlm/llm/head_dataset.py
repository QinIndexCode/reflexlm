from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexlm.data.jsonl import read_jsonl
from reflexlm.experiment import create_experiment_run
from reflexlm.llm.candidate_features import command_intent_for_text
from reflexlm.llm.prompts import _legal_action_mask_lines, _serialize_receptor_state_v2
from reflexlm.llm.receptor_latent import (
    debug_action_stage_signal,
    receptor_failure_signal,
    runtime_command_identity_signal,
)
from reflexlm.llm.sft import SynapseSignalExtractor
from reflexlm.models.features import (
    ACTION_ORDER,
    ROUTE_ORDER,
    candidate_commands,
    candidate_files,
    command_slot_target,
    delta_norm_target,
    file_slot_target,
    risk_target,
    salience_target,
    urgency_target,
    valid_action_mask,
    StateVectorizer,
)
from reflexlm.runtime.nervous_system import (
    INTERNAL_TARGET_ORDER,
    internal_target_for_state,
    route_for_internal_target,
)
from reflexlm.runtime.oracle import primary_route_for_task
from reflexlm.schema import ActionType, InternalTarget, RouteName, SystemStateFrame, TrajectoryRecord


PHASE2C_HEAD_PROMPT_STYLE = "phase2c_head_state_v1"


def _latent_sensitive_prompt(state: SystemStateFrame) -> bool:
    return "low-level receptor latent" in state.goal.description.lower()


@dataclass(slots=True)
class Phase2CHeadExample:
    example_id: str
    episode_id: str
    t: int
    task_type: str
    prompt_style: str
    state_prompt: str
    head_scope: str
    internal_target: str
    internal_target_index: int
    route_name: str
    route_index: int
    action_type: str
    action_index: int
    command_intent: str | None
    command: str | None
    file_target: str | None
    command_slot: int
    file_slot: int
    confidence_target: float
    inhibition_target: float
    salience_target: float
    risk_target: float
    urgency_target: float
    prediction_error_target: float
    legal_action_mask: dict[str, int]
    candidate_commands: list[str]
    candidate_files: list[str]
    nsi_reference: dict[str, Any] | None
    runtime_overrides: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "episode_id": self.episode_id,
            "t": self.t,
            "task_type": self.task_type,
            "prompt_style": self.prompt_style,
            "state_prompt": self.state_prompt,
            "head_scope": self.head_scope,
            "internal_target": self.internal_target,
            "internal_target_index": self.internal_target_index,
            "route_name": self.route_name,
            "route_index": self.route_index,
            "action_type": self.action_type,
            "action_index": self.action_index,
            "command_intent": self.command_intent,
            "command": self.command,
            "file_target": self.file_target,
            "command_slot": self.command_slot,
            "file_slot": self.file_slot,
            "confidence_target": self.confidence_target,
            "inhibition_target": self.inhibition_target,
            "salience_target": self.salience_target,
            "risk_target": self.risk_target,
            "urgency_target": self.urgency_target,
            "prediction_error_target": self.prediction_error_target,
            "legal_action_mask": self.legal_action_mask,
            "candidate_commands": self.candidate_commands,
            "candidate_files": self.candidate_files,
            "nsi_reference": self.nsi_reference,
            "runtime_overrides": self.runtime_overrides,
        }


def build_phase2c_head_state_prompt_from_state(state: SystemStateFrame) -> str:
    """Visible receptor text for a shared-backbone head model.

    This is backbone state input only. Labels for motor output live in explicit
    classification/regression fields, so no JSON target text is embedded here.
    """

    latent_sensitive = _latent_sensitive_prompt(state)
    visible_failure = f"{state.terminal.stderr_delta} {state.terminal.stdout_delta}".lower()
    source_inspected = (
        "source inspected" in visible_failure
        or "rerun the targeted failing test" in visible_failure
    )
    failure_signal = "other"
    if latent_sensitive:
        failure_signal = "latent_required"
    elif "snapshot" in visible_failure and (
        "mismatch" in visible_failure or "update" in visible_failure
    ):
        failure_signal = "snapshot_update"
    elif (
        "modulenotfounderror" in visible_failure
        or "no module named" in visible_failure
        or "missing dependency" in visible_failure
        or "dependency missing" in visible_failure
    ):
        failure_signal = "dependency_install"
    elif "assertionerror" in visible_failure or "assertion failure" in visible_failure:
        failure_signal = "assertion_inspection"
    debug_action_stage = debug_action_stage_signal(state)
    last_command_intent = command_intent_for_text(state.terminal.last_command)
    command_intents = [
        f"{index}:{command_intent_for_text(command)}"
        for index, command in enumerate(candidate_commands(state))
    ]
    lines = [
        "Phase 2C native nervous interface state input.",
        "Classifier heads select internal_target, route, action, slots, confidence, and inhibition.",
        "Text generation is not a motor channel for this training row.",
        "",
        "Visible transition summary:",
        f"failure_signal={failure_signal}",
        f"debug_action_stage={debug_action_stage}",
        f"source_inspected={source_inspected}",
        f"last_command_intent={last_command_intent}",
        "candidate_command_intents=" + ",".join(command_intents),
        "",
        "Motor action space:",
        ", ".join(action.value for action in ACTION_ORDER) + ".",
        "",
        "Internal targets:",
        ", ".join(target.value for target in INTERNAL_TARGET_ORDER) + ".",
        "",
        "Legal action mask:",
    ]
    lines.extend(_legal_action_mask_lines(state))
    lines.extend(["", "Receptor state:"])
    receptor_lines = _serialize_receptor_state_v2(state)
    if latent_sensitive:
        masked_receptor_lines = []
        for line in receptor_lines:
            if line.startswith("stdout_delta="):
                masked_receptor_lines.append("stdout_delta=<compressed_failure_signal>")
            elif line.startswith("stderr_delta="):
                masked_receptor_lines.append("stderr_delta=<compressed_failure_signal>")
            else:
                masked_receptor_lines.append(line)
        receptor_lines = masked_receptor_lines
    lines.extend(receptor_lines)
    lines.extend(["", "Candidate commands:"])
    lines.extend([f"- {command}" for command in candidate_commands(state)] or ["- <none>"])
    lines.extend(["", "Candidate files:"])
    lines.extend([f"- {file_target}" for file_target in candidate_files(state)] or ["- <none>"])
    lines.extend(
        [
            "",
            "Head constraints:",
            "- RUN_COMMAND must select a command slot.",
            "- READ_FILE must select a file slot.",
            "- TEST_FAILURE states should route through the Debug Cortex target.",
            "- External/stale file state should refresh before using cached file content.",
            "- Dangerous commands should activate inhibition and BLOCK.",
        ]
    )
    return "\n".join(lines)


def build_phase2c_head_state_prompt(record: TrajectoryRecord) -> str:
    return build_phase2c_head_state_prompt_from_state(record.state)


def _label_route(record: TrajectoryRecord, target: InternalTarget) -> RouteName:
    state = record.state
    if target != InternalTarget.REFLEX_MOTOR:
        return route_for_internal_target(state, target)
    if state.filesystem.external_change_detected or state.filesystem.stale_cache_detected:
        return RouteName.FILE
    return primary_route_for_task(record.goal.task_type)


def _head_scope(target: InternalTarget) -> str:
    if target == InternalTarget.ESCALATE_TO_DEBUG_CORTEX:
        return "debug_cortex"
    if target == InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX:
        return "semantic_cortex"
    if target == InternalTarget.INHIBIT:
        return "inhibition"
    return "reflex_layer"


def _legal_mask_dict(record: TrajectoryRecord) -> dict[str, int]:
    mask = valid_action_mask(record.state)
    return {action.value: int(mask[index] > 0.0) for index, action in enumerate(ACTION_ORDER)}


def _runtime_overrides(record: TrajectoryRecord, target: InternalTarget) -> list[str]:
    overrides: list[str] = []
    state = record.state
    if state.safety.dangerous_command_detected or target == InternalTarget.INHIBIT:
        overrides.append("safety_inhibition")
    if state.filesystem.external_change_detected or state.filesystem.stale_cache_detected:
        overrides.append("stale_state_refresh_receptor")
    if target == InternalTarget.ESCALATE_TO_DEBUG_CORTEX:
        overrides.append("debug_cortex_escalation")
    return overrides


def _nsi_reference_dict(
    record: TrajectoryRecord,
    extractor: SynapseSignalExtractor | None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    if extractor is not None:
        summary = extractor.summarize(record)
        payload.update(summary.to_dict())
    payload["receptor_failure_signal"] = receptor_failure_signal(record.state)
    payload["debug_action_stage"] = debug_action_stage_signal(record.state)
    payload.update(runtime_command_identity_signal(record.state))
    return payload


def _action_index(action_type: ActionType) -> int:
    return ACTION_ORDER.index(action_type)


def _build_head_examples(
    records: list[TrajectoryRecord],
    *,
    synapse_extractor: SynapseSignalExtractor | None = None,
) -> list[Phase2CHeadExample]:
    examples: list[Phase2CHeadExample] = []
    vectorizer = StateVectorizer()
    ordered = sorted(records, key=lambda record: (record.episode_id, record.t))
    for record in ordered:
        if record.action is None:
            continue
        target = internal_target_for_state(record.state)
        route = _label_route(record, target)
        action_type = record.action.type
        examples.append(
            Phase2CHeadExample(
                example_id=f"{record.episode_id}:{record.t}",
                episode_id=record.episode_id,
                t=record.t,
                task_type=record.goal.task_type.value,
                prompt_style=PHASE2C_HEAD_PROMPT_STYLE,
                state_prompt=build_phase2c_head_state_prompt(record),
                head_scope=_head_scope(target),
                internal_target=target.value,
                internal_target_index=INTERNAL_TARGET_ORDER.index(target),
                route_name=route.value,
                route_index=ROUTE_ORDER.index(route),
                action_type=action_type.value,
                action_index=_action_index(action_type),
                command_intent=(
                    command_intent_for_text(record.action.command)
                    if action_type == ActionType.RUN_COMMAND
                    else None
                ),
                command=record.action.command,
                file_target=record.action.file_target,
                command_slot=command_slot_target(record),
                file_slot=file_slot_target(record),
                confidence_target=float(record.action.confidence),
                inhibition_target=1.0 if target == InternalTarget.INHIBIT else 0.0,
                salience_target=salience_target(record),
                risk_target=risk_target(record),
                urgency_target=urgency_target(record),
                prediction_error_target=delta_norm_target(record, vectorizer),
                legal_action_mask=_legal_mask_dict(record),
                candidate_commands=candidate_commands(record.state),
                candidate_files=candidate_files(record.state),
                nsi_reference=_nsi_reference_dict(record, synapse_extractor),
                runtime_overrides=_runtime_overrides(record, target),
            )
        )
    return examples


def _write_examples(path: Path, examples: list[Phase2CHeadExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False))
            handle.write("\n")


def _count_examples(examples: list[Phase2CHeadExample]) -> dict[str, Any]:
    counts = {
        "records": len(examples),
        "by_task": Counter(example.task_type for example in examples),
        "by_internal_target": Counter(example.internal_target for example in examples),
        "by_head_scope": Counter(example.head_scope for example in examples),
        "by_route": Counter(example.route_name for example in examples),
        "by_action": Counter(example.action_type for example in examples),
        "by_command_intent": Counter(
            example.command_intent for example in examples if example.command_intent
        ),
        "by_runtime_override": Counter(
            override for example in examples for override in example.runtime_overrides
        ),
    }
    return {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in counts.items()
    }


def audit_phase2c_head_coverage(
    split_examples: dict[str, list[Phase2CHeadExample]],
) -> dict[str, Any]:
    split_pairs = {
        split_name: Counter(
            f"{example.head_scope}/{example.action_type}"
            for example in examples
        )
        for split_name, examples in split_examples.items()
    }
    train_pairs = set(split_pairs.get("train", {}))
    val_pairs = set(split_pairs.get("val", {}))
    test_pairs = set(split_pairs.get("test", {}))
    val_missing_test_pairs = sorted(test_pairs - val_pairs)
    train_missing_test_pairs = sorted(test_pairs - train_pairs)
    debug_command_intents = {
        split_name: dict(
            Counter(
                example.command_intent
                for example in examples
                if example.head_scope == "debug_cortex"
                and example.action_type == ActionType.RUN_COMMAND.value
                and example.command_intent
            )
        )
        for split_name, examples in split_examples.items()
    }
    return {
        "pair_counts": {name: dict(counter) for name, counter in split_pairs.items()},
        "debug_command_intents": debug_command_intents,
        "val_missing_test_pairs": val_missing_test_pairs,
        "train_missing_test_pairs": train_missing_test_pairs,
        "passed": not train_missing_test_pairs and not val_missing_test_pairs,
    }


def audit_phase2c_head_examples(
    examples: list[Phase2CHeadExample],
    records: list[TrajectoryRecord] | None = None,
) -> dict[str, Any]:
    forbidden_markers = [
        "recovery_hint=",
        "scenario_template",
        "oracle_action",
        '"target_text"',
        "'target_text'",
        "target_text=",
        "target_text:",
        "Return only JSON",
    ]
    marker_counts: dict[str, int] = {marker: 0 for marker in forbidden_markers}
    recovery_hints = {}
    if records is not None:
        recovery_hints = {
            f"{record.episode_id}:{record.t}": record.goal.recovery_hint
            for record in records
            if record.goal.recovery_hint
        }
    recovery_hint_value_leaks = 0
    task_key_leaks = 0
    for example in examples:
        prompt = example.state_prompt
        for marker in forbidden_markers:
            if marker in prompt:
                marker_counts[marker] += 1
        if "\ntask=" in prompt or prompt.startswith("task="):
            task_key_leaks += 1
        recovery_hint = recovery_hints.get(example.example_id)
        if recovery_hint and recovery_hint in prompt:
            recovery_hint_value_leaks += 1
    return {
        "forbidden_marker_counts": marker_counts,
        "recovery_hint_value_leaks": recovery_hint_value_leaks,
        "task_key_leaks": task_key_leaks,
        "passed": (
            not any(marker_counts.values())
            and task_key_leaks == 0
            and recovery_hint_value_leaks == 0
        ),
    }


def _new_extractor(
    synapse_checkpoint: str | Path | None,
    *,
    synapse_device: str,
) -> SynapseSignalExtractor | None:
    if synapse_checkpoint is None:
        return None
    return SynapseSignalExtractor(synapse_checkpoint, device=synapse_device)


def _records_with_episode_prefix(
    path: str | Path,
    *,
    prefix: str | None = None,
) -> list[TrajectoryRecord]:
    records = read_jsonl(Path(path))
    if not prefix:
        return records
    return [
        record.model_copy(update={"episode_id": f"{prefix}__{record.episode_id}"})
        for record in records
    ]


def _read_split_records(
    primary_path: str | Path,
    *,
    extra_paths: list[str | Path],
    split_name: str,
) -> list[TrajectoryRecord]:
    records = _records_with_episode_prefix(primary_path)
    for index, extra_path in enumerate(extra_paths):
        source_stem = Path(extra_path).parent.name or Path(extra_path).stem
        prefix = f"extra_{split_name}_{index}_{source_stem}"
        records.extend(_records_with_episode_prefix(extra_path, prefix=prefix))
    return records


def materialize_phase2c_head_corpus(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: str | Path,
    test_jsonl: str | Path | None = None,
    extra_train_jsonls: list[str | Path] | None = None,
    extra_val_jsonls: list[str | Path] | None = None,
    synapse_checkpoint: str | Path | None = None,
    synapse_device: str = "cpu",
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    split_paths: dict[str, str | Path] = {
        "train": train_jsonl,
        "val": val_jsonl,
    }
    if test_jsonl is not None:
        split_paths["test"] = test_jsonl
    extra_train_jsonls = list(extra_train_jsonls or [])
    extra_val_jsonls = list(extra_val_jsonls or [])
    extra_paths = {
        "train": extra_train_jsonls,
        "val": extra_val_jsonls,
        "test": [],
    }
    run = create_experiment_run(
        kind="phase2c_head_dataset",
        name="native_nervous_head_corpus",
        config={
            "train_jsonl": str(Path(train_jsonl).resolve()),
            "val_jsonl": str(Path(val_jsonl).resolve()),
            "test_jsonl": str(Path(test_jsonl).resolve()) if test_jsonl else None,
            "extra_train_jsonls": [str(Path(path).resolve()) for path in extra_train_jsonls],
            "extra_val_jsonls": [str(Path(path).resolve()) for path in extra_val_jsonls],
            "output_dir": str(output_dir.resolve()),
            "prompt_style": PHASE2C_HEAD_PROMPT_STYLE,
            "synapse_checkpoint": str(Path(synapse_checkpoint).resolve())
            if synapse_checkpoint
            else None,
            "synapse_device": synapse_device,
            "label_contract": {
                "json_text_target": False,
                "test_failure_internal_target": InternalTarget.ESCALATE_TO_DEBUG_CORTEX.value,
                "external_file_refresh_override": "stale_state_refresh_receptor",
                "dangerous_action_internal_target": InternalTarget.INHIBIT.value,
            },
        },
        run_root=run_root,
    )

    manifest: dict[str, Any] = {
        "prompt_style": PHASE2C_HEAD_PROMPT_STYLE,
        "output_dir": str(output_dir.resolve()),
        "json_text_target": False,
        "splits": {},
        "aggregate_counts": defaultdict(Counter),
    }
    all_examples: list[Phase2CHeadExample] = []
    split_examples: dict[str, list[Phase2CHeadExample]] = {}
    for split_name, split_path in split_paths.items():
        extractor = _new_extractor(synapse_checkpoint, synapse_device=synapse_device)
        records = _read_split_records(
            split_path,
            extra_paths=extra_paths.get(split_name, []),
            split_name=split_name,
        )
        examples = _build_head_examples(records, synapse_extractor=extractor)
        split_examples[split_name] = examples
        split_output = output_dir / f"{split_name}.jsonl"
        _write_examples(split_output, examples)
        split_counts = _count_examples(examples)
        split_audit = audit_phase2c_head_examples(examples, records=records)
        manifest["splits"][split_name] = {
            "source_jsonl": str(Path(split_path).resolve()),
            "extra_source_jsonls": [
                str(Path(path).resolve()) for path in extra_paths.get(split_name, [])
            ],
            "path": str(split_output),
            "counts": split_counts,
            "leakage_audit": split_audit,
        }
        for key, value in split_counts.items():
            if key == "records":
                continue
            manifest["aggregate_counts"][key].update(value)
        all_examples.extend(examples)

    aggregate_counts = _count_examples(all_examples)
    manifest["aggregate_counts"] = aggregate_counts
    # Aggregate audit is the conjunction of split audits; split-level checks keep
    # hidden recovery hints out of the manifest itself.
    manifest["leakage_audit"] = {
        "passed": all(split["leakage_audit"]["passed"] for split in manifest["splits"].values()),
        "split_passed": {
            split_name: split["leakage_audit"]["passed"]
            for split_name, split in manifest["splits"].items()
        },
    }
    manifest["coverage_audit"] = audit_phase2c_head_coverage(split_examples)
    manifest["run_manifest"] = run.finalize({"output_dir": str(output_dir.resolve())})
    run.write_json("phase2c_head_manifest.json", manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
