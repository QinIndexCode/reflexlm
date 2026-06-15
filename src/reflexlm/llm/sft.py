from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexlm.data.jsonl import read_jsonl
from reflexlm.experiment import create_experiment_run
from reflexlm.eval import SequenceModelPolicy
from reflexlm.llm.prompts import (
    SYNAPSE_REQUIRED_PROMPT_STYLES,
    SynapseSummary,
    build_phase2_user_prompt,
    canonical_action_json,
)
from reflexlm.runtime.oracle import primary_route_for_task
from reflexlm.schema import ActionDecision, TrajectoryRecord
from reflexlm.train import load_model_checkpoint


@dataclass(slots=True)
class SFTExample:
    example_id: str
    episode_id: str
    t: int
    task_type: str
    route_name: str
    prompt_style: str
    system_prompt: str
    user_prompt: str
    target_text: str
    action_type: str
    command: str | None
    file_target: str | None
    synapse_summary: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "episode_id": self.episode_id,
            "t": self.t,
            "task_type": self.task_type,
            "route_name": self.route_name,
            "prompt_style": self.prompt_style,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "target_text": self.target_text,
            "action_type": self.action_type,
            "command": self.command,
            "file_target": self.file_target,
            "synapse_summary": self.synapse_summary,
        }


class SynapseSignalExtractor:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu") -> None:
        model, vectorizer, _payload = load_model_checkpoint(checkpoint_path, device=device)
        self.policy = SequenceModelPolicy(model, vectorizer, policy_label="sft_signal_extractor")
        self._current_episode: str | None = None

    def summarize(self, record: TrajectoryRecord) -> SynapseSummary:
        if record.episode_id != self._current_episode:
            self.policy.reset()
            self._current_episode = record.episode_id
        action = self.policy.act(record.state)
        debug = self.policy.last_call
        return SynapseSummary(
            route_name=str(debug["route_name"]),
            salience=float(debug["salience"]),
            risk=float(debug["risk"]),
            prediction_error=float(debug["prediction_error"]),
            confidence=float(debug["confidence"]),
            reflex_action=action.type.value,
            reflex_command=action.command,
            reflex_file_target=action.file_target,
        )


def _build_examples(
    records: list[TrajectoryRecord],
    *,
    prompt_style: str,
    synapse_extractor: SynapseSignalExtractor | None = None,
) -> list[SFTExample]:
    from reflexlm.llm.prompts import phase2_system_prompt

    examples: list[SFTExample] = []
    ordered = sorted(records, key=lambda record: (record.episode_id, record.t))
    for record in ordered:
        if record.action is None:
            continue
        synapse_summary = synapse_extractor.summarize(record) if synapse_extractor else None
        examples.append(
            SFTExample(
                example_id=f"{record.episode_id}:{record.t}",
                episode_id=record.episode_id,
                t=record.t,
                task_type=record.goal.task_type.value,
                route_name=primary_route_for_task(record.goal.task_type).value,
                prompt_style=prompt_style,
                system_prompt=phase2_system_prompt(prompt_style=prompt_style),
                user_prompt=build_phase2_user_prompt(
                    record.state,
                    prompt_style=prompt_style,
                    synapse_summary=synapse_summary,
                ),
                target_text=canonical_action_json(record.action),
                action_type=record.action.type.value,
                command=record.action.command,
                file_target=record.action.file_target,
                synapse_summary=synapse_summary.to_dict() if synapse_summary else None,
            )
        )
    return examples


def _write_examples(path: Path, examples: list[SFTExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False))
            handle.write("\n")


def materialize_sft_corpus(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: str | Path,
    prompt_styles: list[str],
    synapse_checkpoint: str | Path | None = None,
    synapse_device: str = "cpu",
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    train_records = read_jsonl(Path(train_jsonl))
    val_records = read_jsonl(Path(val_jsonl))
    run = create_experiment_run(
        kind="sft_dataset",
        name="phase2_sft_corpus",
        config={
            "train_jsonl": str(Path(train_jsonl).resolve()),
            "val_jsonl": str(Path(val_jsonl).resolve()),
            "prompt_styles": prompt_styles,
            "synapse_checkpoint": str(Path(synapse_checkpoint).resolve()) if synapse_checkpoint else None,
            "synapse_device": synapse_device,
        },
        run_root=run_root,
    )
    manifests: dict[str, Any] = {"styles": {}}
    for prompt_style in prompt_styles:
        extractor = None
        if prompt_style in SYNAPSE_REQUIRED_PROMPT_STYLES:
            if synapse_checkpoint is None:
                raise ValueError(f"{prompt_style} prompts require --synapse-checkpoint")
            extractor = SynapseSignalExtractor(synapse_checkpoint, device=synapse_device)
        train_examples = _build_examples(train_records, prompt_style=prompt_style, synapse_extractor=extractor)
        if extractor is not None:
            extractor = SynapseSignalExtractor(synapse_checkpoint, device=synapse_device)
        val_examples = _build_examples(val_records, prompt_style=prompt_style, synapse_extractor=extractor)
        style_dir = output_dir / prompt_style
        shared_train = style_dir / "shared" / "train.jsonl"
        shared_val = style_dir / "shared" / "val.jsonl"
        _write_examples(shared_train, train_examples)
        _write_examples(shared_val, val_examples)
        route_counts: dict[str, dict[str, int]] = defaultdict(dict)
        for split_name, examples in [("train", train_examples), ("val", val_examples)]:
            grouped: dict[str, list[SFTExample]] = defaultdict(list)
            for example in examples:
                grouped[example.route_name].append(example)
            for route_name, route_examples in grouped.items():
                _write_examples(style_dir / "by_route" / route_name / f"{split_name}.jsonl", route_examples)
                route_counts.setdefault(route_name, {})[split_name] = len(route_examples)
        manifests["styles"][prompt_style] = {
            "shared": {
                "train": len(train_examples),
                "val": len(val_examples),
                "train_path": str(shared_train),
                "val_path": str(shared_val),
            },
            "by_route": route_counts,
        }
    manifests["run_manifest"] = run.finalize({"output_dir": str(output_dir.resolve())})
    run.write_json("sft_manifest.json", manifests)
    (output_dir / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    return manifests
