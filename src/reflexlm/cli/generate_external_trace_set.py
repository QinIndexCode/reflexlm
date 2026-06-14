from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.generate_debug_cortex_challenge import build_debug_cortex_challenge
from reflexlm.data.jsonl import read_jsonl
from reflexlm.experiment import stable_config_hash
from reflexlm.llm.candidate_features import command_intent_for_text
from reflexlm.models.features import candidate_commands, candidate_files, serialize_state_as_text


DEFAULT_OUTPUT = Path("artifacts/datasets/phase2g_external_trace_v1")
DEFAULT_CONTROL_DIR = Path("artifacts/control")
DEFAULT_REFERENCES = [
    Path("artifacts/datasets/phase2d_debug_ood_v2/challenge.jsonl"),
    Path("artifacts/datasets/phase2d_quasi_real_terminal_v1/challenge.jsonl"),
    Path("artifacts/datasets/phase2f_latent_sensitive/challenge.jsonl"),
    Path("artifacts/datasets/phase2g_external_trace_v1/challenge.jsonl"),
]


def _tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9_./-]+", value.lower()) if token}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def _commands_and_files(records) -> tuple[set[str], set[str]]:
    commands: set[str] = set()
    files: set[str] = set()
    for record in records:
        commands.update(candidate_commands(record.state))
        files.update(candidate_files(record.state))
        if record.action.command:
            commands.add(record.action.command)
        if record.action.file_target:
            files.add(record.action.file_target)
    return commands, files


def _state_texts(records) -> list[str]:
    return [serialize_state_as_text(record.state) for record in records]


def build_leakage_audit(challenge_path: str | Path) -> dict[str, Any]:
    records = read_jsonl(Path(challenge_path))
    hidden_hint_leaks = 0
    scenario_template_leaks = 0
    oracle_label_leaks = 0
    for record in records:
        serialized = serialize_state_as_text(record.state)
        lowered = serialized.lower()
        if "recovery_hint=" in serialized or (
            record.goal.recovery_hint and record.goal.recovery_hint in serialized
        ):
            hidden_hint_leaks += 1
        if "scenario_template" in serialized:
            scenario_template_leaks += 1
        if any(
            marker in lowered
            for marker in (
                "oracle_action",
                "oracle_label",
                "oracle=",
                "oracle:",
                "correct_action",
            )
        ):
            oracle_label_leaks += 1
    return {
        "record_count": len(records),
        "hidden_hint_leaks": hidden_hint_leaks,
        "scenario_template_leaks": scenario_template_leaks,
        "oracle_label_leaks": oracle_label_leaks,
        "passed": hidden_hint_leaks == 0
        and scenario_template_leaks == 0
        and oracle_label_leaks == 0,
    }


def build_command_slot_overlap_audit(
    challenge_path: str | Path,
    reference_paths: list[str | Path],
) -> dict[str, Any]:
    records = read_jsonl(Path(challenge_path))
    commands, files = _commands_and_files(records)
    references = {}
    total_command_overlap: set[str] = set()
    total_file_overlap: set[str] = set()
    for reference in reference_paths:
        reference_path = Path(reference)
        if not reference_path.exists():
            continue
        reference_records = read_jsonl(reference_path)
        reference_commands, reference_files = _commands_and_files(reference_records)
        command_overlap = commands & reference_commands
        file_overlap = files & reference_files
        references[str(reference_path)] = {
            "command_overlap_count": len(command_overlap),
            "file_overlap_count": len(file_overlap),
            "command_overlap": sorted(command_overlap),
            "file_overlap": sorted(file_overlap),
        }
        total_command_overlap.update(command_overlap)
        total_file_overlap.update(file_overlap)
    return {
        "external_command_count": len(commands),
        "external_file_slot_count": len(files),
        "total_exact_command_overlap_count": len(total_command_overlap),
        "total_exact_file_overlap_count": len(total_file_overlap),
        "references": references,
        "passed": len(total_command_overlap) == 0,
    }


def build_semantic_nn_audit(
    challenge_path: str | Path,
    reference_paths: list[str | Path],
) -> dict[str, Any]:
    records = read_jsonl(Path(challenge_path))
    external_tokens = [_tokens(text) for text in _state_texts(records)]
    best = {"score": 0.0, "reference": None, "external_index": None, "reference_index": None}
    for reference in reference_paths:
        reference_path = Path(reference)
        if not reference_path.exists():
            continue
        reference_records = read_jsonl(reference_path)
        reference_tokens = [_tokens(text) for text in _state_texts(reference_records)]
        for external_index, left in enumerate(external_tokens):
            for reference_index, right in enumerate(reference_tokens):
                score = _jaccard(left, right)
                if score > float(best["score"]):
                    best = {
                        "score": round(score, 6),
                        "reference": str(reference_path),
                        "external_index": external_index,
                        "reference_index": reference_index,
                    }
    return {
        "method": "token_jaccard_nearest_neighbor_over_serialized_visible_state",
        "max_similarity": best,
        "passed": float(best["score"]) < 0.80,
    }


def build_semantic_necessity_audit(challenge_path: str | Path) -> dict[str, Any]:
    records = read_jsonl(Path(challenge_path))
    by_episode: dict[str, list[Any]] = {}
    for record in records:
        by_episode.setdefault(record.episode_id, []).append(record)

    ambiguous_episodes = 0
    semantic_source_episodes = 0
    non_last_command_episodes = 0
    same_intent_command_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for episode_id, episode_records in by_episode.items():
        ordered = sorted(episode_records, key=lambda record: record.t)
        first_state = ordered[0].state
        commands = candidate_commands(first_state)
        intent_counts = Counter(command_intent_for_text(command) for command in commands)
        same_intent_command_counts.update(intent_counts)
        has_ambiguous_test_commands = intent_counts.get("test_rerun", 0) >= 2
        has_semantic_source = any(
            "semantic disambiguation required"
            in f"{record.state.terminal.stdout_delta} {record.state.terminal.stderr_delta}".lower()
            for record in ordered
        )
        run_records = [
            record
            for record in ordered
            if record.action and record.action.command and record.action.command in commands
        ]
        correct_command = run_records[-1].action.command if run_records else None
        differs_from_last = bool(correct_command and correct_command != first_state.terminal.last_command)
        if has_ambiguous_test_commands:
            ambiguous_episodes += 1
        if has_semantic_source:
            semantic_source_episodes += 1
        if differs_from_last:
            non_last_command_episodes += 1
        if not (has_ambiguous_test_commands and has_semantic_source and differs_from_last):
            failures.append(
                {
                    "episode_id": episode_id,
                    "ambiguous_test_commands": has_ambiguous_test_commands,
                    "semantic_source_visible": has_semantic_source,
                    "correct_command_differs_from_last_command": differs_from_last,
                }
            )

    episode_count = len(by_episode)
    return {
        "episode_count": episode_count,
        "ambiguous_same_intent_episode_count": ambiguous_episodes,
        "semantic_source_episode_count": semantic_source_episodes,
        "correct_command_differs_from_last_command_episode_count": non_last_command_episodes,
        "aggregate_candidate_intent_counts": dict(same_intent_command_counts),
        "failure_examples": failures[:10],
        "passed": (
            episode_count > 0
            and ambiguous_episodes == episode_count
            and semantic_source_episodes == episode_count
            and non_last_command_episodes == episode_count
        ),
    }


def seal_external_trace_set(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    control_dir: str | Path = DEFAULT_CONTROL_DIR,
    version_name: str = "external_trace_v1",
    profile: str = "external_trace_v1",
    episodes_per_scenario: int = 8,
    reference_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    references = reference_paths if reference_paths is not None else DEFAULT_REFERENCES
    output_path = Path(output)
    control_path = Path(control_dir)
    seal_path = control_path / f"{version_name}.sealed"
    if seal_path.exists():
        raise FileExistsError(
            f"External trace set {version_name!r} is already sealed at {seal_path}."
        )

    manifest = build_debug_cortex_challenge(
        output_path,
        profile=profile,
        episodes_per_scenario=episodes_per_scenario,
    )
    challenge_path = output_path / "challenge.jsonl"
    leakage = build_leakage_audit(challenge_path)
    command_overlap = build_command_slot_overlap_audit(challenge_path, references)
    semantic_nn = build_semantic_nn_audit(challenge_path, references)
    semantic_necessity = build_semantic_necessity_audit(challenge_path)
    config_payload = {
        "version_name": version_name,
        "profile": profile,
        "episodes_per_scenario": episodes_per_scenario,
        "challenge_path": str(challenge_path.resolve()),
        "references": [str(Path(path)) for path in references],
    }
    sealed_config_hash = stable_config_hash(config_payload)
    extended_manifest = {
        **manifest,
        "sealed": True,
        "version_name": version_name,
        "sealed_config_hash": sealed_config_hash,
        "reference_datasets": [str(Path(path)) for path in references],
        "leakage_audit": "leakage_audit.json",
        "semantic_nn_audit": "semantic_nn_audit.json",
        "command_slot_overlap_audit": "command_slot_overlap_audit.json",
        "semantic_necessity_audit": "semantic_necessity_audit.json",
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "leakage_audit.json").write_text(
        json.dumps(leakage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_path / "command_slot_overlap_audit.json").write_text(
        json.dumps(command_overlap, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_path / "semantic_nn_audit.json").write_text(
        json.dumps(semantic_nn, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_path / "semantic_necessity_audit.json").write_text(
        json.dumps(semantic_necessity, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_path / "sealed_config_hash").write_text(sealed_config_hash + "\n", encoding="utf-8")
    (output_path / "manifest.json").write_text(
        json.dumps(extended_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    control_path.mkdir(parents=True, exist_ok=True)
    seal_path.write_text(
        json.dumps(
            {
                "version_name": version_name,
                "dataset_path": str(challenge_path.resolve()),
                "sealed_config_hash": sealed_config_hash,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return extended_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and seal Phase2G external trace set.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--control-dir", default=str(DEFAULT_CONTROL_DIR))
    parser.add_argument("--version-name", default="external_trace_v1")
    parser.add_argument("--profile", default="external_trace_v1")
    parser.add_argument("--episodes-per-scenario", type=int, default=8)
    parser.add_argument("--reference", action="append", default=[])
    args = parser.parse_args()
    manifest = seal_external_trace_set(
        output=args.output,
        control_dir=args.control_dir,
        version_name=args.version_name,
        profile=args.profile,
        episodes_per_scenario=args.episodes_per_scenario,
        reference_paths=args.reference or None,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
