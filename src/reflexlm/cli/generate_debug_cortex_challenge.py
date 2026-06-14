from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from reflexlm.data.jsonl import write_jsonl
from reflexlm.data.tasks import (
    TaskType,
    build_env,
    episode_metadata_for,
    rollout_env,
    scenario_templates_for,
)
from reflexlm.llm.candidate_features import command_intent_for_text
from reflexlm.models.features import serialize_state_as_text


def build_debug_cortex_challenge(
    output_dir: str | Path,
    *,
    profile: str = "debug_ood",
    episodes_per_scenario: int = 8,
    max_scan_multiplier: int = 12,
) -> dict[str, object]:
    output = Path(output_dir)
    scenarios = scenario_templates_for(TaskType.TEST_FAILURE, profile)
    if not scenarios:
        raise ValueError(f"No Debug Cortex scenarios for profile {profile!r}")
    target_per_scenario = {scenario: episodes_per_scenario for scenario in scenarios}
    counts: Counter[str] = Counter()
    records = []
    metadata_rows = []
    max_index = len(scenarios) * episodes_per_scenario * max_scan_multiplier

    for episode_index in range(max_index):
        metadata = episode_metadata_for(
            TaskType.TEST_FAILURE,
            episode_index,
            profile=profile,
            seed=0,
        )
        scenario = str(metadata["scenario_template"])
        if counts[scenario] >= target_per_scenario[scenario]:
            continue
        env = build_env(TaskType.TEST_FAILURE, episode_index, profile=profile)
        episode_records = rollout_env(env)
        records.extend(episode_records)
        metadata_rows.append(metadata)
        counts[scenario] += 1
        if all(counts[scenario_name] >= target for scenario_name, target in target_per_scenario.items()):
            break

    missing = {
        scenario: target_per_scenario[scenario] - counts[scenario]
        for scenario in scenarios
        if counts[scenario] < target_per_scenario[scenario]
    }
    if missing:
        raise RuntimeError(f"Could not generate requested Debug Cortex challenge coverage: {missing}")

    output.mkdir(parents=True, exist_ok=True)
    challenge_path = output / "challenge.jsonl"
    metadata_path = output / "episode_metadata.json"
    write_jsonl(challenge_path, records)
    metadata_path.write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")

    hidden_hint_leaks = 0
    scenario_template_leaks = 0
    for record in records:
        serialized = serialize_state_as_text(record.state)
        if "recovery_hint=" in serialized or (
            record.goal.recovery_hint and record.goal.recovery_hint in serialized
        ):
            hidden_hint_leaks += 1
        if "scenario_template" in serialized:
            scenario_template_leaks += 1

    command_targets = Counter(
        record.action.command
        for record in records
        if record.action and record.action.command
    )
    command_intents = Counter(
        command_intent_for_text(record.action.command)
        for record in records
        if record.action and record.action.command
    )
    manifest = {
        "profile": profile,
        "task_type": TaskType.TEST_FAILURE.value,
        "episode_source": (
            "synthetic_debug_cortex_profile"
            if profile != "quasi_real_terminal"
            else "quasi_real_local_project_observable_paths"
        ),
        "completion_definition": (
            "episode completes when the policy matches the oracle action sequence "
            "under the fixed allowlist-only closed-loop environment"
        ),
        "baseline_input_fairness": (
            "all policies reconstruct the same episode ids, visible state fields, "
            "candidate command allowlist, candidate file slots, and hidden-hint exclusions"
        ),
        "episodes_per_scenario": episodes_per_scenario,
        "episode_count": len({record.episode_id for record in records}),
        "record_count": len(records),
        "scenario_counts": dict(counts),
        "variant_counts": dict(Counter(row["variant"] for row in metadata_rows)),
        "command_targets": dict(command_targets),
        "command_intents": dict(command_intents),
        "metadata_path": metadata_path.name,
        "challenge_path": challenge_path.name,
        "model_visible_hidden_hint_leaks": hidden_hint_leaks,
        "model_visible_scenario_template_leaks": scenario_template_leaks,
        "passed": hidden_hint_leaks == 0 and scenario_template_leaks == 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a focused Debug Cortex OOD challenge.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="debug_ood")
    parser.add_argument("--episodes-per-scenario", type=int, default=8)
    args = parser.parse_args()
    manifest = build_debug_cortex_challenge(
        args.output,
        profile=args.profile,
        episodes_per_scenario=args.episodes_per_scenario,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
