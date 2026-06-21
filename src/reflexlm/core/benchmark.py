from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from reflexlm.core.dataset import (
    build_reflexcore_examples,
    split_hashes,
    write_reflexcore_jsonl,
)
from reflexlm.core.schema import ReflexCoreTrainingExample, dataset_hash
from reflexlm.data.jsonl import (
    split_records_by_episode,
    split_records_by_episode_fingerprint,
    split_records_by_scenario_holdout,
    write_jsonl,
)
from reflexlm.data.tasks import build_episode_metadata, build_env, rollout_env
from reflexlm.models.features import StateVectorizer
from reflexlm.runtime.oracle import RuleOracle
from reflexlm.schema import TaskType, TrajectoryRecord


@dataclass(slots=True)
class ReflexCoreBenchmarkConfig:
    output_dir: Path
    profile: str = "default"
    episodes_per_task: int = 6
    split_strategy: str = "scenario_holdout"
    seed: int = 13
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    hash_bins: int = 256
    vocab_size: int = 4096
    max_text_tokens: int = 64


def build_reflexcore_benchmark(config: ReflexCoreBenchmarkConfig) -> dict[str, object]:
    output_dir = config.output_dir
    trajectory_dir = output_dir / "trajectories"
    reflexcore_dir = output_dir / "reflexcore"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    reflexcore_dir.mkdir(parents=True, exist_ok=True)

    records = generate_reflexcore_benchmark_records(
        profile=config.profile,
        episodes_per_task=config.episodes_per_task,
    )
    metadata = build_episode_metadata(records, profile=config.profile, seed=config.seed)
    trajectory_splits = split_benchmark_records(
        records,
        metadata=metadata,
        split_strategy=config.split_strategy,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    write_jsonl(trajectory_dir / "all.jsonl", records)
    for name, split_records in trajectory_splits.items():
        write_jsonl(trajectory_dir / f"{name}.jsonl", split_records)

    vectorizer = StateVectorizer(hash_bins=config.hash_bins)
    reflexcore_splits: dict[str, list[ReflexCoreTrainingExample]] = {}
    for name, split_records in trajectory_splits.items():
        examples = build_reflexcore_examples(
            split_records,
            vectorizer=vectorizer,
            vocab_size=config.vocab_size,
            max_text_tokens=config.max_text_tokens,
        )
        reflexcore_splits[name] = examples
        write_reflexcore_jsonl(reflexcore_dir / f"{name}.jsonl", examples)
    all_examples = [
        example
        for split_examples in reflexcore_splits.values()
        for example in split_examples
    ]
    write_reflexcore_jsonl(reflexcore_dir / "all.jsonl", all_examples)

    episode_metadata_path = output_dir / "episode_metadata.json"
    episode_metadata_path.write_text(
        json.dumps(
            [metadata[key] for key in sorted(metadata)],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "config": _json_config(config),
        "scope": "terminal_process_filesystem_time_only",
        "free_shell_generation": False,
        "gui_or_vision": False,
        "trajectory": {
            "record_count": len(records),
            "episode_count": len({record.episode_id for record in records}),
            "hash": trajectory_hash(records),
            "split_hashes": {
                name: trajectory_hash(split_records)
                for name, split_records in trajectory_splits.items()
            },
            "split_counts": {
                name: len(split_records)
                for name, split_records in trajectory_splits.items()
            },
            "split_episode_counts": {
                name: len({record.episode_id for record in split_records})
                for name, split_records in trajectory_splits.items()
            },
        },
        "reflexcore": {
            "example_count": len(all_examples),
            "hash": dataset_hash(all_examples),
            "split_hashes": split_hashes(reflexcore_splits),
            "split_counts": {
                name: len(split_examples)
                for name, split_examples in reflexcore_splits.items()
            },
            "vectorizer": {
                "hash_bins": config.hash_bins,
                "vector_dim": vectorizer.vector_dim,
            },
            "vocab_size": config.vocab_size,
            "max_text_tokens": config.max_text_tokens,
        },
        "paths": {
            "trajectories": "trajectories",
            "reflexcore": "reflexcore",
            "episode_metadata": "episode_metadata.json",
        },
        "recommended_gate": {
            "offline_eval_cli": (
                "eval-reflexcore-v0 --compare-baselines "
                "--require-beats-baseline prompt_only_heuristic "
                "--require-world-model-improvement "
                "--require-prediction-error-improvement"
            ),
            "closed_loop_eval_cli": (
                "eval-reflexcore-closed-loop --compare-baselines "
                "--require-beats-baseline prompt_only_heuristic"
            ),
            "stability_eval_cli": (
                "run-reflexcore-stability --seed 13 --seed 17 --seed 23 "
                "--required-baseline prompt_only_heuristic --min-pass-rate 1.0"
            ),
            "profile_transfer_stability_cli": (
                "run-reflexcore-stability --profile default --eval-profile hard "
                "--seed 13 --seed 17 --seed 23 "
                "--required-baseline prompt_only_heuristic --min-pass-rate 1.0"
            ),
            "profile_matrix_stability_cli": (
                "run-reflexcore-profile-matrix --profile default "
                "--eval-profile default --eval-profile hard --eval-profile wide_ood "
                "--seed 13 --seed 17 --seed 23 "
                "--required-baseline prompt_only_heuristic "
                "--min-pass-rate 1.0 --min-profile-pass-rate 1.0"
            ),
            "local_feasibility_cli": (
                "run-reflexcore-local-feasibility "
                "--config configs/reflexcore/local.yaml "
                "--episodes-per-task 1 --epochs 1 --batch-size 1 "
                "--sequence-mode --max-sequence-len 8 "
                "--min-parameters 20000000 --max-parameters 100000000"
            ),
            "local_stability_cli": (
                "run-reflexcore-stability --config configs/reflexcore/local.yaml "
                "--profile default --seed 13 --seed 17 --seed 23 "
                "--episodes-per-task 6 --vocab-size 4096 "
                "--max-text-tokens 128 --epochs 8 --batch-size 4 "
                "--sequence-mode --max-sequence-len 8 "
                "--closed-loop-episodes-per-task 1 "
                "--required-baseline prompt_only_heuristic "
                "--min-parameters 20000000 --max-parameters 100000000 "
                "--min-pass-rate 1.0"
            ),
            "local_profile_matrix_cli": (
                "run-reflexcore-profile-matrix --config configs/reflexcore/local.yaml "
                "--profile default --eval-profile default --eval-profile hard "
                "--eval-profile wide_ood --seed 13 --seed 17 --seed 23 "
                "--episodes-per-task 6 --vocab-size 4096 "
                "--max-text-tokens 128 --epochs 8 --batch-size 4 "
                "--sequence-mode --max-sequence-len 8 "
                "--closed-loop-episodes-per-task 1 "
                "--required-baseline prompt_only_heuristic "
                "--min-parameters 20000000 --max-parameters 100000000 "
                "--min-pass-rate 1.0 --min-profile-pass-rate 1.0"
            ),
            "real_sandbox_eval_cli": (
                "eval-reflexcore-real-sandbox --checkpoint <reflexcore_v0.pt> "
                "--output-dir <real_sandbox_eval_dir> --max-steps 4 "
                "--require-beats-baseline prompt_only_heuristic"
            ),
            "online_experience_cli": (
                "run-reflexcore-sandbox --checkpoint <reflexcore_v0.pt> "
                "--sandbox-root <sandbox_dir> --steps 2 --loop "
                "--write-experience <experience.jsonl> "
                "--episode-id <model_rollout_episode>"
            ),
            "online_adaptation_cli": (
                "adapt-reflexcore-from-experience "
                "--checkpoint <reflexcore_v0.pt> "
                "--experience <experience.jsonl> "
                "--output-dir <online_adaptation_dir> "
                "--epochs 2 --batch-size 1 --learning-rate 0.001 "
                "--sequence-mode"
            ),
            "prediction_error_diagnostic_cli": (
                "eval-reflexcore-prediction-error "
                "--checkpoint <reflexcore_v0.pt> "
                "--dataset <reflexcore_eval.jsonl> "
                "--output-dir <prediction_error_report_dir> "
                "--sequence-mode --min-relative-improvement 0.0"
            ),
            "real_sandbox_dataset_cli": (
                "build-reflexcore-real-sandbox-dataset "
                "--output <real_sandbox_train.jsonl> --work-dir <sandbox_work_dir> "
                "--variants 12 --start-variant 1 --vocab-size 512 "
                "--max-text-tokens 64"
            ),
            "real_sandbox_adaptation_cli": (
                "run-reflexcore-real-sandbox-adaptation "
                "--config configs/reflexcore/local.yaml "
                "--output-dir <real_sandbox_adaptation_dir> "
                "--episodes-per-task 12 --vocab-size 4096 "
                "--max-text-tokens 128 --epochs 12 --batch-size 4 "
                "--sequence-mode --max-sequence-len 8 "
                "--real-sandbox-variants 12 --real-sandbox-start-variant 1 "
                "--min-parameters 20000000 --max-parameters 100000000"
            ),
            "real_sandbox_adaptation_matrix_cli": (
                "run-reflexcore-real-sandbox-adaptation-matrix "
                "--config configs/reflexcore/local.yaml "
                "--output-dir <real_sandbox_adaptation_matrix_dir> "
                "--seed 13 --seed 17 --seed 23 "
                "--episodes-per-task 12 --vocab-size 4096 "
                "--max-text-tokens 128 --epochs 12 --batch-size 4 "
                "--sequence-mode --max-sequence-len 8 "
                "--real-sandbox-variants 12 --real-sandbox-start-variant 1 "
                "--min-parameters 20000000 --max-parameters 100000000 "
                "--min-pass-rate 1.0"
            ),
            "real_sandbox_adaptation_profile_matrix_cli": (
                "run-reflexcore-real-sandbox-adaptation-profile-matrix "
                "--config configs/reflexcore/local_pe_calibrated.yaml "
                "--output-dir <real_sandbox_adaptation_profile_matrix_dir> "
                "--profile default --eval-profile default --eval-profile hard "
                "--eval-profile wide_ood --seed 13 --seed 17 --seed 23 "
                "--episodes-per-task 12 --vocab-size 4096 "
                "--max-text-tokens 128 --epochs 12 --batch-size 4 "
                "--sequence-mode --max-sequence-len 8 "
                "--real-sandbox-variants 12 --real-sandbox-start-variant 1 "
                "--synthetic-repeat 3 --real-sandbox-repeat 1 "
                "--min-parameters 20000000 --max-parameters 100000000 "
                "--min-pass-rate 1.0 --min-profile-pass-rate 1.0"
            ),
            "claim_boundary": (
                "Passing these gates only supports bounded terminal/process/"
                "filesystem/time behavior, not GUI or production autonomy."
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def generate_reflexcore_benchmark_records(
    *,
    profile: str,
    episodes_per_task: int,
) -> list[TrajectoryRecord]:
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    oracle = RuleOracle()
    records: list[TrajectoryRecord] = []
    for task_type in TaskType:
        for episode_index in range(episodes_per_task):
            env = build_env(task_type, episode_index, profile=profile)
            records.extend(rollout_env(env, policy=oracle))
    return sorted(records, key=lambda item: (item.episode_id, item.t))


def split_benchmark_records(
    records: list[TrajectoryRecord],
    *,
    metadata: dict[str, dict[str, object]],
    split_strategy: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[TrajectoryRecord]]:
    if split_strategy == "episode_random":
        return split_records_by_episode(
            records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    if split_strategy == "episode_fingerprint":
        return split_records_by_episode_fingerprint(
            records,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    if split_strategy == "scenario_holdout":
        return split_records_by_scenario_holdout(
            records,
            episode_metadata=metadata,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
    raise ValueError(f"unsupported split strategy: {split_strategy}")


def trajectory_hash(records: list[TrajectoryRecord]) -> str:
    payload = [
        record.model_dump(mode="json")
        for record in sorted(records, key=lambda item: (item.episode_id, item.t))
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_config(config: ReflexCoreBenchmarkConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    return payload
