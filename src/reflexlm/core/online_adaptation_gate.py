from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import torch

from reflexlm.core.dataset import read_reflexcore_jsonl, write_reflexcore_jsonl
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
from reflexlm.core.online_adaptation import (
    ReflexCoreOnlineAdaptationConfig,
    adapt_reflexcore_from_experience,
)
from reflexlm.core.sandbox_benchmark import evaluate_reflexcore_real_sandbox_families
from reflexlm.core.schema import ReflexCoreTrainingExample, dataset_hash


@dataclass(slots=True)
class ReflexCoreOnlineAdaptationGateConfig:
    checkpoint_path: Path
    dataset_path: Path
    output_dir: Path
    split_strategy: str = "episode_holdout"
    split_seed: int = 13
    train_episode_count: int = 4
    retention_episode_count: int = 1
    holdout_episode_count: int | None = None
    holdout_families: tuple[str, ...] = ()
    max_retention_loss_increase: float = 0.0
    max_holdout_loss_increase: float = 0.0
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 1e-4
    device: str = "cpu"
    sequence_mode: bool = True
    max_sequence_len: int | None = 8
    max_text_tokens: int = 128
    trainable_scope: str = "all"
    behavior_eval_variants: int = 0
    behavior_eval_start_variant: int = 0
    behavior_eval_max_steps: int = 4


@dataclass(slots=True)
class ReflexCoreFamilyHoldoutMatrixConfig:
    checkpoint_path: Path
    dataset_path: Path
    output_dir: Path
    split_seed: int = 13
    train_episode_count: int | None = None
    retention_episode_count: int = 1
    holdout_families: tuple[str, ...] = ()
    max_retention_loss_increase: float = 0.0
    max_holdout_loss_increase: float = 0.0
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 1e-4
    device: str = "cpu"
    sequence_mode: bool = True
    max_sequence_len: int | None = 8
    max_text_tokens: int = 128
    trainable_scope: str = "all"
    behavior_eval_variants: int = 0
    behavior_eval_start_variant: int = 0
    behavior_eval_max_steps: int = 4
    require_behavior_capability: bool = False
    min_behavior_success_rate: float = 0.0


@dataclass(slots=True)
class ReflexCoreOnlineAdaptationSplit:
    train: list[ReflexCoreTrainingExample]
    retention: list[ReflexCoreTrainingExample]
    holdout: list[ReflexCoreTrainingExample]
    train_episode_ids: list[str]
    retention_episode_ids: list[str]
    holdout_episode_ids: list[str]


def run_family_holdout_matrix(
    config: ReflexCoreFamilyHoldoutMatrixConfig,
) -> dict[str, object]:
    """Run one disjoint online-adaptation gate per held-out task family."""

    examples = read_reflexcore_jsonl(config.dataset_path)
    grouped = _group_by_episode(examples)
    families = sorted({_episode_family(episode_id) for episode_id in grouped})
    selected_families = (
        [family for family in families if family in set(config.holdout_families)]
        if config.holdout_families
        else families
    )
    if not selected_families:
        raise ValueError("family holdout matrix requires at least one family")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, family in enumerate(selected_families):
        heldout_ids = [
            episode_id
            for episode_id in grouped
            if _episode_family(episode_id) == family
        ]
        nonheldout_count = len(grouped) - len(heldout_ids)
        if nonheldout_count <= 0:
            raise ValueError(f"cannot hold out only available family: {family}")
        train_count = (
            config.train_episode_count
            if config.train_episode_count is not None
            else max(1, nonheldout_count - config.retention_episode_count)
        )
        if train_count + config.retention_episode_count > nonheldout_count:
            raise ValueError(
                f"train + retention episodes exceed non-heldout episodes for {family}"
            )
        family_dir = config.output_dir / f"{index:02d}_{_safe_name(family)}"
        family_report = run_online_adaptation_gate(
            ReflexCoreOnlineAdaptationGateConfig(
                checkpoint_path=config.checkpoint_path,
                dataset_path=config.dataset_path,
                output_dir=family_dir,
                split_strategy="family_holdout",
                split_seed=config.split_seed + index,
                train_episode_count=train_count,
                retention_episode_count=config.retention_episode_count,
                holdout_episode_count=None,
                holdout_families=(family,),
                max_retention_loss_increase=config.max_retention_loss_increase,
                max_holdout_loss_increase=config.max_holdout_loss_increase,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                device=config.device,
                sequence_mode=config.sequence_mode,
                max_sequence_len=config.max_sequence_len,
                max_text_tokens=config.max_text_tokens,
                trainable_scope=config.trainable_scope,
            )
        )
        behavior_report = (
            _evaluate_behavior_regression(
                checkpoint_path=config.checkpoint_path,
                adapted_checkpoint_path=Path(
                    str(family_report["adaptation"]["adapted_checkpoint"])
                ),
                output_dir=family_dir / "behavior",
                family=family,
                variants=config.behavior_eval_variants,
                start_variant=config.behavior_eval_start_variant,
                max_steps=config.behavior_eval_max_steps,
                require_capability=config.require_behavior_capability,
                min_success_rate=config.min_behavior_success_rate,
                device=config.device,
            )
            if config.behavior_eval_variants > 0
            else None
        )
        results.append(_family_result_summary(family, family_report, family_dir, behavior_report))
    passed_count = sum(1 for result in results if result["passed"])
    behavior_passed_count = sum(
        1 for result in results if result.get("behavior_passed") is True
    )
    behavior_capability_passed_count = sum(
        1 for result in results if result.get("behavior_capability_passed") is True
    )
    failed_families = [result["family"] for result in results if not result["passed"]]
    holdout_deltas = [
        float(result["holdout_loss_delta"])
        for result in results
        if result["holdout_loss_delta"] is not None
    ]
    behavior_enabled = config.behavior_eval_variants > 0
    report = {
        "config": _family_matrix_json_config(config),
        "dataset": str(config.dataset_path),
        "dataset_examples": len(examples),
        "dataset_hash": dataset_hash(examples),
        "episode_count": len(grouped),
        "family_count": len(selected_families),
        "families": selected_families,
        "results": results,
        "passed_count": passed_count,
        "behavior_passed_count": behavior_passed_count if config.behavior_eval_variants > 0 else None,
        "behavior_capability_passed_count": (
            behavior_capability_passed_count
            if config.behavior_eval_variants > 0 and config.require_behavior_capability
            else None
        ),
        "failed_families": failed_families,
        "pass_rate": passed_count / max(len(results), 1),
        "min_holdout_loss_delta": min(holdout_deltas) if holdout_deltas else None,
        "passed": passed_count == len(results)
        and (
            not behavior_enabled
            or behavior_passed_count == len(results)
        ),
        "free_shell_generation": False,
        "gui_or_vision": False,
        "claim_boundary": _family_matrix_claim_boundary(behavior_enabled),
    }
    (config.output_dir / "family_holdout_matrix_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def run_online_adaptation_gate(
    config: ReflexCoreOnlineAdaptationGateConfig,
) -> dict[str, object]:
    """Run online adaptation with disjoint train/retention/holdout episodes."""

    examples = read_reflexcore_jsonl(config.dataset_path)
    split = split_online_adaptation_examples(
        examples,
        split_strategy=config.split_strategy,
        split_seed=config.split_seed,
        train_episode_count=config.train_episode_count,
        retention_episode_count=config.retention_episode_count,
        holdout_episode_count=config.holdout_episode_count,
        holdout_families=config.holdout_families,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = config.output_dir / "train.jsonl"
    retention_path = config.output_dir / "retention.jsonl"
    holdout_path = config.output_dir / "holdout.jsonl"
    write_reflexcore_jsonl(train_path, split.train)
    write_reflexcore_jsonl(retention_path, split.retention)
    write_reflexcore_jsonl(holdout_path, split.holdout)
    adaptation = adapt_reflexcore_from_experience(
        ReflexCoreOnlineAdaptationConfig(
            checkpoint_path=config.checkpoint_path,
            experience_path=train_path,
            output_dir=config.output_dir / "adapted",
            retention_path=retention_path,
            holdout_path=holdout_path,
            max_retention_loss_increase=config.max_retention_loss_increase,
            max_holdout_loss_increase=config.max_holdout_loss_increase,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            device=config.device,
            sequence_mode=config.sequence_mode,
            max_sequence_len=config.max_sequence_len,
            max_text_tokens=config.max_text_tokens,
            trainable_scope=config.trainable_scope,
        )
    )
    report = {
        "config": _json_config(config),
        "dataset": str(config.dataset_path),
        "dataset_examples": len(examples),
        "dataset_hash": dataset_hash(examples),
        "split": _split_report(split),
        "adaptation": adaptation,
        "passed": bool(adaptation["accepted"]),
        "free_shell_generation": False,
        "gui_or_vision": False,
        "claim_boundary": (
            "This gate adapts ReflexCore V0 on disjoint bounded sensory-motor "
            "episodes only. It does not execute actions and does not expand "
            "beyond terminal/process/filesystem/time sandbox scope."
        ),
    }
    (config.output_dir / "online_adaptation_gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def split_online_adaptation_examples(
    examples: list[ReflexCoreTrainingExample],
    *,
    split_strategy: str = "episode_holdout",
    split_seed: int = 13,
    train_episode_count: int = 4,
    retention_episode_count: int = 1,
    holdout_episode_count: int | None = None,
    holdout_families: tuple[str, ...] = (),
) -> ReflexCoreOnlineAdaptationSplit:
    if train_episode_count <= 0:
        raise ValueError("train_episode_count must be positive")
    if retention_episode_count < 0:
        raise ValueError("retention_episode_count must be non-negative")
    if holdout_episode_count is not None and holdout_episode_count <= 0:
        raise ValueError("holdout_episode_count must be positive when provided")
    grouped = _group_by_episode(examples)
    if split_strategy == "episode_holdout":
        return _episode_holdout_split(
            grouped,
            split_seed=split_seed,
            train_episode_count=train_episode_count,
            retention_episode_count=retention_episode_count,
            holdout_episode_count=holdout_episode_count,
        )
    if split_strategy == "family_holdout":
        return _family_holdout_split(
            grouped,
            split_seed=split_seed,
            train_episode_count=train_episode_count,
            retention_episode_count=retention_episode_count,
            holdout_episode_count=holdout_episode_count,
            holdout_families=holdout_families,
        )
    raise ValueError(f"unknown split_strategy: {split_strategy}")


def _episode_holdout_split(
    grouped: dict[str, list[ReflexCoreTrainingExample]],
    *,
    split_seed: int,
    train_episode_count: int,
    retention_episode_count: int,
    holdout_episode_count: int | None,
) -> ReflexCoreOnlineAdaptationSplit:
    episode_ids = sorted(grouped)
    rng = Random(split_seed)
    rng.shuffle(episode_ids)
    train_ids = episode_ids[:train_episode_count]
    retention_start = train_episode_count
    retention_end = retention_start + retention_episode_count
    retention_ids = episode_ids[retention_start:retention_end]
    remaining = episode_ids[retention_end:]
    holdout_ids = remaining if holdout_episode_count is None else remaining[:holdout_episode_count]
    return _build_split(grouped, train_ids, retention_ids, holdout_ids)


def _family_holdout_split(
    grouped: dict[str, list[ReflexCoreTrainingExample]],
    *,
    split_seed: int,
    train_episode_count: int,
    retention_episode_count: int,
    holdout_episode_count: int | None,
    holdout_families: tuple[str, ...],
) -> ReflexCoreOnlineAdaptationSplit:
    family_to_episodes: dict[str, list[str]] = defaultdict(list)
    for episode_id in sorted(grouped):
        family_to_episodes[_episode_family(episode_id)].append(episode_id)
    families = sorted(family_to_episodes)
    if not families:
        raise ValueError("dataset must contain at least one episode")
    if holdout_families:
        selected_families = [family for family in families if family in set(holdout_families)]
        if not selected_families:
            raise ValueError("no requested holdout_families are present in dataset")
    else:
        rng = Random(split_seed)
        selected_families = [rng.choice(families)]
    holdout_pool = [
        episode_id
        for family in selected_families
        for episode_id in family_to_episodes[family]
    ]
    holdout_ids = (
        sorted(holdout_pool)
        if holdout_episode_count is None
        else sorted(holdout_pool)[:holdout_episode_count]
    )
    remaining = sorted(set(grouped) - set(holdout_ids))
    rng = Random(split_seed + 1)
    rng.shuffle(remaining)
    train_ids = remaining[:train_episode_count]
    retention_start = train_episode_count
    retention_end = retention_start + retention_episode_count
    retention_ids = remaining[retention_start:retention_end]
    return _build_split(grouped, train_ids, retention_ids, holdout_ids)


def _build_split(
    grouped: dict[str, list[ReflexCoreTrainingExample]],
    train_ids: list[str],
    retention_ids: list[str],
    holdout_ids: list[str],
) -> ReflexCoreOnlineAdaptationSplit:
    _validate_disjoint_ids(train_ids, retention_ids, holdout_ids)
    if not train_ids:
        raise ValueError("split must contain at least one train episode")
    if not holdout_ids:
        raise ValueError("split must contain at least one holdout episode")
    train = _examples_for_ids(grouped, train_ids)
    retention = _examples_for_ids(grouped, retention_ids)
    holdout = _examples_for_ids(grouped, holdout_ids)
    if not train:
        raise ValueError("train split is empty")
    if not holdout:
        raise ValueError("holdout split is empty")
    return ReflexCoreOnlineAdaptationSplit(
        train=train,
        retention=retention,
        holdout=holdout,
        train_episode_ids=sorted(train_ids),
        retention_episode_ids=sorted(retention_ids),
        holdout_episode_ids=sorted(holdout_ids),
    )


def _group_by_episode(
    examples: list[ReflexCoreTrainingExample],
) -> dict[str, list[ReflexCoreTrainingExample]]:
    grouped: dict[str, list[ReflexCoreTrainingExample]] = defaultdict(list)
    for example in examples:
        grouped[example.episode_id].append(example)
    if not grouped:
        raise ValueError("dataset must contain at least one example")
    return {
        episode_id: sorted(items, key=lambda item: item.t)
        for episode_id, items in grouped.items()
    }


def _examples_for_ids(
    grouped: dict[str, list[ReflexCoreTrainingExample]],
    episode_ids: list[str],
) -> list[ReflexCoreTrainingExample]:
    return [
        example
        for episode_id in sorted(episode_ids)
        for example in grouped[episode_id]
    ]


def _validate_disjoint_ids(*groups: list[str]) -> None:
    seen: set[str] = set()
    for group in groups:
        for episode_id in group:
            if episode_id in seen:
                raise ValueError(f"episode appears in multiple splits: {episode_id}")
            seen.add(episode_id)


def _episode_family(episode_id: str) -> str:
    match = re.match(r"^(?P<family>.+)-\d+$", episode_id)
    if match:
        return match.group("family")
    if "::" in episode_id:
        return episode_id.split("::", 1)[0]
    return episode_id


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "family"


def _family_result_summary(
    family: str,
    report: dict[str, object],
    output_dir: Path,
    behavior_report: dict[str, object] | None = None,
) -> dict[str, object]:
    adaptation = report["adaptation"]
    split = report["split"]
    assert isinstance(adaptation, dict)
    assert isinstance(split, dict)
    payload = {
        "family": family,
        "passed": bool(report["passed"]),
        "output_dir": str(output_dir),
        "train_examples": split["train_examples"],
        "retention_examples": split["retention_examples"],
        "holdout_examples": split["holdout_examples"],
        "train_episode_ids": split["train_episode_ids"],
        "retention_episode_ids": split["retention_episode_ids"],
        "holdout_episode_ids": split["holdout_episode_ids"],
        "before_loss": adaptation["before_loss"],
        "after_loss": adaptation["after_loss"],
        "loss_delta": adaptation["loss_delta"],
        "prediction_error_loss_delta": adaptation.get("prediction_error_loss_delta"),
        "live_prediction_error_examples": adaptation.get("live_prediction_error_examples"),
        "live_prediction_error_target_mean": adaptation.get(
            "live_prediction_error_target_mean"
        ),
        "prediction_error_motor_probe": adaptation.get("prediction_error_motor_probe"),
        "retention_loss_delta": adaptation["retention_loss_delta"],
        "holdout_loss_delta": adaptation["holdout_loss_delta"],
        "holdout_gate": adaptation["holdout_gate"],
        "trainable_scope": adaptation["trainable_scope"],
        "trainable_parameter_count": adaptation["trainable_parameter_count"],
        "frozen_parameter_count": adaptation["frozen_parameter_count"],
        "accepted": adaptation["accepted"],
        "rejected_reason": adaptation["rejected_reason"],
        "adapted_model_hash": adaptation["adapted_model_hash"],
    }
    if behavior_report is not None:
        payload["behavior"] = behavior_report
        payload["behavior_passed"] = behavior_report["passed"]
        payload["behavior_non_regression_passed"] = behavior_report[
            "non_regression_passed"
        ]
        payload["behavior_capability_passed"] = behavior_report[
            "capability_passed"
        ]
        payload["behavior_min_success_rate"] = behavior_report[
            "min_success_rate"
        ]
        payload["base_success_rate"] = behavior_report["base_success_rate"]
        payload["adapted_success_rate"] = behavior_report["adapted_success_rate"]
        payload["success_rate_delta"] = behavior_report["success_rate_delta"]
        payload["passed"] = bool(payload["passed"]) and bool(behavior_report["passed"])
        if not behavior_report["passed"] and payload["rejected_reason"] is None:
            payload["rejected_reason"] = behavior_report["rejected_reason"]
    return payload


def _family_matrix_claim_boundary(behavior_enabled: bool) -> str:
    if behavior_enabled:
        return (
            "This matrix repeats bounded online adaptation while holding out "
            "entire task families, then executes decoded model actions only "
            "inside the real temporary terminal/process/filesystem/time sandbox "
            "for behavior regression and capability checks. It does not enable "
            "GUI control, vision, or unrestricted shell generation."
        )
    return (
        "This matrix repeats bounded online adaptation while holding out "
        "entire task families. It consumes recorded terminal/process/"
        "filesystem/time transitions and does not execute model actions."
    )


def _evaluate_behavior_regression(
    *,
    checkpoint_path: Path,
    adapted_checkpoint_path: Path,
    output_dir: Path,
    family: str,
    variants: int,
    start_variant: int,
    max_steps: int,
    require_capability: bool,
    min_success_rate: float,
    device: str,
) -> dict[str, object]:
    if min_success_rate < 0.0 or min_success_rate > 1.0:
        raise ValueError("min_success_rate must be between 0 and 1")
    base_model = _load_model(checkpoint_path, device=device)
    adapted_model = _load_model(adapted_checkpoint_path, device=device)
    base_report = evaluate_reflexcore_real_sandbox_families(
        base_model,
        output_dir=output_dir / "base",
        families=(family,),
        variants=variants,
        start_variant=start_variant,
        max_steps=max_steps,
    )
    adapted_report = evaluate_reflexcore_real_sandbox_families(
        adapted_model,
        output_dir=output_dir / "adapted",
        families=(family,),
        variants=variants,
        start_variant=start_variant,
        max_steps=max_steps,
    )
    base_success = float(base_report["overall"]["success_rate"])
    adapted_success = float(adapted_report["overall"]["success_rate"])
    delta = adapted_success - base_success
    non_regression_passed = adapted_success >= base_success
    capability_passed = (
        adapted_success >= min_success_rate
        if require_capability
        else True
    )
    if not non_regression_passed:
        rejected_reason = "behavior_regression"
    elif not capability_passed:
        rejected_reason = "behavior_capability_below_minimum"
    else:
        rejected_reason = None
    return {
        "family": family,
        "variants": variants,
        "start_variant": start_variant,
        "max_steps": max_steps,
        "require_capability": require_capability,
        "min_success_rate": min_success_rate,
        "base_success_rate": base_success,
        "adapted_success_rate": adapted_success,
        "success_rate_delta": delta,
        "non_regression_passed": non_regression_passed,
        "capability_passed": capability_passed,
        "passed": non_regression_passed and capability_passed,
        "rejected_reason": rejected_reason,
        "base": base_report,
        "adapted": adapted_report,
    }


def _load_model(path: Path, *, device: str) -> ReflexCoreV0:
    checkpoint = torch.load(path, map_location=device)
    model = ReflexCoreV0(ReflexCoreV0Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


def _split_report(split: ReflexCoreOnlineAdaptationSplit) -> dict[str, object]:
    return {
        "train_examples": len(split.train),
        "retention_examples": len(split.retention),
        "holdout_examples": len(split.holdout),
        "train_episode_ids": split.train_episode_ids,
        "retention_episode_ids": split.retention_episode_ids,
        "holdout_episode_ids": split.holdout_episode_ids,
        "train_hash": dataset_hash(split.train),
        "retention_hash": dataset_hash(split.retention) if split.retention else None,
        "holdout_hash": dataset_hash(split.holdout),
        "disjoint_episodes": True,
        "holdout_families": sorted({_episode_family(item) for item in split.holdout_episode_ids}),
    }


def _json_config(config: ReflexCoreOnlineAdaptationGateConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["checkpoint_path"] = str(config.checkpoint_path)
    payload["dataset_path"] = str(config.dataset_path)
    payload["output_dir"] = str(config.output_dir)
    payload["holdout_families"] = list(config.holdout_families)
    return payload


def _family_matrix_json_config(config: ReflexCoreFamilyHoldoutMatrixConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["checkpoint_path"] = str(config.checkpoint_path)
    payload["dataset_path"] = str(config.dataset_path)
    payload["output_dir"] = str(config.output_dir)
    payload["holdout_families"] = list(config.holdout_families)
    return payload
