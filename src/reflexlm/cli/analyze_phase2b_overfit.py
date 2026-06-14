from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reflexlm.cli.analyze_phase2b_generalization import (
    _action_tuple,
    _build_test_sft_rows,
    _metadata_by_episode,
    _read_jsonl_dicts,
    _scenario_key,
    analyze_phase2b_generalization,
)


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:-]+")
_DYNAMIC_SECTIONS = {
    "Legal action mask:",
    "Receptor state:",
    "Candidate commands:",
    "Candidate files:",
    "Synaptic state:",
}
_STOP_TOKENS = {
    "none",
    "null",
    "true",
    "false",
    "empty",
    "workspace",
    "var",
    "tmp",
    "logs",
    "log",
    "txt",
    "py",
    "json",
}


def _dynamic_prompt_values(prompt: str) -> list[str]:
    values: list[str] = []
    active = False
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in _DYNAMIC_SECTIONS:
            active = True
            continue
        if "=" not in line and line.endswith(":") and line not in _DYNAMIC_SECTIONS:
            active = False
            continue
        if not active:
            continue
        if "=" in line:
            _, value = line.split("=", 1)
            values.append(value.strip())
        elif line.startswith("- "):
            values.append(line[2:].strip())
    return values


def _semantic_tokens(prompt: str) -> frozenset[str]:
    tokens: set[str] = set()
    for value in _dynamic_prompt_values(prompt):
        for token in _TOKEN_RE.findall(value.lower()):
            token = token.strip("._-:/")
            if len(token) <= 1:
                continue
            if token.isdigit():
                continue
            if token in _STOP_TOKENS:
                continue
            tokens.add(token)
    return frozenset(tokens)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _nearest_neighbor_report(
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    top_k: int = 20,
) -> dict[str, Any]:
    train_index = [
        {
            "example_id": row.get("example_id"),
            "episode_id": row.get("episode_id"),
            "task_type": row.get("task_type"),
            "target_text": row.get("target_text"),
            "action_tuple": _action_tuple(row),
            "tokens": _semantic_tokens(str(row.get("user_prompt") or "")),
        }
        for row in train_rows
    ]
    similarities: list[float] = []
    per_task: dict[str, list[float]] = defaultdict(list)
    top_matches: list[dict[str, Any]] = []
    for test_row in test_rows:
        test_tokens = _semantic_tokens(str(test_row.get("user_prompt") or ""))
        best_score = 0.0
        best_train: dict[str, Any] | None = None
        for train_row in train_index:
            score = _jaccard(test_tokens, train_row["tokens"])
            if score > best_score:
                best_score = score
                best_train = train_row
        task_type = str(test_row.get("task_type") or "")
        similarities.append(best_score)
        per_task[task_type].append(best_score)
        if best_train is not None:
            top_matches.append(
                {
                    "test_example_id": test_row.get("example_id"),
                    "test_episode_id": test_row.get("episode_id"),
                    "task_type": task_type,
                    "nearest_train_example_id": best_train.get("example_id"),
                    "nearest_train_episode_id": best_train.get("episode_id"),
                    "nearest_train_task_type": best_train.get("task_type"),
                    "semantic_jaccard": best_score,
                    "target_text_match": test_row.get("target_text")
                    == best_train.get("target_text"),
                    "action_tuple_match": _action_tuple(test_row)
                    == best_train.get("action_tuple"),
                    "test_token_count": len(test_tokens),
                    "train_token_count": len(best_train["tokens"]),
                }
            )
    top_matches.sort(key=lambda item: item["semantic_jaccard"], reverse=True)
    return {
        "semantic_nearest_neighbor": _stats(similarities),
        "per_task_semantic_nearest_neighbor": {
            task: _stats(values) for task, values in sorted(per_task.items())
        },
        "top_semantic_neighbors": top_matches[:top_k],
    }


def _target_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(str(row.get("action_type") or "") for row in rows)
    tuple_counts = Counter(_action_tuple(row) for row in rows)
    command_counts = Counter(
        str(row.get("command"))
        for row in rows
        if row.get("command") is not None
    )
    file_counts = Counter(
        str(row.get("file_target"))
        for row in rows
        if row.get("file_target") is not None
    )
    return {
        "row_count": len(rows),
        "action_counts": dict(sorted(action_counts.items())),
        "unique_action_tuple_count": len(tuple_counts),
        "top_action_tuples": [
            {"action": key[0], "command": key[1], "file_target": key[2], "count": count}
            for key, count in tuple_counts.most_common(20)
        ],
        "unique_command_count": len(command_counts),
        "top_commands": [
            {"command": key, "count": count} for key, count in command_counts.most_common(20)
        ],
        "unique_file_count": len(file_counts),
        "top_files": [
            {"file_target": key, "count": count}
            for key, count in file_counts.most_common(20)
        ],
    }


def _loss_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    history = payload.get("history") or []
    final = history[-1] if history else {}
    first_train_loss = final.get("first_train_loss")
    train_loss = final.get("train_loss")
    val_loss = final.get("val_loss")
    val_train_ratio = (
        float(val_loss) / max(float(train_loss), 1e-12)
        if val_loss is not None and train_loss is not None
        else None
    )
    loss_drop = (
        (float(first_train_loss) - float(train_loss)) / max(float(first_train_loss), 1e-12)
        if first_train_loss is not None and train_loss is not None
        else None
    )
    return {
        "path": str(Path(path)),
        "adapter_name": payload.get("adapter_name"),
        "train_examples": payload.get("train_examples"),
        "val_examples": payload.get("val_examples"),
        "first_train_loss": first_train_loss,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "val_train_ratio": val_train_ratio,
        "loss_drop_rate": loss_drop,
    }


def _loss_warnings(
    summaries: list[dict[str, Any]],
    *,
    max_val_train_ratio: float,
    min_loss_drop_rate: float,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for summary in summaries:
        ratio = summary.get("val_train_ratio")
        if ratio is not None and ratio > max_val_train_ratio:
            warnings.append(
                {
                    "type": "classic_train_val_overfit",
                    "adapter_name": summary.get("adapter_name"),
                    "val_train_ratio": ratio,
                    "threshold": max_val_train_ratio,
                }
            )
        drop = summary.get("loss_drop_rate")
        if drop is not None and drop < min_loss_drop_rate:
            warnings.append(
                {
                    "type": "weak_fit_or_undertraining",
                    "adapter_name": summary.get("adapter_name"),
                    "loss_drop_rate": drop,
                    "threshold": min_loss_drop_rate,
                }
            )
    return warnings


def analyze_phase2b_overfit(
    *,
    train_sft_jsonl: str | Path,
    val_sft_jsonl: str | Path,
    test_jsonl: str | Path,
    synapse_checkpoint: str | Path,
    dataset_dir: str | Path | None = None,
    prompt_style: str = "nsi_state_v2",
    synapse_device: str = "cpu",
    train_summary_paths: list[str | Path] | None = None,
    max_semantic_neighbor_p95: float = 0.92,
    warn_action_tuple_overlap: float = 0.75,
    warn_command_slot_overlap: float = 0.50,
    max_val_train_ratio: float = 2.0,
    min_loss_drop_rate: float = 0.20,
) -> dict[str, Any]:
    metadata_by_episode = _metadata_by_episode(Path(dataset_dir) if dataset_dir else None)
    train_rows = _read_jsonl_dicts(train_sft_jsonl)
    val_rows = _read_jsonl_dicts(val_sft_jsonl)
    test_rows = _build_test_sft_rows(
        test_jsonl=test_jsonl,
        prompt_style=prompt_style,
        synapse_checkpoint=synapse_checkpoint,
        synapse_device=synapse_device,
    )
    generalization = analyze_phase2b_generalization(
        train_sft_jsonl=train_sft_jsonl,
        val_sft_jsonl=val_sft_jsonl,
        test_jsonl=test_jsonl,
        synapse_checkpoint=synapse_checkpoint,
        dataset_dir=dataset_dir,
        prompt_style=prompt_style,
        synapse_device=synapse_device,
    )
    nearest = _nearest_neighbor_report(train_rows=train_rows, test_rows=test_rows)
    loss_summaries = [
        _loss_summary(path) for path in (train_summary_paths or []) if Path(path).exists()
    ]
    loss_warnings = _loss_warnings(
        loss_summaries,
        max_val_train_ratio=max_val_train_ratio,
        min_loss_drop_rate=min_loss_drop_rate,
    )
    p95 = nearest["semantic_nearest_neighbor"]["p95"]
    overlap = generalization["overlap_with_train"]
    warnings: list[dict[str, Any]] = []
    if p95 is not None and p95 > max_semantic_neighbor_p95:
        warnings.append(
            {
                "type": "high_semantic_nearest_neighbor_similarity",
                "semantic_jaccard_p95": p95,
                "threshold": max_semantic_neighbor_p95,
            }
        )
    if overlap["test_action_tuple_overlap_rate"] >= warn_action_tuple_overlap:
        warnings.append(
            {
                "type": "high_action_tuple_reuse",
                "overlap_rate": overlap["test_action_tuple_overlap_rate"],
                "threshold": warn_action_tuple_overlap,
            }
        )
    if overlap["test_command_slot_overlap_rate"] >= warn_command_slot_overlap:
        warnings.append(
            {
                "type": "high_command_slot_reuse",
                "overlap_rate": overlap["test_command_slot_overlap_rate"],
                "threshold": warn_command_slot_overlap,
            }
        )
    warnings.extend(loss_warnings)
    exact_memorization_clear = bool(generalization["passed"])
    classic_overfit_clear = not any(
        warning["type"] == "classic_train_val_overfit" for warning in loss_warnings
    )
    semantic_similarity_clear = not any(
        warning["type"] == "high_semantic_nearest_neighbor_similarity"
        for warning in warnings
    )
    return {
        "passed": exact_memorization_clear and classic_overfit_clear and semantic_similarity_clear,
        "exact_memorization_clear": exact_memorization_clear,
        "classic_train_val_overfit_clear": classic_overfit_clear,
        "semantic_similarity_clear": semantic_similarity_clear,
        "thresholds": {
            "max_semantic_neighbor_p95": max_semantic_neighbor_p95,
            "warn_action_tuple_overlap": warn_action_tuple_overlap,
            "warn_command_slot_overlap": warn_command_slot_overlap,
            "max_val_train_ratio": max_val_train_ratio,
            "min_loss_drop_rate": min_loss_drop_rate,
        },
        "split_counts": generalization["split_counts"],
        "overlap_with_train": overlap,
        "semantic_similarity": nearest,
        "target_distribution": {
            "train": _target_distribution(train_rows),
            "val": _target_distribution(val_rows),
            "test": _target_distribution(test_rows),
        },
        "train_loss_summaries": loss_summaries,
        "warnings": warnings,
        "hidden_leakage": generalization["hidden_leakage"],
        "scenario_holdout": {
            "test_scenario_overlap_with_train": overlap["test_scenario_overlap_rate"],
            "per_task_overlap_with_train": generalization["per_task_overlap_with_train"],
        },
        "interpretation": (
            "This audit prevents exact split leakage and flags memorization-like pattern "
            "risk. High action or command-slot reuse is expected in bounded motor-schema "
            "tasks, so it is reported separately from exact memorization. Claims should "
            "not be upgraded unless held-out evaluation also passes the Phase 2B gates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase 2B QLoRA runs for overfit and memorization risk."
    )
    parser.add_argument("--train-sft-jsonl", required=True)
    parser.add_argument("--val-sft-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--synapse-checkpoint", required=True)
    parser.add_argument("--dataset-dir")
    parser.add_argument("--prompt-style", default="nsi_state_v2")
    parser.add_argument("--synapse-device", default="cpu")
    parser.add_argument("--train-summary", action="append", default=[])
    parser.add_argument("--max-semantic-neighbor-p95", type=float, default=0.92)
    parser.add_argument("--warn-action-tuple-overlap", type=float, default=0.75)
    parser.add_argument("--warn-command-slot-overlap", type=float, default=0.50)
    parser.add_argument("--max-val-train-ratio", type=float, default=2.0)
    parser.add_argument("--min-loss-drop-rate", type=float, default=0.20)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    payload = analyze_phase2b_overfit(
        train_sft_jsonl=args.train_sft_jsonl,
        val_sft_jsonl=args.val_sft_jsonl,
        test_jsonl=args.test_jsonl,
        synapse_checkpoint=args.synapse_checkpoint,
        dataset_dir=args.dataset_dir,
        prompt_style=args.prompt_style,
        synapse_device=args.synapse_device,
        train_summary_paths=args.train_summary,
        max_semantic_neighbor_p95=args.max_semantic_neighbor_p95,
        warn_action_tuple_overlap=args.warn_action_tuple_overlap,
        warn_command_slot_overlap=args.warn_command_slot_overlap,
        max_val_train_ratio=args.max_val_train_ratio,
        min_loss_drop_rate=args.min_loss_drop_rate,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not args.no_fail and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
