from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2i_data_health import (
    _challenge_split_summary,
    _head_split_summary,
    _overlap,
    _parse_named_path,
    _read_head_rows,
    _semantic_nn,
)
from reflexlm.llm.candidate_features import source_overlap_command_slot_prediction
from reflexlm.llm.native_head_training import _balanced_limited_rows, _balance_debug_command_intent_rows
from reflexlm.llm.receptor_latent import DEBUG_ACTION_STAGE_ORDER


DEFAULT_OUTPUT = Path("artifacts/reports/phase2j_semantic_command_identity/phase2j_data_health_audit.json")
DEFAULT_HEAD_SPLITS = {
    "phase2j_head_train": Path("artifacts/datasets/phase2j_semantic_command_identity_head/train.jsonl"),
    "phase2j_head_val": Path("artifacts/datasets/phase2j_semantic_command_identity_head/val.jsonl"),
}
DEFAULT_CHALLENGE_SPLITS = {
    "phase2j_semantic_train": Path("artifacts/datasets/phase2j_semantic_train/challenge.jsonl"),
    "phase2j_semantic_val": Path("artifacts/datasets/phase2j_semantic_val/challenge.jsonl"),
}


def _challenge_state_texts(path: Path) -> list[str]:
    summary = _challenge_split_summary(path)
    return list(summary.get("_state_texts", []))


def _effective_head_rows(
    path: Path,
    *,
    limit: int | None = None,
    balance_debug_command_intents: bool = False,
) -> list[dict[str, Any]]:
    rows = _read_head_rows(path)
    if balance_debug_command_intents:
        rows = _balance_debug_command_intent_rows(rows)
    if limit is not None and limit > 0:
        rows = _balanced_limited_rows(rows, limit)
    return rows


def _balance_training_split_only(split_name: str, enabled: bool) -> bool:
    """Match native-head training: intent balancing is a train-only sampler."""
    return enabled and "train" in split_name


def _phase2j_source_overlap_hard_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    source_overlap_correct = 0
    identity_correct = 0
    identity_confident = 0
    prompt_sidecar_leak_rows = 0
    examples: list[dict[str, Any]] = []
    for row in rows:
        try:
            command_slot = int(row.get("command_slot", -100))
        except (TypeError, ValueError):
            command_slot = -100
        if command_slot == -100:
            continue
        total += 1
        prompt = str(row.get("state_prompt") or "")
        if "phase2j_command_identity_tokens=" in prompt.lower():
            prompt_sidecar_leak_rows += 1
        candidates = list(row.get("candidate_commands") or [])
        source_prediction = source_overlap_command_slot_prediction(prompt, candidates)
        source_overlap_correct += int(source_prediction == command_slot)
        nsi_reference = row.get("nsi_reference") or {}
        identity_scores = []
        for index in range(4):
            try:
                score = float(nsi_reference.get(f"command_identity_slot:{index}") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            identity_scores.append(score)
        best = max(identity_scores) if identity_scores else 0.0
        identity_prediction = max(range(len(identity_scores)), key=lambda index: identity_scores[index])
        unique_identity = best > 0.0 and identity_scores.count(best) == 1
        identity_confident += int(unique_identity)
        identity_correct += int(unique_identity and identity_prediction == command_slot)
        if len(examples) < 8:
            examples.append(
                {
                    "example_id": row.get("example_id"),
                    "command_slot": command_slot,
                    "source_overlap_prediction": source_prediction,
                    "identity_prediction": identity_prediction if unique_identity else None,
                    "identity_best_score": round(best, 6),
                }
            )
    return {
        "total": total,
        "source_overlap_correct": source_overlap_correct,
        "source_overlap_accuracy": source_overlap_correct / total if total else 0.0,
        "identity_confident_rows": identity_confident,
        "identity_correct": identity_correct,
        "identity_accuracy": identity_correct / total if total else 0.0,
        "prompt_sidecar_leak_rows": prompt_sidecar_leak_rows,
        "examples": examples,
    }


def _synapse_reference_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_fields = [
        "reflex_action",
        "salience",
        "risk",
        "prediction_error",
        "confidence",
    ]
    rows_with_reference = 0
    rows_with_all_required = 0
    reflex_actions: dict[str, int] = {}
    missing_examples: list[dict[str, Any]] = []
    for row in rows:
        reference = row.get("nsi_reference") or {}
        if reference:
            rows_with_reference += 1
        missing = [field for field in required_fields if field not in reference]
        if not missing:
            rows_with_all_required += 1
            action = str(reference.get("reflex_action") or "")
            reflex_actions[action] = reflex_actions.get(action, 0) + 1
            continue
        if len(missing_examples) < 8:
            missing_examples.append(
                {
                    "example_id": row.get("example_id"),
                    "missing_fields": missing,
                }
            )
    total = len(rows)
    return {
        "total": total,
        "rows_with_reference": rows_with_reference,
        "rows_with_all_required": rows_with_all_required,
        "coverage": rows_with_all_required / total if total else 0.0,
        "required_fields": required_fields,
        "reflex_actions": dict(sorted(reflex_actions.items())),
        "missing_examples": missing_examples,
    }


def _debug_action_stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts: dict[str, int] = {}
    action_by_stage: dict[str, dict[str, int]] = {}
    rows_with_stage = 0
    missing_examples: list[dict[str, Any]] = []
    for row in rows:
        reference = row.get("nsi_reference") or {}
        stage = str(reference.get("debug_action_stage") or "")
        if stage:
            rows_with_stage += 1
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            action = str(row.get("action_type") or "")
            bucket = action_by_stage.setdefault(stage, {})
            bucket[action] = bucket.get(action, 0) + 1
            continue
        if len(missing_examples) < 8:
            missing_examples.append({"example_id": row.get("example_id")})
    total = len(rows)
    required_stages = [
        stage for stage in DEBUG_ACTION_STAGE_ORDER if stage != "other"
    ]
    present_stages = set(stage_counts)
    return {
        "total": total,
        "rows_with_stage": rows_with_stage,
        "coverage": rows_with_stage / total if total else 0.0,
        "required_stages": required_stages,
        "stage_counts": dict(sorted(stage_counts.items())),
        "action_by_stage": {
            stage: dict(sorted(actions.items()))
            for stage, actions in sorted(action_by_stage.items())
        },
        "required_stages_present": all(stage in present_stages for stage in required_stages),
        "missing_examples": missing_examples,
    }


def build_phase2j_data_health_audit(
    *,
    head_splits: dict[str, Path],
    challenge_splits: dict[str, Path],
    reference_splits: dict[str, Path] | None = None,
    head_limits: dict[str, int] | None = None,
    balance_debug_command_intents: bool = False,
    max_command_slot_share: float = 0.45,
    min_val_target_commands: int = 6,
    max_train_val_target_overlap: float = 0.25,
    max_semantic_nn: float = 0.80,
    source_overlap_hard: bool = False,
    max_source_overlap_val_accuracy: float = 0.80,
    min_identity_signal_accuracy: float = 1.0,
    min_command_candidate_count_bands: int = 1,
    required_command_candidate_counts: list[int] | None = None,
    require_synapse_reference: bool = False,
    require_debug_action_stage: bool = False,
) -> dict[str, Any]:
    effective_head_limits = head_limits or {}
    head = {
        name: _head_split_summary(
            path,
            limit=effective_head_limits.get(name),
            balance_debug_command_intents=_balance_training_split_only(
                name,
                balance_debug_command_intents,
            ),
        )
        for name, path in head_splits.items()
    }
    challenge = {name: _challenge_split_summary(path) for name, path in challenge_splits.items()}
    references = {
        name: _challenge_split_summary(path)
        for name, path in (reference_splits or {}).items()
    }

    overlaps: dict[str, Any] = {}
    for left_name, left in {**head, **challenge}.items():
        left_targets = set(left.get("target_commands", []))
        left_candidates = set(left.get("candidate_commands", []))
        for right_name, right in {**head, **challenge, **references}.items():
            if left_name == right_name:
                continue
            overlaps[f"{left_name}__vs__{right_name}"] = {
                "target_command_overlap": _overlap(left_targets, set(right.get("target_commands", []))),
                "candidate_command_overlap": _overlap(
                    left_candidates,
                    set(right.get("candidate_commands", [])),
                ),
            }

    semantic_nn: dict[str, Any] = {}
    challenge_texts = {name: _challenge_state_texts(path) for name, path in challenge_splits.items()}
    challenge_names = list(challenge_texts)
    for index, left_name in enumerate(challenge_names):
        for right_name in challenge_names[index + 1 :]:
            semantic_nn[f"{left_name}__vs__{right_name}"] = _semantic_nn(
                challenge_texts[left_name],
                challenge_texts[right_name],
            )
    for summary in challenge.values():
        summary.pop("_state_texts", None)
    for summary in references.values():
        summary.pop("_state_texts", None)

    head_checks: dict[str, bool] = {}
    required_candidate_counts = {
        str(value) for value in (required_command_candidate_counts or []) if value > 0
    }
    for name, summary in head.items():
        head_checks[f"{name}_basic_health"] = bool(summary["passed"])
        candidate_count_bands = {
            str(count)
            for count, rows in (summary.get("command_candidate_counts") or {}).items()
            if int(rows) > 0
        }
        head_checks[f"{name}_command_candidate_count_band_coverage"] = (
            len(candidate_count_bands) >= min_command_candidate_count_bands
        )
        if required_candidate_counts:
            head_checks[f"{name}_required_command_candidate_counts_present"] = (
                required_candidate_counts <= candidate_count_bands
            )
        if summary["run_rows"]:
            head_checks[f"{name}_command_slot_max_share"] = (
                float(summary["command_slot_max_share"]) <= max_command_slot_share
            )
        if "val" in name:
            head_checks[f"{name}_target_command_coverage"] = (
                int(summary["unique_target_command_count"]) >= min_val_target_commands
            )

    challenge_checks = {
        f"{name}_basic_health": bool(summary["passed"])
        for name, summary in challenge.items()
    }
    train_val_overlap_ok = True
    train_val_intent_coverage_ok = True
    train_val_intent_gap: dict[str, Any] = {
        "missing_train_command_intents": [],
        "train_command_intents": [],
        "val_command_intents": [],
    }
    if "phase2j_head_train" in head and "phase2j_head_val" in head:
        overlap = _overlap(
            set(head["phase2j_head_train"]["target_commands"]),
            set(head["phase2j_head_val"]["target_commands"]),
        )
        train_val_overlap_ok = float(overlap["rate_vs_right"]) <= max_train_val_target_overlap
        overlaps["phase2j_head_train__vs__phase2j_head_val"]["target_command_overlap"] = overlap
        train_intents = set(head["phase2j_head_train"].get("command_intents", {}))
        val_intents = set(head["phase2j_head_val"].get("command_intents", {}))
        missing_train_intents = sorted(val_intents - train_intents)
        train_val_intent_gap = {
            "missing_train_command_intents": missing_train_intents,
            "train_command_intents": sorted(train_intents),
            "val_command_intents": sorted(val_intents),
        }
        train_val_intent_coverage_ok = not missing_train_intents

    reference_overlap_ok = True
    for reference_name, reference in references.items():
        reference_targets = set(reference.get("target_commands", []))
        for source_name, source in {**head, **challenge}.items():
            if not source_name.startswith("phase2j"):
                continue
            overlap = _overlap(reference_targets, set(source.get("target_commands", [])))
            overlaps[f"{reference_name}__phase2j_reference_guard__{source_name}"] = {
                "target_command_overlap": overlap,
            }
            if overlap["count"]:
                reference_overlap_ok = False

    semantic_nn_ok = all(float(best["score"]) < max_semantic_nn for best in semantic_nn.values())
    checks = {
        **head_checks,
        **challenge_checks,
        "phase2j_effective_split_hashes_present": all(
            bool(summary.get("effective_split_sha256")) for summary in head.values()
        ),
        "phase2j_train_val_target_overlap": train_val_overlap_ok,
        "phase2j_train_val_command_intent_coverage": train_val_intent_coverage_ok,
        "reference_splits_have_no_phase2j_target_overlap": reference_overlap_ok,
        "semantic_nearest_neighbor_below_threshold": semantic_nn_ok,
    }
    source_overlap_hard_summary: dict[str, Any] = {}
    if source_overlap_hard:
        source_overlap_hard_summary = {
            name: _phase2j_source_overlap_hard_summary(
                _effective_head_rows(
                    path,
                    limit=effective_head_limits.get(name),
                    balance_debug_command_intents=_balance_training_split_only(
                        name,
                        balance_debug_command_intents,
                    ),
                )
            )
            for name, path in head_splits.items()
            if "val" in name
        }
        val_summaries = list(source_overlap_hard_summary.values())
        val_total = sum(int(summary["total"]) for summary in val_summaries)
        max_val_source_overlap = max(
            (float(summary["source_overlap_accuracy"]) for summary in val_summaries),
            default=0.0,
        )
        min_val_identity = min(
            (float(summary["identity_accuracy"]) for summary in val_summaries),
            default=0.0,
        )
        prompt_leak_rows = sum(int(summary["prompt_sidecar_leak_rows"]) for summary in val_summaries)
        checks.update(
            {
                "phase2j_source_overlap_hard_val_rows_present": val_total > 0,
                "phase2j_source_overlap_hard_val_baseline_below_threshold": (
                    val_total > 0 and max_val_source_overlap <= max_source_overlap_val_accuracy
                ),
                "phase2j_source_overlap_hard_identity_signal_present": (
                    val_total > 0 and min_val_identity >= min_identity_signal_accuracy
                ),
                "phase2j_source_overlap_hard_prompt_redacts_identity_sidecar": prompt_leak_rows == 0,
            }
        )
    synapse_reference_summary: dict[str, Any] = {}
    if require_synapse_reference:
        synapse_reference_summary = {
            name: _synapse_reference_summary(
                _effective_head_rows(
                    path,
                    limit=effective_head_limits.get(name),
                    balance_debug_command_intents=_balance_training_split_only(
                        name,
                        balance_debug_command_intents,
                    ),
                )
            )
            for name, path in head_splits.items()
        }
        checks.update(
            {
                f"{name}_synapse_reference_present": (
                    int(summary["total"]) > 0
                    and int(summary["rows_with_all_required"]) == int(summary["total"])
                )
                for name, summary in synapse_reference_summary.items()
            }
        )
    debug_action_stage_summary: dict[str, Any] = {}
    if require_debug_action_stage:
        debug_action_stage_summary = {
            name: _debug_action_stage_summary(
                _effective_head_rows(
                    path,
                    limit=effective_head_limits.get(name),
                    balance_debug_command_intents=_balance_training_split_only(
                        name,
                        balance_debug_command_intents,
                    ),
                )
            )
            for name, path in head_splits.items()
        }
        checks.update(
            {
                f"{name}_debug_action_stage_present": (
                    int(summary["total"]) > 0
                    and int(summary["rows_with_stage"]) == int(summary["total"])
                )
                for name, summary in debug_action_stage_summary.items()
            }
        )
        checks.update(
            {
                f"{name}_debug_action_stage_coverage": bool(
                    summary["required_stages_present"]
                )
                for name, summary in debug_action_stage_summary.items()
            }
        )
    return {
        "audit_family": "phase2j_data_health",
        "sealed_usage": {
            "sealed_splits_used_for_training": False,
            "sealed_splits_used_for_tuning": False,
            "reference_splits_are_overlap_guards_only": bool(references),
        },
        "thresholds": {
            "max_command_slot_share": max_command_slot_share,
            "min_val_target_commands": min_val_target_commands,
            "max_train_val_target_overlap": max_train_val_target_overlap,
            "max_semantic_nn": max_semantic_nn,
            "source_overlap_hard": source_overlap_hard,
            "max_source_overlap_val_accuracy": max_source_overlap_val_accuracy,
            "min_identity_signal_accuracy": min_identity_signal_accuracy,
            "min_command_candidate_count_bands": min_command_candidate_count_bands,
            "required_command_candidate_counts": sorted(required_candidate_counts),
            "require_synapse_reference": require_synapse_reference,
            "require_debug_action_stage": require_debug_action_stage,
        },
        "passed": all(checks.values()),
        "checks": checks,
        "effective_split_hashes": {
            name: summary["effective_split_sha256"] for name, summary in head.items()
        },
        "head_splits": head,
        "challenge_splits": challenge,
        "reference_splits": references,
        "train_val_command_intent_gap": train_val_intent_gap,
        "overlaps": overlaps,
        "semantic_nearest_neighbors": semantic_nn,
        "source_overlap_hard": source_overlap_hard_summary,
        "synapse_reference": synapse_reference_summary,
        "debug_action_stage": debug_action_stage_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2J non-sealed data health and overlap.")
    parser.add_argument("--head-split", action="append", default=[])
    parser.add_argument("--challenge-split", action="append", default=[])
    parser.add_argument("--reference-split", action="append", default=[])
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-command-slot-share", type=float, default=0.45)
    parser.add_argument("--min-val-target-commands", type=int, default=6)
    parser.add_argument("--max-train-val-target-overlap", type=float, default=0.25)
    parser.add_argument("--max-semantic-nn", type=float, default=0.80)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=0)
    parser.add_argument("--balance-debug-command-intents", action="store_true")
    parser.add_argument("--source-overlap-hard", action="store_true")
    parser.add_argument("--max-source-overlap-val-accuracy", type=float, default=0.80)
    parser.add_argument("--min-identity-signal-accuracy", type=float, default=1.0)
    parser.add_argument("--min-command-candidate-count-bands", type=int, default=1)
    parser.add_argument(
        "--required-command-candidate-count",
        action="append",
        default=[],
        type=int,
        help="Candidate count that must appear in every Phase2J head split; may be repeated.",
    )
    parser.add_argument("--require-synapse-reference", action="store_true")
    parser.add_argument("--require-debug-action-stage", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    head_splits = (
        dict(_parse_named_path(value) for value in args.head_split)
        if args.head_split
        else DEFAULT_HEAD_SPLITS
    )
    challenge_splits = (
        dict(_parse_named_path(value) for value in args.challenge_split)
        if args.challenge_split
        else DEFAULT_CHALLENGE_SPLITS
    )
    reference_splits = dict(_parse_named_path(value) for value in args.reference_split)
    head_limits: dict[str, int] = {}
    if args.max_train_records > 0:
        for name in head_splits:
            if "train" in name:
                head_limits[name] = args.max_train_records
    if args.max_val_records > 0:
        for name in head_splits:
            if "val" in name:
                head_limits[name] = args.max_val_records
    report = build_phase2j_data_health_audit(
        head_splits=head_splits,
        challenge_splits=challenge_splits,
        reference_splits=reference_splits,
        head_limits=head_limits,
        balance_debug_command_intents=args.balance_debug_command_intents,
        max_command_slot_share=args.max_command_slot_share,
        min_val_target_commands=args.min_val_target_commands,
        max_train_val_target_overlap=args.max_train_val_target_overlap,
        max_semantic_nn=args.max_semantic_nn,
        source_overlap_hard=args.source_overlap_hard,
        max_source_overlap_val_accuracy=args.max_source_overlap_val_accuracy,
        min_identity_signal_accuracy=args.min_identity_signal_accuracy,
        min_command_candidate_count_bands=args.min_command_candidate_count_bands,
        required_command_candidate_counts=args.required_command_candidate_count,
        require_synapse_reference=args.require_synapse_reference,
        require_debug_action_stage=args.require_debug_action_stage,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
