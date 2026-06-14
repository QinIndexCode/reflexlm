from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.data.jsonl import read_jsonl
from reflexlm.llm.candidate_features import (
    command_candidate_source_overlap_rows,
    command_intent_for_text,
    source_overlap_command_slot_prediction,
)
from reflexlm.llm.head_dataset import build_phase2c_head_state_prompt_from_state
from reflexlm.llm.native_head_training import NSI_LATENT_FIELDS
from reflexlm.llm.receptor_latent import COMMAND_IDENTITY_LATENT_FIELDS
from reflexlm.models.features import candidate_commands


DEFAULT_OUTPUT = Path("artifacts/reports/phase2i_latent_necessity_audit.json")
DEFAULT_HEAD_SPLITS = {
    "phase2i_head_train": Path(
        "artifacts/datasets/phase2i_semantic_pairwise_head_intentbalanced_canary/train.jsonl"
    ),
    "phase2i_head_val": Path(
        "artifacts/datasets/phase2i_semantic_pairwise_head_intentbalanced_canary/val.jsonl"
    ),
}
DEFAULT_CHALLENGE_SPLITS = {
    "phase2f_latent_sensitive": Path("artifacts/datasets/phase2f_latent_sensitive/challenge.jsonl"),
}


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name, Path(path)


def _read_head_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / max(float(denominator), 1.0), 6)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _last_command_from_prompt(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("last_command="):
            return line.split("=", 1)[1].strip()
    return ""


def _is_latent_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return "failure_signal=latent_required" in lowered or "low-level receptor latent" in lowered


def _is_compressed_prompt(prompt: str) -> bool:
    return "<compressed_failure_signal>" in prompt


def _is_sealed_path(name: str, path: Path) -> bool:
    text = f"{name} {path}".lower()
    return "external_trace" in text or ".sealed" in text or "sealed_" in text


def _target_intent_is_ambiguous(candidates: list[str], slot: int) -> bool:
    if slot < 0 or slot >= len(candidates):
        return False
    target_intent = command_intent_for_text(candidates[slot])
    return Counter(command_intent_for_text(command) for command in candidates)[target_intent] >= 2


def _empty_metrics(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "command_rows": 0,
        "latent_command_rows": 0,
        "compressed_latent_command_rows": 0,
        "same_intent_ambiguous_latent_rows": 0,
        "non_last_latent_rows": 0,
        "source_overlap_correct_rows": 0,
        "latent_source_overlap_correct_rows": 0,
        "source_overlap_accuracy": 0.0,
        "latent_source_overlap_accuracy": 0.0,
        "compressed_latent_rate": 0.0,
        "latent_same_intent_ambiguous_rate": 0.0,
        "latent_non_last_rate": 0.0,
        "latent_target_source_overlap_mean": 0.0,
        "latent_max_source_overlap_mean": 0.0,
        "command_intents": {},
        "examples": [],
    }


def _summarize_command_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _empty_metrics(path)
    command_intents: Counter[str] = Counter()
    latent_target_scores: list[float] = []
    latent_max_scores: list[float] = []
    examples: list[dict[str, Any]] = []

    for item in rows:
        candidates = list(item["candidate_commands"])
        slot = int(item["command_slot"])
        prompt = str(item["state_prompt"])
        if slot < 0 or slot >= len(candidates):
            continue
        target_command = candidates[slot]
        command_intent = command_intent_for_text(target_command)
        command_intents[command_intent] += 1
        prediction = source_overlap_command_slot_prediction(prompt, candidates)
        overlap_rows = command_candidate_source_overlap_rows(prompt, candidates)
        target_score = float(overlap_rows[slot][1]) if slot < len(overlap_rows) else 0.0
        max_score = max((float(row[1]) for row in overlap_rows), default=0.0)
        latent = _is_latent_prompt(prompt)
        compressed = _is_compressed_prompt(prompt)
        same_intent = _target_intent_is_ambiguous(candidates, slot)
        non_last = target_command != str(item.get("last_command") or "")

        metrics["command_rows"] += 1
        metrics["source_overlap_correct_rows"] += int(prediction == slot)
        if latent:
            metrics["latent_command_rows"] += 1
            metrics["compressed_latent_command_rows"] += int(compressed)
            metrics["same_intent_ambiguous_latent_rows"] += int(same_intent)
            metrics["non_last_latent_rows"] += int(non_last)
            metrics["latent_source_overlap_correct_rows"] += int(prediction == slot)
            latent_target_scores.append(target_score)
            latent_max_scores.append(max_score)
            if len(examples) < 5:
                examples.append(
                    {
                        "example_id": item.get("example_id"),
                        "target_slot": slot,
                        "source_overlap_prediction": prediction,
                        "target_intent": command_intent,
                        "target_score": round(target_score, 6),
                        "max_score": round(max_score, 6),
                        "compressed": compressed,
                        "same_intent_ambiguous": same_intent,
                        "non_last": non_last,
                    }
                )

    metrics["source_overlap_accuracy"] = _rate(
        metrics["source_overlap_correct_rows"],
        metrics["command_rows"],
    )
    metrics["latent_source_overlap_accuracy"] = _rate(
        metrics["latent_source_overlap_correct_rows"],
        metrics["latent_command_rows"],
    )
    metrics["compressed_latent_rate"] = _rate(
        metrics["compressed_latent_command_rows"],
        metrics["latent_command_rows"],
    )
    metrics["latent_same_intent_ambiguous_rate"] = _rate(
        metrics["same_intent_ambiguous_latent_rows"],
        metrics["latent_command_rows"],
    )
    metrics["latent_non_last_rate"] = _rate(
        metrics["non_last_latent_rows"],
        metrics["latent_command_rows"],
    )
    metrics["latent_target_source_overlap_mean"] = _mean(latent_target_scores)
    metrics["latent_max_source_overlap_mean"] = _mean(latent_max_scores)
    metrics["command_intents"] = dict(sorted(command_intents.items()))
    metrics["examples"] = examples
    return metrics


def _head_split_summary(path: Path) -> dict[str, Any]:
    command_rows: list[dict[str, Any]] = []
    for row in _read_head_rows(path):
        try:
            slot = int(row.get("command_slot", -100))
        except (TypeError, ValueError):
            continue
        candidates = list(row.get("candidate_commands") or [])
        if slot < 0 or slot >= len(candidates):
            continue
        prompt = str(row.get("state_prompt") or "")
        command_rows.append(
            {
                "example_id": row.get("example_id"),
                "candidate_commands": candidates,
                "command_slot": slot,
                "state_prompt": prompt,
                "last_command": _last_command_from_prompt(prompt),
            }
        )
    return _summarize_command_rows(path, command_rows)


def _challenge_split_summary(path: Path) -> dict[str, Any]:
    command_rows: list[dict[str, Any]] = []
    records = read_jsonl(path) if path.exists() else []
    for record in records:
        if not (record.action and record.action.command):
            continue
        commands = candidate_commands(record.state)
        if record.action.command not in commands:
            continue
        command_rows.append(
            {
                "example_id": f"{record.episode_id}:{record.t}",
                "candidate_commands": commands,
                "command_slot": commands.index(record.action.command),
                "state_prompt": build_phase2c_head_state_prompt_from_state(record.state),
                "last_command": record.state.terminal.last_command,
            }
        )
    return _summarize_command_rows(path, command_rows)


def _rollup(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "command_rows",
        "latent_command_rows",
        "compressed_latent_command_rows",
        "same_intent_ambiguous_latent_rows",
        "non_last_latent_rows",
        "source_overlap_correct_rows",
        "latent_source_overlap_correct_rows",
    ]
    output = {key: sum(int(summary.get(key, 0)) for summary in summaries.values()) for key in keys}
    output["source_overlap_accuracy"] = _rate(
        output["source_overlap_correct_rows"],
        output["command_rows"],
    )
    output["latent_source_overlap_accuracy"] = _rate(
        output["latent_source_overlap_correct_rows"],
        output["latent_command_rows"],
    )
    output["compressed_latent_rate"] = _rate(
        output["compressed_latent_command_rows"],
        output["latent_command_rows"],
    )
    output["latent_same_intent_ambiguous_rate"] = _rate(
        output["same_intent_ambiguous_latent_rows"],
        output["latent_command_rows"],
    )
    output["latent_non_last_rate"] = _rate(
        output["non_last_latent_rows"],
        output["latent_command_rows"],
    )
    return output


def _is_phase2j_command_identity_field(field: str) -> bool:
    return field in set(COMMAND_IDENTITY_LATENT_FIELDS) or field.lower().startswith(
        "command_identity_"
    )


def _command_identity_latent_fields(
    fields: tuple[str, ...], *, include_phase2j_command_identity: bool = False
) -> list[str]:
    identity_markers = (
        "reflex_command",
        "target_command",
        "command_hash",
        "command_id",
        "command_slot",
        "cmd_hash",
        "cmd_id",
        "slot:",
    )
    output: list[str] = []
    for field in fields:
        lowered = field.lower()
        if (
            not include_phase2j_command_identity
            and _is_phase2j_command_identity_field(field)
        ):
            continue
        # Action-class features such as ``reflex_action:RUN_COMMAND`` only say
        # that a command should be run; they do not identify which candidate.
        if lowered.startswith("reflex_action:"):
            continue
        if any(marker in lowered for marker in identity_markers):
            output.append(field)
    return output


def build_phase2i_latent_necessity_audit(
    *,
    head_splits: dict[str, Path],
    challenge_splits: dict[str, Path],
    nsi_latent_fields: tuple[str, ...] = NSI_LATENT_FIELDS,
    min_latent_command_rows: int = 16,
    max_latent_source_overlap_accuracy: float = 0.60,
    min_latent_same_intent_rate: float = 0.80,
    min_latent_non_last_rate: float = 0.50,
    require_head_coverage: bool = True,
    include_phase2j_command_identity: bool = False,
) -> dict[str, Any]:
    head = {name: _head_split_summary(path) for name, path in head_splits.items()}
    challenge = {name: _challenge_split_summary(path) for name, path in challenge_splits.items()}
    head_rollup = _rollup(head)
    challenge_rollup = _rollup(challenge)
    sealed_inputs = [
        {"name": name, "path": str(path)}
        for name, path in {**head_splits, **challenge_splits}.items()
        if _is_sealed_path(name, path)
    ]
    command_identity_fields = _command_identity_latent_fields(
        nsi_latent_fields,
        include_phase2j_command_identity=include_phase2j_command_identity,
    )
    ignored_phase2j_command_identity_fields = [
        field for field in nsi_latent_fields if _is_phase2j_command_identity_field(field)
    ]
    same_intent_latent_rows = int(
        head_rollup["same_intent_ambiguous_latent_rows"]
        + challenge_rollup["same_intent_ambiguous_latent_rows"]
    )
    command_identity_needed = same_intent_latent_rows > 0

    head_coverage_ok = (
        True
        if not require_head_coverage
        else bool(head)
        and all(
            int(summary["latent_command_rows"]) >= min_latent_command_rows
            for summary in head.values()
        )
    )
    checks = {
        "non_sealed_inputs_only": not sealed_inputs,
        "latent_challenge_present": int(challenge_rollup["latent_command_rows"])
        >= min_latent_command_rows,
        "latent_challenge_visible_text_compressed": float(
            challenge_rollup["compressed_latent_rate"]
        )
        >= 1.0,
        "latent_challenge_same_intent_ambiguous": float(
            challenge_rollup["latent_same_intent_ambiguous_rate"]
        )
        >= min_latent_same_intent_rate,
        "latent_challenge_non_last_command": float(challenge_rollup["latent_non_last_rate"])
        >= min_latent_non_last_rate,
        "latent_challenge_source_overlap_not_sufficient": float(
            challenge_rollup["latent_source_overlap_accuracy"]
        )
        <= max_latent_source_overlap_accuracy,
        "head_latent_coverage": head_coverage_ok,
        "head_latent_visible_text_compressed": (
            True
            if not require_head_coverage
            else bool(head)
            and all(float(summary["compressed_latent_rate"]) >= 1.0 for summary in head.values())
        ),
        "head_latent_source_overlap_not_sufficient": (
            True
            if not require_head_coverage
            else bool(head)
            and all(
                float(summary["latent_source_overlap_accuracy"])
                <= max_latent_source_overlap_accuracy
                for summary in head.values()
            )
        ),
        "nsi_latent_command_identity_available": (
            True if not command_identity_needed else bool(command_identity_fields)
        ),
    }
    passed = all(checks.values())
    return {
        "audit_family": "phase2i_latent_necessity",
        "passed": passed,
        "checks": checks,
        "thresholds": {
            "min_latent_command_rows": min_latent_command_rows,
            "max_latent_source_overlap_accuracy": max_latent_source_overlap_accuracy,
            "min_latent_same_intent_rate": min_latent_same_intent_rate,
            "min_latent_non_last_rate": min_latent_non_last_rate,
            "require_head_coverage": require_head_coverage,
            "include_phase2j_command_identity": include_phase2j_command_identity,
        },
        "interpretation": {
            "can_use_current_nonsealed_data_to_claim_nsi_latent_necessity": passed,
            "reason": (
                "nonsealed latent rows are present, compressed, ambiguous, and not solved by source-overlap"
                if passed
                else "current nonsealed evidence does not isolate an NSI latent mechanism"
            ),
            "architecture_blocker": (
                None
                if checks["nsi_latent_command_identity_available"]
                else (
                    "same-intent command-slot ambiguity needs target-command identity in the "
                    "NSI latent path; current NSI_LATENT_FIELDS do not include command or slot identity"
                )
            ),
        },
        "nsi_latent_fields": list(nsi_latent_fields),
        "command_identity_latent_fields": command_identity_fields,
        "phase2j_command_identity_latent_fields_ignored_for_phase2i_claim": (
            []
            if include_phase2j_command_identity
            else ignored_phase2j_command_identity_fields
        ),
        "head_rollup": head_rollup,
        "challenge_rollup": challenge_rollup,
        "head_splits": head,
        "challenge_splits": challenge,
        "sealed_inputs": sealed_inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether non-sealed Phase2I data can isolate NSI latent necessity."
    )
    parser.add_argument("--head-split", action="append", default=[])
    parser.add_argument("--challenge-split", action="append", default=[])
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-latent-command-rows", type=int, default=16)
    parser.add_argument("--max-latent-source-overlap-accuracy", type=float, default=0.60)
    parser.add_argument("--min-latent-same-intent-rate", type=float, default=0.80)
    parser.add_argument("--min-latent-non-last-rate", type=float, default=0.50)
    parser.add_argument("--no-require-head-coverage", action="store_true")
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
    report = build_phase2i_latent_necessity_audit(
        head_splits=head_splits,
        challenge_splits=challenge_splits,
        min_latent_command_rows=args.min_latent_command_rows,
        max_latent_source_overlap_accuracy=args.max_latent_source_overlap_accuracy,
        min_latent_same_intent_rate=args.min_latent_same_intent_rate,
        min_latent_non_last_rate=args.min_latent_non_last_rate,
        require_head_coverage=not args.no_require_head_coverage,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
