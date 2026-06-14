from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _training_slot_distribution(summary: dict[str, Any], split: str) -> dict[str, int]:
    distribution = summary.get("slot_intent_distribution")
    if not isinstance(distribution, dict):
        return {}
    split_payload = distribution.get(split)
    if not isinstance(split_payload, dict):
        return {}
    slots = split_payload.get("command_slots")
    if not isinstance(slots, dict):
        return {}
    return {str(slot): int(count) for slot, count in slots.items()}


def _row_slot_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        slot = row.get("expected_patch_candidate_slot", row.get("command_slot"))
        if slot is None:
            continue
        counts[str(slot)] += 1
    return dict(sorted(counts.items()))


def audit_phase2ae_slot_support(
    *,
    training_summary_json: str | Path,
    eval_rows_jsonl: str | Path,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    summary = _read_json(training_summary_json)
    eval_rows = _read_jsonl(eval_rows_jsonl)
    train_slots = _training_slot_distribution(summary, "train")
    val_slots = _training_slot_distribution(summary, "val")
    eval_slots = _row_slot_distribution(eval_rows)
    train_support = {slot for slot, count in train_slots.items() if count > 0}
    val_support = {slot for slot, count in val_slots.items() if count > 0}
    eval_support = {slot for slot, count in eval_slots.items() if count > 0}
    missing_from_train = sorted(eval_support - train_support)
    missing_from_val = sorted(eval_support - val_support)
    checks = {
        "training_summary_has_slot_distribution": bool(train_slots) and bool(val_slots),
        "eval_rows_have_expected_slots": bool(eval_slots),
        "train_covers_eval_slots": not missing_from_train,
        "val_covers_eval_slots": not missing_from_val,
    }
    report = {
        "artifact_family": "phase2ae_slot_support_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "train_command_slot_distribution": train_slots,
            "val_command_slot_distribution": val_slots,
            "eval_expected_slot_distribution": eval_slots,
            "eval_slots_missing_from_train": missing_from_train,
            "eval_slots_missing_from_val": missing_from_val,
        },
        "interpretation": {
            "unsupported_if_failed": [
                "learned-head generalization to unseen candidate slots",
                "phase2ae initial policy selection claim",
            ],
            "claim_boundary": (
                "A package cannot support a Phase2AE learned-head selection claim when eval correct slots are absent from effective train/val slot distributions."
            ),
        },
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "eval_rows_jsonl": str(Path(eval_rows_jsonl)),
        },
    }
    if output_json is not None:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether Phase2AE eval candidate slots are covered by the effective native-head training split."
    )
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--eval-rows-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = audit_phase2ae_slot_support(
        training_summary_json=args.training_summary_json,
        eval_rows_jsonl=args.eval_rows_jsonl,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
