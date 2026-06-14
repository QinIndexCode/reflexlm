from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import source_overlap_command_slot_prediction


IDENTITY_PREFIX = "command_identity_slot:"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_count(row: dict[str, Any]) -> int:
    candidates = row.get("candidate_commands")
    return min(4, len(candidates)) if isinstance(candidates, list) else 0


def _scores(row: dict[str, Any]) -> list[float]:
    nsi = row.get("nsi_reference") if isinstance(row.get("nsi_reference"), dict) else {}
    return [float(nsi.get(f"{IDENTITY_PREFIX}{index}", 0.0) or 0.0) for index in range(4)]


def _set_scores(row: dict[str, Any], scores: list[float]) -> None:
    nsi = row.setdefault("nsi_reference", {})
    if not isinstance(nsi, dict):
        nsi = {}
        row["nsi_reference"] = nsi
    padded = list(scores[:4]) + [0.0] * max(0, 4 - len(scores))
    for index, value in enumerate(padded[:4]):
        nsi[f"{IDENTITY_PREFIX}{index}"] = float(value)
    ordered = sorted(padded[:4], reverse=True)
    confidence = ordered[0] if ordered else 0.0
    margin = confidence - (ordered[1] if len(ordered) > 1 else 0.0)
    nsi["command_identity_confidence"] = float(confidence)
    nsi["command_identity_margin"] = float(margin)


def _erased(row: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    _set_scores(out, [0.0, 0.0, 0.0, 0.0])
    out["phase2am_sidecar_control"] = "sidecar_erased"
    return out


def _wrong(row: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    count = _candidate_count(out)
    scores = _scores(out)
    if count >= 2:
        valid = scores[:count]
        shifted = [valid[-1], *valid[:-1]]
        _set_scores(out, shifted + [0.0] * (4 - count))
    else:
        _set_scores(out, [0.0, 0.0, 0.0, 0.0])
    out["phase2am_sidecar_control"] = "wrong_sidecar"
    return out


def _source_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    for row in rows:
        candidates = row.get("candidate_commands")
        slot = row.get("command_slot")
        if not isinstance(candidates, list) or not isinstance(slot, int):
            continue
        prediction = source_overlap_command_slot_prediction(str(row.get("state_prompt") or ""), candidates)
        total += 1
        correct += int(prediction == slot)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
    }


def _is_usable(row: dict[str, Any], min_gold_identity_score: float) -> bool:
    slot = row.get("command_slot")
    count = _candidate_count(row)
    if not isinstance(slot, int) or slot < 0 or slot >= count or count < 2:
        return False
    scores = _scores(row)
    if scores[slot] < min_gold_identity_score:
        return False
    erased = _erased(row)
    wrong = _wrong(row)
    return _scores(erased) != scores and _scores(wrong) != scores


def build_phase2am_natural_sidecar_controls(
    *,
    input_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
    min_rows: int = 64,
    min_gold_identity_score: float = 0.1,
    min_source_accuracy: float = 0.2,
    max_source_accuracy: float = 0.75,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    selected = [row for row in rows if _is_usable(row, min_gold_identity_score)]
    erased_rows = [_erased(row) for row in selected]
    wrong_rows = [_wrong(row) for row in selected]
    output = Path(output_dir)
    original_path = output / "original.jsonl"
    erased_path = output / "sidecar_erased.jsonl"
    wrong_path = output / "wrong_sidecar.jsonl"
    original_hash = _write_jsonl(original_path, selected)
    erased_hash = _write_jsonl(erased_path, erased_rows)
    wrong_hash = _write_jsonl(wrong_path, wrong_rows)
    source = _source_accuracy(selected)
    source_acc = source.get("accuracy")
    repos = sorted(
        {
            str((row.get("source_trace") or {}).get("repo_id"))
            for row in selected
            if isinstance(row.get("source_trace"), dict)
            and (row.get("source_trace") or {}).get("repo_id")
        }
    )
    checks = {
        "row_count_min": len(selected) >= min_rows,
        "source_overlap_nonzero": isinstance(source_acc, float)
        and source_acc >= min_source_accuracy,
        "source_overlap_not_ceiling": isinstance(source_acc, float)
        and source_acc <= max_source_accuracy,
        "all_selected_sidecar_usable": len(selected) > 0
        and len(erased_rows) == len(wrong_rows) == len(selected),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2am_natural_sidecar_controls",
        "passed": passed,
        "claim_bearing_training_candidate": False,
        "source_data_unchanged": True,
        "sealed_v3_used": False,
        "input_jsonl": str(Path(input_jsonl)),
        "row_count_input": len(rows),
        "row_count_selected": len(selected),
        "repo_ids": repos,
        "repo_count": len(repos),
        "checks": checks,
        "metrics": {
            "source_overlap_accuracy": source_acc,
            "source_overlap_total": source.get("total"),
            "source_overlap_correct": source.get("correct"),
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_gold_identity_score": min_gold_identity_score,
            "min_source_accuracy": min_source_accuracy,
            "max_source_accuracy": max_source_accuracy,
        },
        "splits": {
            "original": {"path": str(original_path), "sha256": original_hash},
            "sidecar_erased": {"path": str(erased_path), "sha256": erased_hash},
            "wrong_sidecar": {"path": str(wrong_path), "sha256": wrong_hash},
        },
        "blocked_actions": []
        if passed
        else [
            "do_not_train_phase2am",
            "do_not_claim_natural_sidecar_dependency",
        ],
        "claim_boundary": (
            "Phase2AM filters an existing non-sealed repo-disjoint head split to rows where "
            "the command-identity sidecar is present and perturbable. It does not alter prompts "
            "or candidate text and does not establish sealed transfer or open-ended repair."
        ),
    }
    _write_json(manifest_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AM natural sidecar-present original/erased/wrong controls."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--min-gold-identity-score", type=float, default=0.1)
    parser.add_argument("--min-source-accuracy", type=float, default=0.2)
    parser.add_argument("--max-source-accuracy", type=float, default=0.75)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2am_natural_sidecar_controls(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        min_rows=args.min_rows,
        min_gold_identity_score=args.min_gold_identity_score,
        min_source_accuracy=args.min_source_accuracy,
        max_source_accuracy=args.max_source_accuracy,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
