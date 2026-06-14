from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import _expected_slot
from reflexlm.cli.build_phase2af_hardened_structural_sidecar_split import (
    _read_jsonl,
    _row_candidate,
    _shortcut_key,
)


ASSERT_LITERAL_RE = re.compile(
    r">\s*assert\s+(?P<quote>['\"])(?P<literal>.+?)(?P=quote)\s+in\s+text",
    re.DOTALL,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _assert_literal(row: dict[str, Any]) -> str | None:
    runtime = row.get("runtime_visible_evidence")
    if not isinstance(runtime, dict):
        return None
    pytest_before = runtime.get("pytest_before_patch")
    if not isinstance(pytest_before, dict):
        return None
    stdout = str(pytest_before.get("stdout_excerpt") or "")
    match = ASSERT_LITERAL_RE.search(stdout)
    if not match:
        return None
    literal = " ".join(match.group("literal").split())
    return literal[:240] if literal else None


def _phase2ak_row(row: dict[str, Any]) -> dict[str, Any] | None:
    literal = _assert_literal(row)
    if not literal:
        return None
    converted = json.loads(json.dumps(row))
    candidates = converted.get("repair_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None
    try:
        expected_slot = _expected_slot(converted)
    except ValueError:
        return None
    expected_candidate = candidates[expected_slot]
    if not isinstance(expected_candidate, dict):
        return None

    expected_candidate["target_symbol"] = literal
    expected_candidate["phase2ak_source_cue"] = literal
    expected_candidate["phase2ak_source_cue_kind"] = "runtime_visible_assert_literal"
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, dict):
            candidate["identity_sidecar_ambiguity_control"] = True
            candidate["phase2ak_expected_candidate"] = index == expected_slot

    runtime = converted.get("runtime_visible_evidence")
    if isinstance(runtime, dict):
        structural_hashes = sorted(
            {
                str(candidate.get("structural_probe_hash") or "")
                for candidate in candidates
                if isinstance(candidate, dict) and candidate.get("structural_probe_hash")
            }
        )
        runtime["structural_probe_hashes"] = structural_hashes
        runtime["phase2ak_identity_probe_hashes_ambiguated"] = bool(structural_hashes)

    if expected_slot == 0:
        candidates[0], candidates[1] = candidates[1], candidates[0]

    converted["current_visible_text"] = "\n".join(
        [
            str(converted.get("current_visible_text") or ""),
            "Phase2AK controlled source cue: assertion literal is runtime-visible; "
            "identity probe hashes are intentionally ambiguous across all candidates.",
        ]
    ).strip()
    converted["benchmark_family"] = "phase2ak_source_cue_balanced_identity_ambiguous"
    converted["claim_boundary"] = (
        "nonsealed_controlled_source_cue_identity_ambiguous_pressure_not_natural_trace_claim"
    )
    converted["phase2ak_transform"] = {
        "source_cue_from_runtime_visible_assert_literal": True,
        "candidate_target_symbol_replaced_with_source_cue_for_expected_action": True,
        "identity_probe_hashes_replaced_with_all_candidate_hashes": True,
        "slot0_expected_candidate_reindexed_to_slot1": expected_slot == 0,
        "uses_expected_repair_action_for_nonsealed_controlled_source_cue_assignment": True,
        "uses_sealed_feedback": False,
    }
    converted["unsupported_claims"] = [
        "natural_public_trace_distribution",
        "sealed_transfer",
        "production_autonomy",
        "open_ended_debugging_generalization",
        "epoch_making_architecture",
    ]
    candidate = _row_candidate(converted, require_tie_residual_feasible=False)
    if candidate is None or _shortcut_key(candidate) != (1, 0):
        return None
    candidate["benchmark_family"] = converted["benchmark_family"]
    candidate["claim_boundary"] = converted["claim_boundary"]
    candidate["phase2ak_transform"] = converted["phase2ak_transform"]
    candidate["trace_hash"] = _sha256(candidate)
    return candidate


def _select_rows(rows: list[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        converted = _phase2ak_row(row)
        if converted is None:
            continue
        trace_id = str(converted.get("trace_id") or f"row:{index}")
        if trace_id in seen:
            continue
        seen.add(trace_id)
        selected.append(converted)
        if max_rows is not None and len(selected) >= max_rows:
            break
    return selected


def _bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        candidate = _row_candidate(row, require_tie_residual_feasible=False)
        if candidate is None:
            continue
        source, identity = _shortcut_key(candidate)
        counts[f"source_{source}_identity_{identity}"] += 1
    return dict(sorted(counts.items()))


def build_phase2ak_source_cue_balanced_identity_ambiguous(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_holdout_rows: int | None = None,
    min_train_rows: int = 16,
    min_val_rows: int = 16,
    min_holdout_rows: int = 16,
) -> dict[str, Any]:
    source_rows = {
        "train": _read_jsonl(train_jsonl),
        "val": _read_jsonl(val_jsonl),
        "holdout": _read_jsonl(holdout_jsonl),
    }
    split_rows = {
        "train": _select_rows(source_rows["train"], max_train_rows),
        "val": _select_rows(source_rows["val"], max_val_rows),
        "holdout": _select_rows(source_rows["holdout"], max_holdout_rows),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        _write_jsonl(output / f"{split}.jsonl", rows)

    split_counts = {split: len(rows) for split, rows in split_rows.items()}
    bucket_counts = {split: _bucket_counts(rows) for split, rows in split_rows.items()}
    checks = {
        "sealed_feedback_absent": True,
        "train_rows_min": split_counts["train"] >= min_train_rows,
        "val_rows_min": split_counts["val"] >= min_val_rows,
        "holdout_rows_min": split_counts["holdout"] >= min_holdout_rows,
        "train_source_correct_identity_wrong": bucket_counts["train"].get(
            "source_1_identity_0", 0
        )
        == split_counts["train"],
        "val_source_correct_identity_wrong": bucket_counts["val"].get(
            "source_1_identity_0", 0
        )
        == split_counts["val"],
        "holdout_source_correct_identity_wrong": bucket_counts["holdout"].get(
            "source_1_identity_0", 0
        )
        == split_counts["holdout"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ak_source_cue_balanced_identity_ambiguous",
        "passed": passed,
        "claim_bearing_natural_trace_evidence": False,
        "claim_bearing_controlled_pressure_evidence": passed,
        "checks": checks,
        "source_split_inputs": {
            "train": str(Path(train_jsonl)),
            "val": str(Path(val_jsonl)),
            "holdout": str(Path(holdout_jsonl)),
        },
        "source_split_counts": {split: len(rows) for split, rows in source_rows.items()},
        "split_counts": split_counts,
        "bucket_counts": bucket_counts,
        "split_hashes": {split: _sha256(rows) for split, rows in split_rows.items()},
        "transform": {
            "source_cue_from_runtime_visible_assert_literal": True,
            "identity_probe_hashes_replaced_with_all_candidate_hashes": True,
            "slot0_expected_candidate_reindexed_to_slot1": True,
            "uses_expected_repair_action_for_nonsealed_controlled_source_cue_assignment": True,
            "uses_sealed_feedback": False,
        },
        "blocked_actions": (
            []
            if passed
            else [
                "do_not_train_phase2ak_until_source_cue_balanced_split_passes",
                "do_not_package_phase2ak",
                "do_not_claim_source_residual_mechanism",
            ]
        ),
        "allowed_next_action": (
            "build_phase2ak_head_dataset_and_smoke_train"
            if passed
            else "revise_nonsealed_source_cue_balanced_design"
        ),
        "claim_boundary": (
            "Phase2AK is a non-sealed controlled pressure benchmark. It uses runtime-visible "
            "assertion literals to expose a source cue on the expected candidate and ambiguates "
            "identity probe hashes across all candidates. It may test whether the training path "
            "can learn source-cue selection under identity ambiguity, but it is not natural trace "
            "distribution evidence and cannot support sealed transfer, production autonomy, "
            "open-ended debugging generalization, or an epoch-making architecture claim."
        ),
    }
    if manifest_json:
        _write_json(manifest_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AK source-cue balanced identity-ambiguous pressure splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--max-holdout-rows", type=int)
    parser.add_argument("--min-train-rows", type=int, default=16)
    parser.add_argument("--min-val-rows", type=int, default=16)
    parser.add_argument("--min-holdout-rows", type=int, default=16)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2ak_source_cue_balanced_identity_ambiguous(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_holdout_rows=args.max_holdout_rows,
        min_train_rows=args.min_train_rows,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
