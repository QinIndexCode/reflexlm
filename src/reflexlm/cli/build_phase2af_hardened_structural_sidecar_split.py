from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2aa_candidate_selection_baseline_report import (
    _identity_prediction,
    _identity_text_ablated_state,
    _runtime_state_for_row,
    _source_overlap_prediction,
)
from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import _expected_slot
from reflexlm.cli.build_phase2s_head_dataset import (
    _candidate_commands as _phase2s_candidate_commands,
    _command_identity_signal,
    phase2s_repair_trace_to_head_row,
)
from reflexlm.llm.candidate_features import command_candidate_source_overlap_rows


CLAIM_BOUNDARY = (
    "hardened_structural_sidecar_requires_non_ceiling_measured_shortcuts"
)
MARKER_RE = re.compile(
    r"\b(?:candidate_[0-9]+|gold(?:en)?_?(?:slot|label|answer)|sealed[_ -]?v?[0-9]*|"
    r"expected_patch_candidate_slot)\b",
    re.IGNORECASE,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _has_forbidden_marker(row: dict[str, Any]) -> bool:
    text = "\n".join(
        [
            str(row.get("current_visible_text") or ""),
            json.dumps(row.get("runtime_visible_evidence") or {}, ensure_ascii=False),
        ]
    )
    return bool(MARKER_RE.search(text))


def measured_shortcut_predictions(row: dict[str, Any]) -> dict[str, Any]:
    expected_slot = int(_expected_slot(row))
    state = _runtime_state_for_row(row)
    text_ablated_state = _identity_text_ablated_state(state)
    predictions = {
        "raw_source_overlap": _source_overlap_prediction(state),
        "identity_text_ablated_source_overlap": _source_overlap_prediction(text_ablated_state),
        "runtime_identity_heuristic": _identity_prediction(state),
    }
    return {
        "expected_slot": expected_slot,
        "predictions": predictions,
        "correct": {name: prediction == expected_slot for name, prediction in predictions.items()},
    }


def _row_tie_residual_feasible(row: dict[str, Any]) -> bool:
    try:
        commands = _phase2s_candidate_commands(row)
        head_row = phase2s_repair_trace_to_head_row(row)
        identity = _command_identity_signal(row, commands)
        identity_margin = float(identity.get("command_identity_margin") or 0.0)
        source_slot = _source_overlap_unique_best_slot(
            str(head_row.get("state_prompt") or ""),
            list(head_row.get("candidate_commands") or []),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return identity_margin > 0.0 or source_slot is not None


def _row_candidate(
    row: dict[str, Any],
    *,
    require_tie_residual_feasible: bool = False,
) -> dict[str, Any] | None:
    candidates = row.get("repair_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None
    if _has_forbidden_marker(row):
        return None
    if require_tie_residual_feasible and not _row_tie_residual_feasible(row):
        return None
    try:
        shortcuts = measured_shortcut_predictions(row)
    except (KeyError, TypeError, ValueError):
        return None
    converted = json.loads(json.dumps(row))
    converted["benchmark_family"] = "phase2af_hardened_structural_sidecar"
    converted["claim_boundary"] = CLAIM_BOUNDARY
    converted["phase2af_measured_shortcuts"] = shortcuts
    converted["phase2af_selection_rule"] = {
        "requires_measured_shortcuts": True,
        "requires_non_ceiling_shortcut_distribution_at_split_level": True,
        "uses_expected_repair_action_for_offline_metric_only": True,
        "uses_sealed_feedback": False,
        "does_not_train_if_pretrain_gate_fails": True,
    }
    converted["unsupported_claims"] = [
        "freeform_patch_generation",
        "production_autonomy",
        "open_ended_debugging_generalization",
        "sealed_transfer",
        "epoch_making_architecture",
    ]
    converted["trace_hash"] = _sha256_text(_canonical_json(converted))
    return converted


def _repo_origin(row: dict[str, Any]) -> str:
    return str(row.get("repo_url_or_origin") or row.get("repo_id") or "")


def _split_repos(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    return {
        split: sorted({_repo_origin(row) for row in rows if _repo_origin(row)})
        for split, rows in split_rows.items()
    }


def _repo_overlaps(split_repos: dict[str, list[str]]) -> dict[str, list[str]]:
    names = list(split_repos)
    overlaps: dict[str, list[str]] = {}
    for index, name in enumerate(names):
        left = set(split_repos.get(name, []))
        for other in names[index + 1 :]:
            overlap = sorted(left & set(split_repos.get(other, [])))
            if overlap:
                overlaps[f"{name}__{other}"] = overlap
    return overlaps


def _split_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter(
        {
            "raw_source_overlap": 0,
            "identity_text_ablated_source_overlap": 0,
            "runtime_identity_heuristic": 0,
        }
    )
    for row in rows:
        shortcut = row.get("phase2af_measured_shortcuts")
        correct = shortcut.get("correct") if isinstance(shortcut, dict) else {}
        for name in totals:
            totals[name] += int(bool(correct.get(name)))
    total = len(rows)
    return {
        name: {"correct": int(correct), "total": total, "accuracy": correct / total if total else 0.0}
        for name, correct in totals.items()
    }


def _source_overlap_unique_best_slot(state_prompt: str, candidates: list[str]) -> int | None:
    rows = command_candidate_source_overlap_rows(state_prompt, candidates)
    scores = [row[1] for row in rows[: len(candidates)]]
    if not scores:
        return None
    best = max(scores)
    if best <= 0.0 or scores.count(best) != 1:
        return None
    return scores.index(best)


def _tie_residual_feasibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_reports: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            commands = _phase2s_candidate_commands(row)
            head_row = phase2s_repair_trace_to_head_row(row)
            label = int(head_row["command_slot"])
            identity = _command_identity_signal(row, commands)
            identity_margin = float(identity.get("command_identity_margin") or 0.0)
            source_slot = _source_overlap_unique_best_slot(
                str(head_row.get("state_prompt") or ""),
                list(head_row.get("candidate_commands") or []),
            )
        except (KeyError, TypeError, ValueError):
            row_reports.append(
                {
                    "row_index": index,
                    "trace_id": row.get("trace_id"),
                    "identity_tie": True,
                    "source_overlap_unique_best_slot": None,
                    "source_overlap_unique_best_correct": False,
                    "unresolved_identity_tie": True,
                }
            )
            continue
        identity_tie = identity_margin <= 0.0
        source_correct = source_slot == label if source_slot is not None else False
        row_reports.append(
            {
                "row_index": index,
                "trace_id": row.get("trace_id"),
                "repo_id": row.get("repo_id"),
                "command_slot": label,
                "identity_margin": identity_margin,
                "identity_tie": identity_tie,
                "source_overlap_unique_best_slot": source_slot,
                "source_overlap_unique_best_correct": source_correct,
                "unresolved_identity_tie": identity_tie and source_slot is None,
            }
        )
    identity_ties = [row for row in row_reports if row["identity_tie"]]
    unresolved = [row for row in row_reports if row["unresolved_identity_tie"]]
    return {
        "rows": len(row_reports),
        "identity_tie_rows": len(identity_ties),
        "unresolved_identity_tie_rows": len(unresolved),
        "unresolved_identity_tie_rate": len(unresolved) / len(row_reports)
        if row_reports
        else 0.0,
        "unresolved_by_repo": dict(
            sorted(Counter(str(row.get("repo_id") or "<unknown>") for row in unresolved).items())
        ),
        "unresolved_by_slot": dict(
            sorted(Counter(str(row.get("command_slot")) for row in unresolved).items())
        ),
    }


def _expected_slot_for_manifest(row: dict[str, Any]) -> int:
    try:
        return int(_expected_slot(row))
    except (KeyError, TypeError, ValueError):
        pass
    if "expected_patch_candidate_slot" in row:
        return int(row["expected_patch_candidate_slot"])
    shortcut = row.get("phase2af_measured_shortcuts")
    if isinstance(shortcut, dict) and "expected_slot" in shortcut:
        return int(shortcut["expected_slot"])
    raise ValueError(f"expected slot unavailable: {row.get('trace_id')}")


def _ordered_candidates(
    rows: list[dict[str, Any]],
    *,
    require_tie_residual_feasible: bool = False,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in (
            _row_candidate(
                row,
                require_tie_residual_feasible=require_tie_residual_feasible,
            )
            for row in rows
        )
        if row is not None
    ]

    def rank(row: dict[str, Any]) -> tuple[int, str]:
        correct = row["phase2af_measured_shortcuts"]["correct"]
        shortcut_failures = sum(int(not correct.get(name, False)) for name in correct)
        return (-shortcut_failures, str(row.get("repo_id") or ""), str(row.get("trace_id") or ""))

    return sorted(candidates, key=rank)


def _select_rows(rows: list[dict[str, Any]], *, max_rows: int | None) -> list[dict[str, Any]]:
    ordered = _ordered_candidates(rows)
    if max_rows is not None:
        ordered = ordered[:max_rows]
    return ordered


def _shortcut_key(row: dict[str, Any]) -> tuple[int, int]:
    correct = row["phase2af_measured_shortcuts"]["correct"]
    source_correct = bool(correct.get("identity_text_ablated_source_overlap"))
    identity_correct = bool(correct.get("runtime_identity_heuristic"))
    return int(source_correct), int(identity_correct)


def _read_candidate_sources(
    paths: list[str | Path],
    *,
    require_tie_residual_feasible: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        source_name = str(path)
        for row in _ordered_candidates(
            _read_jsonl(path),
            require_tie_residual_feasible=require_tie_residual_feasible,
        ):
            row["phase2af_source_jsonl"] = source_name
            rows.append(row)
    return rows


def _bucket_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_shortcut_key(row), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (str(row.get("repo_id") or ""), str(row.get("trace_id") or "")))
    return buckets


def _choose_stratified_counts(
    *,
    bucket_sizes: dict[tuple[int, int], int],
    min_rows: int,
    max_rows: int,
    min_source_accuracy: float,
    max_source_accuracy: float,
    max_identity_accuracy: float,
) -> dict[tuple[int, int], int]:
    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    best: tuple[tuple[float, ...], dict[tuple[int, int], int]] | None = None
    ranges = [range(bucket_sizes.get(key, 0) + 1) for key in keys]
    for c00 in ranges[0]:
        for c01 in ranges[1]:
            for c10 in ranges[2]:
                for c11 in ranges[3]:
                    counts = {(0, 0): c00, (0, 1): c01, (1, 0): c10, (1, 1): c11}
                    total = sum(counts.values())
                    if total < min_rows or total > max_rows:
                        continue
                    source_correct = c10 + c11
                    identity_correct = c01 + c11
                    source_accuracy = source_correct / total
                    identity_accuracy = identity_correct / total
                    if not (min_source_accuracy <= source_accuracy <= max_source_accuracy):
                        continue
                    if identity_accuracy > max_identity_accuracy:
                        continue
                    if c00 == 0 and c10 == 0:
                        continue
                    if c01 == 0 and c11 == 0:
                        continue
                    score = (
                        -total,
                        abs(source_accuracy - 0.40),
                        abs(identity_accuracy - min(max_identity_accuracy, 0.85)),
                        -min(c00 + c10, c01 + c11),
                    )
                    if best is None or score < best[0]:
                        best = (score, counts)
    if best is None:
        raise ValueError(
            "unable to compose a Phase2AF stratified split that satisfies measured shortcut thresholds; "
            f"bucket_sizes={bucket_sizes}"
        )
    return {key: value for key, value in best[1].items() if value > 0}


def _select_stratified_rows(
    rows: list[dict[str, Any]],
    *,
    min_rows: int,
    max_rows: int,
    min_source_accuracy: float,
    max_source_accuracy: float,
    max_identity_accuracy: float,
) -> list[dict[str, Any]]:
    buckets = _bucket_rows(rows)
    counts = _choose_stratified_counts(
        bucket_sizes={key: len(value) for key, value in buckets.items()},
        min_rows=min_rows,
        max_rows=max_rows,
        min_source_accuracy=min_source_accuracy,
        max_source_accuracy=max_source_accuracy,
        max_identity_accuracy=max_identity_accuracy,
    )
    selected: list[dict[str, Any]] = []
    for key in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        selected.extend(buckets.get(key, [])[: counts.get(key, 0)])
    return sorted(selected, key=lambda row: (str(row.get("repo_id") or ""), str(row.get("trace_id") or "")))


def _candidate_action(candidate: Any) -> str:
    return str(candidate.get("repair_action") or "") if isinstance(candidate, dict) else ""


def _with_recomputed_phase2af_fields(row: dict[str, Any]) -> dict[str, Any]:
    converted = json.loads(json.dumps(row))
    converted["phase2af_measured_shortcuts"] = measured_shortcut_predictions(converted)
    converted["trace_hash"] = _sha256_text(_canonical_json(converted))
    return converted


def _candidate_order_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Create train-only candidate-order invariance variants.

    The runtime-visible evidence and expected repair action are unchanged. Only
    the candidate list order changes, which is a legitimate invariance pressure
    for a slot-selection head and avoids hard-coding any specific test case.
    """
    candidates = row.get("repair_candidates")
    expected_action = str(row.get("expected_repair_action") or "")
    if not isinstance(candidates, list) or len(candidates) < 2 or not expected_action:
        return [row]
    current_slot = next(
        (
            index
            for index, candidate in enumerate(candidates)
            if _candidate_action(candidate) == expected_action
        ),
        -1,
    )
    if current_slot < 0:
        return [row]
    correct = candidates[current_slot]
    distractors = [candidate for index, candidate in enumerate(candidates) if index != current_slot]
    variants: list[dict[str, Any]] = []
    for target_slot in range(len(candidates)):
        reordered = list(distractors)
        reordered.insert(target_slot, correct)
        converted = json.loads(json.dumps(row))
        converted["repair_candidates"] = reordered
        if target_slot != current_slot:
            converted["trace_id"] = f"{row.get('trace_id')}:candidate_order_slot_{target_slot}"
            converted["phase2af_train_augmentation"] = {
                "type": "candidate_order_invariance",
                "train_only": True,
                "runtime_visible_evidence_unchanged": True,
                "expected_repair_action_unchanged": True,
                "sealed_feedback_used": False,
                "original_expected_slot": current_slot,
                "augmented_expected_slot": target_slot,
            }
        variants.append(_with_recomputed_phase2af_fields(converted))
    return variants


def _augment_train_candidate_order(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    variant_count = 0
    for row in rows:
        variants = _candidate_order_variants(row)
        variant_count += max(0, len(variants) - 1)
        augmented.extend(variants)
    return augmented, {
        "enabled": True,
        "method": "candidate_order_invariance_train_only",
        "rows_before": len(rows),
        "rows_after": len(augmented),
        "added_variants": variant_count,
        "runtime_visible_evidence_unchanged": True,
        "expected_repair_action_unchanged": True,
        "sealed_feedback_used": False,
    }


def _shortcut_bucket_count_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        f"source_{key[0]}_identity_{key[1]}": len(value)
        for key, value in sorted(_bucket_rows(rows).items())
    }


def _balance_shortcut_buckets_for_training(
    rows: list[dict[str, Any]],
    *,
    target_per_bucket: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Repeat/downsample train rows so measured shortcut buckets are equally represented.

    This is used only for the training split. Validation and holdout rows remain
    untouched so postflight gates still measure the preregistered distribution.
    """
    buckets = _bucket_rows(rows)
    nonempty = [(key, value) for key, value in sorted(buckets.items()) if value]
    if not nonempty:
        return rows, {
            "enabled": True,
            "target_per_bucket": 0,
            "before": {},
            "after": {},
            "unique_rows_before": len(rows),
            "rows_after": len(rows),
        }
    sizes = sorted(len(value) for _, value in nonempty)
    target = int(target_per_bucket or sizes[len(sizes) // 2])
    target = max(target, 1)
    balanced: list[dict[str, Any]] = []
    for _, bucket in nonempty:
        for index in range(target):
            balanced.append(bucket[index % len(bucket)])
    balanced = sorted(
        balanced,
        key=lambda row: (
            _shortcut_key(row),
            str(row.get("repo_id") or ""),
            str(row.get("trace_id") or ""),
        ),
    )
    return balanced, {
        "enabled": True,
        "target_per_bucket": target,
        "before": _shortcut_bucket_count_labels(rows),
        "after": _shortcut_bucket_count_labels(balanced),
        "unique_rows_before": len(rows),
        "rows_after": len(balanced),
        "duplicates_are_training_sampling_only": True,
    }


def _balance_slots_for_training(
    rows: list[dict[str, Any]],
    *,
    target_per_slot: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_expected_slot_for_manifest(row), []).append(row)
    nonempty = [(slot, value) for slot, value in sorted(buckets.items()) if value]
    if not nonempty:
        return rows, {
            "enabled": True,
            "target_per_slot": 0,
            "before": {},
            "after": {},
            "rows_after": len(rows),
        }
    sizes = sorted(len(value) for _, value in nonempty)
    target = int(target_per_slot or sizes[len(sizes) // 2])
    target = max(target, 1)
    balanced: list[dict[str, Any]] = []
    for _, bucket in nonempty:
        bucket = sorted(
            bucket,
            key=lambda row: (
                _shortcut_key(row),
                str(row.get("repo_id") or ""),
                str(row.get("trace_id") or ""),
            ),
        )
        for index in range(target):
            balanced.append(bucket[index % len(bucket)])
    balanced = sorted(
        balanced,
        key=lambda row: (
            _expected_slot_for_manifest(row),
            _shortcut_key(row),
            str(row.get("repo_id") or ""),
            str(row.get("trace_id") or ""),
        ),
    )
    before = Counter(_expected_slot_for_manifest(row) for row in rows)
    after = Counter(_expected_slot_for_manifest(row) for row in balanced)
    return balanced, {
        "enabled": True,
        "target_per_slot": target,
        "before": dict(sorted((str(key), value) for key, value in before.items())),
        "after": dict(sorted((str(key), value) for key, value in after.items())),
        "rows_after": len(balanced),
        "duplicates_are_training_sampling_only": True,
    }


def audit_phase2af_hardened_structural_sidecar_split(
    *,
    manifest_json: str | Path,
    output_json: str | Path | None = None,
    min_val_rows: int = 16,
    min_holdout_rows: int = 16,
    max_raw_source_accuracy: float = 0.75,
    max_runtime_identity_accuracy: float = 0.90,
    min_text_ablated_source_accuracy: float = 0.05,
    max_text_ablated_source_accuracy: float = 0.75,
    max_unresolved_identity_tie_rate: float = 0.0,
    require_repo_disjoint: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_json).read_text(encoding="utf-8-sig"))
    metrics = manifest.get("split_metrics") if isinstance(manifest.get("split_metrics"), dict) else {}
    counts = manifest.get("split_counts") if isinstance(manifest.get("split_counts"), dict) else {}

    val_metrics = metrics.get("val", {})
    holdout_metrics = metrics.get("holdout", {})

    def accuracy(split_metrics: dict[str, Any], name: str) -> float:
        payload = split_metrics.get(name) if isinstance(split_metrics, dict) else {}
        return float(payload.get("accuracy") or 0.0) if isinstance(payload, dict) else 0.0

    val_raw = accuracy(val_metrics, "raw_source_overlap")
    val_identity = accuracy(val_metrics, "runtime_identity_heuristic")
    val_text_ablated = accuracy(val_metrics, "identity_text_ablated_source_overlap")
    holdout_raw = accuracy(holdout_metrics, "raw_source_overlap")
    holdout_identity = accuracy(holdout_metrics, "runtime_identity_heuristic")
    holdout_text_ablated = accuracy(holdout_metrics, "identity_text_ablated_source_overlap")
    repo_overlaps = (
        manifest.get("repo_overlaps") if isinstance(manifest.get("repo_overlaps"), dict) else {}
    )
    tie_residual = (
        manifest.get("tie_residual_feasibility")
        if isinstance(manifest.get("tie_residual_feasibility"), dict)
        else {}
    )
    val_tie = (
        tie_residual.get("val") if isinstance(tie_residual.get("val"), dict) else {}
    )
    holdout_tie = (
        tie_residual.get("holdout")
        if isinstance(tie_residual.get("holdout"), dict)
        else {}
    )
    val_unresolved = float(val_tie.get("unresolved_identity_tie_rate") or 0.0)
    holdout_unresolved = float(holdout_tie.get("unresolved_identity_tie_rate") or 0.0)

    checks = {
        "artifact_family_ok": manifest.get("artifact_family") == "phase2af_hardened_structural_sidecar_split",
        "sealed_feedback_absent": manifest.get("sealed_feedback_used") is False,
        "split_hashes_present": all(
            split in (manifest.get("split_hashes") or {}) for split in ("train", "val", "holdout")
        ),
        "tie_residual_feasibility_present": all(
            split in tie_residual for split in ("train", "val", "holdout")
        ),
        "val_row_count_ok": int(counts.get("val") or 0) >= min_val_rows,
        "holdout_row_count_ok": int(counts.get("holdout") or 0) >= min_holdout_rows,
        "val_raw_source_not_ceiling": val_raw <= max_raw_source_accuracy,
        "holdout_raw_source_not_ceiling": holdout_raw <= max_raw_source_accuracy,
        "val_runtime_identity_not_sufficient": val_identity <= max_runtime_identity_accuracy,
        "holdout_runtime_identity_not_sufficient": holdout_identity <= max_runtime_identity_accuracy,
        "val_text_ablated_source_nonzero": val_text_ablated >= min_text_ablated_source_accuracy,
        "holdout_text_ablated_source_nonzero": holdout_text_ablated >= min_text_ablated_source_accuracy,
        "val_text_ablated_source_not_ceiling": val_text_ablated <= max_text_ablated_source_accuracy,
        "holdout_text_ablated_source_not_ceiling": holdout_text_ablated <= max_text_ablated_source_accuracy,
        "val_no_unresolved_identity_ties": val_unresolved <= max_unresolved_identity_tie_rate,
        "holdout_no_unresolved_identity_ties": holdout_unresolved
        <= max_unresolved_identity_tie_rate,
        "repo_origin_disjoint": not repo_overlaps if require_repo_disjoint else True,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2af_hardened_structural_sidecar_pretrain_gate",
        "passed": passed,
        "manifest_json": str(Path(manifest_json)),
        "thresholds": {
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "max_raw_source_accuracy": max_raw_source_accuracy,
            "max_runtime_identity_accuracy": max_runtime_identity_accuracy,
            "min_text_ablated_source_accuracy": min_text_ablated_source_accuracy,
            "max_text_ablated_source_accuracy": max_text_ablated_source_accuracy,
            "max_unresolved_identity_tie_rate": max_unresolved_identity_tie_rate,
            "require_repo_disjoint": require_repo_disjoint,
        },
        "checks": checks,
        "repo_overlaps": repo_overlaps,
        "split_metrics": metrics,
        "tie_residual_feasibility": tie_residual,
        "blocked_actions": []
        if passed
        else [
            "do_not_train_phase2af_full",
            "do_not_package_phase2af",
            "do_not_claim_hardened_structural_sidecar_mechanism",
        ],
        "next_step": "train_phase2af_smoke" if passed else "collect_or_design_nonsealed_hardened_rows",
    }
    if output_json is not None:
        _write_json(output_json, report)
    return report


def build_phase2af_hardened_structural_sidecar_split(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_holdout_rows: int | None = None,
    require_tie_residual_feasible_rows: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    split_inputs = {
        "train": Path(train_jsonl),
        "val": Path(val_jsonl),
        "holdout": Path(holdout_jsonl),
    }
    max_rows_by_split = {
        "train": max_train_rows,
        "val": max_val_rows,
        "holdout": max_holdout_rows,
    }
    split_rows: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, int] = {}
    eligible_counts: dict[str, int] = {}
    for split, path in split_inputs.items():
        raw_rows = _read_jsonl(path)
        eligible = _ordered_candidates(
            raw_rows,
            require_tie_residual_feasible=require_tie_residual_feasible_rows,
        )
        rows = _select_rows(raw_rows, max_rows=max_rows_by_split[split])
        if require_tie_residual_feasible_rows:
            rows = _ordered_candidates(
                raw_rows,
                require_tie_residual_feasible=True,
            )
            if max_rows_by_split[split] is not None:
                rows = rows[: max_rows_by_split[split]]
        source_counts[split] = len(raw_rows)
        eligible_counts[split] = len(eligible)
        split_rows[split] = rows
        _write_jsonl(output / f"{split}.jsonl", rows)

    manifest = {
        "artifact_family": "phase2af_hardened_structural_sidecar_split",
        "claim_boundary": CLAIM_BOUNDARY,
        "output_dir": str(output),
        "source_split_inputs": {split: str(path) for split, path in split_inputs.items()},
        "source_split_counts": source_counts,
        "eligible_split_counts": eligible_counts,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": {
            split: _sha256_text(_canonical_json(rows)) for split, rows in split_rows.items()
        },
        "split_metrics": {split: _split_metrics(rows) for split, rows in split_rows.items()},
        "tie_residual_feasibility": {
            split: _tie_residual_feasibility(rows) for split, rows in split_rows.items()
        },
        "split_repos": _split_repos(split_rows),
        "repo_overlaps": _repo_overlaps(_split_repos(split_rows)),
        "expected_slot_distribution": {
            split: dict(sorted(Counter(_expected_slot_for_manifest(row) for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "candidate_count_distribution": {
            split: dict(sorted(Counter(len(row.get("repair_candidates", [])) for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "selection_rule": (
            "measure shortcut controls on non-sealed public repair rows, then expose only rows "
            "to the pretrain gate; training is blocked unless split-level shortcut distributions "
            "are nonzero but not ceiling"
        ),
        "require_tie_residual_feasible_rows": require_tie_residual_feasible_rows,
        "freeform_patch_generation": False,
        "sealed_feedback_used": False,
        "next_gate": "phase2af_hardened_structural_sidecar_pretrain_gate",
    }
    _write_json(manifest_json, manifest)
    return manifest


def build_phase2af_stratified_hardened_structural_sidecar_split(
    *,
    train_jsonls: list[str | Path],
    val_jsonls: list[str | Path],
    holdout_jsonls: list[str | Path],
    output_dir: str | Path,
    manifest_json: str | Path,
    min_train_rows: int = 32,
    min_val_rows: int = 30,
    min_holdout_rows: int = 40,
    max_train_rows: int = 96,
    max_val_rows: int = 64,
    max_holdout_rows: int = 96,
    min_source_accuracy: float = 0.05,
    max_source_accuracy: float = 0.75,
    max_identity_accuracy: float = 0.90,
    balance_train_shortcut_buckets: bool = False,
    train_shortcut_bucket_target: int | None = None,
    augment_train_candidate_order: bool = False,
    balance_train_slots: bool = False,
    train_slot_target: int | None = None,
    require_tie_residual_feasible_rows: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    split_inputs = {
        "train": [Path(path) for path in train_jsonls],
        "val": [Path(path) for path in val_jsonls],
        "holdout": [Path(path) for path in holdout_jsonls],
    }
    row_bounds = {
        "train": (min_train_rows, max_train_rows),
        "val": (min_val_rows, max_val_rows),
        "holdout": (min_holdout_rows, max_holdout_rows),
    }
    split_rows: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, int] = {}
    eligible_counts: dict[str, int] = {}
    bucket_counts: dict[str, dict[str, int]] = {}
    train_sampling: dict[str, Any] = {"shortcut_bucket_balanced_train": False}
    train_slot_sampling: dict[str, Any] = {"slot_balanced_train": False}
    train_augmentation: dict[str, Any] = {"candidate_order_invariance_train_only": False}
    for split, paths in split_inputs.items():
        raw_count = sum(len(_read_jsonl(path)) for path in paths)
        candidates = _read_candidate_sources(
            paths,
            require_tie_residual_feasible=require_tie_residual_feasible_rows,
        )
        if split == "train" and augment_train_candidate_order:
            candidates, augmentation_report = _augment_train_candidate_order(candidates)
            train_augmentation = {
                "candidate_order_invariance_train_only": True,
                **augmentation_report,
            }
        min_rows, max_rows = row_bounds[split]
        rows = _select_stratified_rows(
            candidates,
            min_rows=min_rows,
            max_rows=max_rows,
            min_source_accuracy=min_source_accuracy,
            max_source_accuracy=max_source_accuracy,
            max_identity_accuracy=max_identity_accuracy,
        )
        if split == "train" and balance_train_shortcut_buckets:
            rows, balance_report = _balance_shortcut_buckets_for_training(
                rows,
                target_per_bucket=train_shortcut_bucket_target,
            )
            train_sampling = {
                "shortcut_bucket_balanced_train": True,
                **balance_report,
            }
        if split == "train" and balance_train_slots:
            rows, slot_balance_report = _balance_slots_for_training(
                rows,
                target_per_slot=train_slot_target,
            )
            train_slot_sampling = {
                "slot_balanced_train": True,
                **slot_balance_report,
            }
        source_counts[split] = raw_count
        eligible_counts[split] = len(candidates)
        bucket_counts[split] = {
            f"source_{key[0]}_identity_{key[1]}": len(value)
            for key, value in sorted(_bucket_rows(candidates).items())
        }
        split_rows[split] = rows
        _write_jsonl(output / f"{split}.jsonl", rows)

    manifest = {
        "artifact_family": "phase2af_hardened_structural_sidecar_split",
        "build_mode": "multi_source_stratified_shortcut_controls",
        "claim_boundary": CLAIM_BOUNDARY,
        "output_dir": str(output),
        "source_split_inputs": {
            split: [str(path) for path in paths] for split, paths in split_inputs.items()
        },
        "source_split_counts": source_counts,
        "eligible_split_counts": eligible_counts,
        "shortcut_bucket_counts": bucket_counts,
        "stratification_thresholds": {
            "min_source_accuracy": min_source_accuracy,
            "max_source_accuracy": max_source_accuracy,
            "max_identity_accuracy": max_identity_accuracy,
            "min_train_rows": min_train_rows,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "max_train_rows": max_train_rows,
            "max_val_rows": max_val_rows,
            "max_holdout_rows": max_holdout_rows,
        },
        "train_sampling": train_sampling,
        "train_slot_sampling": train_slot_sampling,
        "train_augmentation": train_augmentation,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": {
            split: _sha256_text(_canonical_json(rows)) for split, rows in split_rows.items()
        },
        "split_metrics": {split: _split_metrics(rows) for split, rows in split_rows.items()},
        "tie_residual_feasibility": {
            split: _tie_residual_feasibility(rows) for split, rows in split_rows.items()
        },
        "split_repos": _split_repos(split_rows),
        "repo_overlaps": _repo_overlaps(_split_repos(split_rows)),
        "expected_slot_distribution": {
            split: dict(sorted(Counter(_expected_slot_for_manifest(row) for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "candidate_count_distribution": {
            split: dict(sorted(Counter(len(row.get("repair_candidates", [])) for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "selection_rule": (
            "compose existing non-sealed public repair rows across measured shortcut buckets; "
            "do not alter row text or labels; require split-level nonzero-but-non-ceiling controls"
        ),
        "require_tie_residual_feasible_rows": require_tie_residual_feasible_rows,
        "freeform_patch_generation": False,
        "sealed_feedback_used": False,
        "next_gate": "phase2af_hardened_structural_sidecar_pretrain_gate",
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and gate Phase2AF hardened structural-sidecar splits."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--train-jsonl", required=True)
    build.add_argument("--val-jsonl", required=True)
    build.add_argument("--holdout-jsonl", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--manifest-json", required=True)
    build.add_argument("--max-train-rows", type=int)
    build.add_argument("--max-val-rows", type=int)
    build.add_argument("--max-holdout-rows", type=int)
    build.add_argument("--require-tie-residual-feasible-rows", action="store_true")

    stratified = sub.add_parser("build-stratified")
    stratified.add_argument("--train-jsonl", action="append", required=True)
    stratified.add_argument("--val-jsonl", action="append", required=True)
    stratified.add_argument("--holdout-jsonl", action="append", required=True)
    stratified.add_argument("--output-dir", required=True)
    stratified.add_argument("--manifest-json", required=True)
    stratified.add_argument("--min-train-rows", type=int, default=32)
    stratified.add_argument("--min-val-rows", type=int, default=30)
    stratified.add_argument("--min-holdout-rows", type=int, default=40)
    stratified.add_argument("--max-train-rows", type=int, default=96)
    stratified.add_argument("--max-val-rows", type=int, default=64)
    stratified.add_argument("--max-holdout-rows", type=int, default=96)
    stratified.add_argument("--balance-train-shortcut-buckets", action="store_true")
    stratified.add_argument("--train-shortcut-bucket-target", type=int)
    stratified.add_argument("--augment-train-candidate-order", action="store_true")
    stratified.add_argument("--balance-train-slots", action="store_true")
    stratified.add_argument("--train-slot-target", type=int)
    stratified.add_argument("--require-tie-residual-feasible-rows", action="store_true")

    audit = sub.add_parser("audit")
    audit.add_argument("--manifest-json", required=True)
    audit.add_argument("--output-json", required=True)
    audit.add_argument("--min-val-rows", type=int, default=16)
    audit.add_argument("--min-holdout-rows", type=int, default=16)
    audit.add_argument("--require-repo-disjoint", action="store_true")

    args = parser.parse_args()
    if args.command == "build":
        report = build_phase2af_hardened_structural_sidecar_split(
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            holdout_jsonl=args.holdout_jsonl,
            output_dir=args.output_dir,
            manifest_json=args.manifest_json,
            max_train_rows=args.max_train_rows,
            max_val_rows=args.max_val_rows,
            max_holdout_rows=args.max_holdout_rows,
            require_tie_residual_feasible_rows=args.require_tie_residual_feasible_rows,
        )
    elif args.command == "build-stratified":
        report = build_phase2af_stratified_hardened_structural_sidecar_split(
            train_jsonls=args.train_jsonl,
            val_jsonls=args.val_jsonl,
            holdout_jsonls=args.holdout_jsonl,
            output_dir=args.output_dir,
            manifest_json=args.manifest_json,
            min_train_rows=args.min_train_rows,
            min_val_rows=args.min_val_rows,
            min_holdout_rows=args.min_holdout_rows,
            max_train_rows=args.max_train_rows,
            max_val_rows=args.max_val_rows,
            max_holdout_rows=args.max_holdout_rows,
            balance_train_shortcut_buckets=args.balance_train_shortcut_buckets,
            train_shortcut_bucket_target=args.train_shortcut_bucket_target,
            augment_train_candidate_order=args.augment_train_candidate_order,
            balance_train_slots=args.balance_train_slots,
            train_slot_target=args.train_slot_target,
            require_tie_residual_feasible_rows=args.require_tie_residual_feasible_rows,
        )
    else:
        report = audit_phase2af_hardened_structural_sidecar_split(
            manifest_json=args.manifest_json,
            output_json=args.output_json,
            min_val_rows=args.min_val_rows,
            min_holdout_rows=args.min_holdout_rows,
            require_repo_disjoint=args.require_repo_disjoint,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
