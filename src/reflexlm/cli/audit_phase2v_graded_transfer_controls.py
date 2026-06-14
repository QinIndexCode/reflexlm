from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


BENCHMARK_FAMILY = "graded_transfer_nonzero_controls"
REQUIRED_TIERS = {"control_feasible", "mixed_mechanism", "mechanism_required"}
REQUIRED_CONTROLS = {
    "full_package",
    "no_nsi_latent",
    "native_head_only_no_cache",
    "continuation_only",
    "prompt_only",
    "react",
    "source_overlap",
    "modern_coding_agent_loop",
}
NON_FULL_CONTROLS = REQUIRED_CONTROLS - {"full_package"}
FORBIDDEN_MARKERS = (
    "external_trace_v3_semantic_required",
    "phase2i_external_trace",
    "phase2g_external_trace",
    "sealed_failure",
    "sealed_feedback",
    "hidden_hint",
    "gold_label",
    "gold_command",
    "correct_patch",
    "expected_patch",
    "correct_command_hint",
)
CANDIDATE_SLOT_RE = re.compile(r"(?i)\bcandidate[_-]?\d+\b|\bslot[_-]?\d+\b")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _visible_payload(row: dict[str, Any]) -> str:
    return _canonical_json(
        {
            "current_visible_text": row.get("current_visible_text"),
            "runtime_visible_evidence": row.get("runtime_visible_evidence"),
            "repair_candidates": row.get("repair_candidates"),
        }
    )


def _mentions(value: Any, markers: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(_mentions(item, markers) for item in value.values())
    if isinstance(value, list):
        return any(_mentions(item, markers) for item in value)
    if isinstance(value, str):
        text = value.replace("\\", "/").lower()
        return any(marker in text for marker in markers)
    return False


def _row_mentions_forbidden_visible(row: dict[str, Any]) -> bool:
    text = _visible_payload(row).replace("\\", "/").lower()
    return any(marker in text for marker in FORBIDDEN_MARKERS) or bool(
        CANDIDATE_SLOT_RE.search(text)
    )


def _row_mentions_sealed_anywhere(row: dict[str, Any]) -> bool:
    return _mentions(row, FORBIDDEN_MARKERS[:5])


def _repo(row: dict[str, Any]) -> str:
    return str(row.get("repo_url_or_origin") or row.get("repo_id") or "")


def _trace_id(row: dict[str, Any]) -> str:
    return str(row.get("trace_id") or "")


def _source_trace_id(row: dict[str, Any]) -> str:
    return str(row.get("phase2v_source_trace_id") or "")


def _repos_disjoint(*splits: list[dict[str, Any]]) -> bool:
    repo_sets = [{_repo(row) for row in rows} for rows in splits]
    for index, repos in enumerate(repo_sets):
        for other in repo_sets[index + 1 :]:
            if repos & other:
                return False
    return True


def _tier(row: dict[str, Any]) -> str:
    return str(row.get("phase2v_tier") or _dict(row.get("difficulty")).get("phase2v_tier"))


def _control_score(row: dict[str, Any], control: str, metric: str = "task_success") -> float:
    payload = _dict(_dict(row.get("baseline_results")).get(control))
    value = payload.get(metric)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _control_means(rows: list[dict[str, Any]], metric: str = "task_success") -> dict[str, float]:
    if not rows:
        return {control: 0.0 for control in NON_FULL_CONTROLS}
    return {
        control: sum(_control_score(row, control, metric) for row in rows) / len(rows)
        for control in sorted(NON_FULL_CONTROLS)
    }


def _nonzero_controls(rows: list[dict[str, Any]]) -> list[str]:
    means = _control_means(rows)
    return [control for control, value in means.items() if value > 0.0]


def _best_nonfull(rows: list[dict[str, Any]]) -> float:
    means = _control_means(rows)
    return max(means.values()) if means else 0.0


def _baseline_metadata_ok(row: dict[str, Any]) -> bool:
    metadata = _dict(row.get("baseline_metadata"))
    results = _dict(row.get("baseline_results"))
    if not REQUIRED_CONTROLS.issubset(metadata) or not REQUIRED_CONTROLS.issubset(results):
        return False
    for control in REQUIRED_CONTROLS:
        payload = _dict(metadata.get(control))
        if payload.get("declared_only") is True:
            return False
        if payload.get("uses_sealed_feedback") is not False:
            return False
        if control == "full_package" and payload.get("oracle_reference") is True:
            continue
        if control == "no_nsi_latent" and payload.get("requires_trained_ablation") is True:
            continue
        if payload.get("measured") is not True:
            return False
        if payload.get("uses_expected_repair_action") is not False:
            return False
    return True


def _row_ok(row: dict[str, Any]) -> bool:
    return (
        row.get("phase") == "Phase2V"
        and row.get("benchmark_family") == BENCHMARK_FAMILY
        and row.get("trace_construction_mode")
        == "phase2v_graded_transfer_nonzero_control_trace"
        and row.get("source_kind") == "public_repo"
        and _tier(row) in REQUIRED_TIERS
        and _baseline_metadata_ok(row)
    )


def _split_hash(rows: list[dict[str, Any]]) -> str:
    return _sha256(rows)


def _rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "repos": sorted({_repo(row) for row in rows}),
        "tiers": sorted({_tier(row) for row in rows}),
        "nonzero_controls": _nonzero_controls(rows),
        "control_means": _control_means(rows),
        "best_nonfull_task_success": _best_nonfull(rows),
    }


def _overlap(left: set[str], right: set[str]) -> list[str]:
    return sorted(item for item in left & right if item)


def _pairwise_split_overlaps(
    split_values: dict[str, set[str]],
) -> dict[str, list[str]]:
    names = list(split_values)
    overlaps: dict[str, list[str]] = {}
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            values = _overlap(split_values[name], split_values[other])
            if values:
                overlaps[f"{name}__{other}"] = values
    return overlaps


def build_phase2v_data_health(
    *,
    manifest_json: str | Path,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    min_train_rows: int = 64,
    min_val_rows: int = 64,
    min_holdout_rows: int = 64,
    min_nonzero_controls: int = 3,
    min_best_nonfull: float = 0.20,
    max_best_nonfull: float = 0.75,
) -> dict[str, Any]:
    manifest = _read_json(manifest_json)
    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl)
    holdout_rows = _read_jsonl(holdout_jsonl)
    all_rows = train_rows + val_rows + holdout_rows
    val_best = _best_nonfull(val_rows)
    holdout_best = _best_nonfull(holdout_rows)
    checks = {
        "phase2v_manifest_family": manifest.get("benchmark_family") == BENCHMARK_FAMILY,
        "phase2v_train_rows_min": len(train_rows) >= min_train_rows,
        "phase2v_val_rows_min": len(val_rows) >= min_val_rows,
        "phase2v_holdout_rows_min": len(holdout_rows) >= min_holdout_rows,
        "phase2v_rows_shape_valid": bool(all_rows) and all(_row_ok(row) for row in all_rows),
        "phase2v_repo_origin_disjoint": _repos_disjoint(train_rows, val_rows, holdout_rows),
        "phase2v_required_tier_coverage": all(
            REQUIRED_TIERS.issubset({_tier(row) for row in rows})
            for rows in [train_rows, val_rows, holdout_rows]
        ),
        "phase2v_no_visible_candidate_or_gold_markers": not any(
            _row_mentions_forbidden_visible(row) for row in all_rows
        ),
        "phase2v_no_sealed_reference_anywhere": not any(
            _row_mentions_sealed_anywhere(row) for row in all_rows
        ),
        "phase2v_val_nonzero_control_floor": len(_nonzero_controls(val_rows))
        >= min_nonzero_controls,
        "phase2v_holdout_nonzero_control_floor": len(_nonzero_controls(holdout_rows))
        >= min_nonzero_controls,
        "phase2v_val_best_nonfull_in_range": min_best_nonfull <= val_best <= max_best_nonfull,
        "phase2v_holdout_best_nonfull_in_range": min_best_nonfull
        <= holdout_best
        <= max_best_nonfull,
    }
    passed = all(checks.values())
    blocked: list[str] = []
    if not passed:
        blocked.append("do_not_train_or_claim_phase2v_until_data_health_passes")
    if not checks["phase2v_val_nonzero_control_floor"] or not checks[
        "phase2v_holdout_nonzero_control_floor"
    ]:
        blocked.append("do_not_use_phase2v_all_zero_control_transfer")
    if not checks["phase2v_val_best_nonfull_in_range"] or not checks[
        "phase2v_holdout_best_nonfull_in_range"
    ]:
        blocked.append("rebalance_phase2v_controls_before_training")
    return {
        "audit_family": "phase2v_graded_transfer_data_health",
        "passed": passed,
        "claim_bearing_transfer_ready": passed,
        "checks": checks,
        "blocked_actions": sorted(set(blocked)),
        "rollups": {
            "train": _rollup(train_rows),
            "val": _rollup(val_rows),
            "holdout": _rollup(holdout_rows),
        },
        "effective_split_hashes": {
            "phase2v_train": _split_hash(train_rows),
            "phase2v_val": _split_hash(val_rows),
            "phase2v_holdout": _split_hash(holdout_rows),
        },
        "thresholds": {
            "min_train_rows": min_train_rows,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "min_nonzero_controls": min_nonzero_controls,
            "min_best_nonfull": min_best_nonfull,
            "max_best_nonfull": max_best_nonfull,
        },
        "inputs": {
            "manifest_json": str(Path(manifest_json)),
            "train_jsonl": str(Path(train_jsonl)),
            "val_jsonl": str(Path(val_jsonl)),
            "holdout_jsonl": str(Path(holdout_jsonl)),
        },
    }


def build_phase2v_independence_audit(
    *,
    manifest_json: str | Path,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    source_manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_json)
    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl)
    holdout_rows = _read_jsonl(holdout_jsonl)
    source_manifest = _read_json(source_manifest_json) if source_manifest_json else {}
    splits = {"train": train_rows, "val": val_rows, "holdout": holdout_rows}
    all_rows = train_rows + val_rows + holdout_rows
    repo_sets = {name: {_repo(row) for row in rows} for name, rows in splits.items()}
    trace_sets = {name: {_trace_id(row) for row in rows} for name, rows in splits.items()}
    source_trace_sets = {
        name: {_source_trace_id(row) for row in rows} for name, rows in splits.items()
    }
    repo_overlaps = _pairwise_split_overlaps(repo_sets)
    trace_overlaps = _pairwise_split_overlaps(trace_sets)
    source_trace_overlaps = _pairwise_split_overlaps(source_trace_sets)
    checks = {
        "phase2v_manifest_family": manifest.get("benchmark_family") == BENCHMARK_FAMILY,
        "phase2v_rows_present": bool(all_rows),
        "phase2v_trace_namespace": all(
            _trace_id(row).startswith("phase2v:") for row in all_rows
        ),
        "phase2v_source_trace_refs_present": all(_source_trace_id(row) for row in all_rows),
        "phase2v_trace_not_equal_source_trace": all(
            _trace_id(row) != _source_trace_id(row) for row in all_rows
        ),
        "phase2v_repo_origin_disjoint": not repo_overlaps,
        "phase2v_trace_ids_disjoint": not trace_overlaps,
        "phase2v_source_trace_ids_disjoint": not source_trace_overlaps,
        "phase2v_rows_not_phase2u_primary": all(row.get("phase") == "Phase2V" for row in all_rows),
        "phase2v_no_sealed_reference_anywhere": not any(
            _row_mentions_sealed_anywhere(row) for row in all_rows
        ),
        "phase2v_source_manifest_nonsealed": source_manifest.get("sealed_feedback_used")
        in (False, None),
    }
    passed = all(checks.values())
    blocked: list[str] = []
    if not passed:
        blocked.append("do_not_claim_phase2v_independent_transfer_until_fixed")
    if repo_overlaps:
        blocked.append("fix_phase2v_repo_origin_overlap")
    if trace_overlaps or source_trace_overlaps:
        blocked.append("fix_phase2v_trace_identity_overlap")
    if not checks["phase2v_trace_namespace"]:
        blocked.append("rewrite_phase2v_trace_ids_with_phase2v_namespace")
    return {
        "audit_family": "phase2v_transfer_independence_audit",
        "passed": passed,
        "checks": checks,
        "blocked_actions": sorted(set(blocked)),
        "repo_overlaps": repo_overlaps,
        "trace_overlaps": trace_overlaps,
        "source_trace_overlaps": source_trace_overlaps,
        "split_repos": {name: sorted(values) for name, values in repo_sets.items()},
        "counts": {name: len(rows) for name, rows in splits.items()},
        "identity_hashes": {
            "phase2v_trace_ids": _sha256(
                sorted(_trace_id(row) for row in all_rows)
            ),
            "phase2v_source_trace_ids": _sha256(
                sorted(_source_trace_id(row) for row in all_rows)
            ),
            "phase2v_split_repos": _sha256(
                {name: sorted(values) for name, values in repo_sets.items()}
            ),
        },
        "inputs": {
            "manifest_json": str(Path(manifest_json)),
            "train_jsonl": str(Path(train_jsonl)),
            "val_jsonl": str(Path(val_jsonl)),
            "holdout_jsonl": str(Path(holdout_jsonl)),
            "source_manifest_json": str(Path(source_manifest_json))
            if source_manifest_json
            else None,
        },
    }


def build_phase2v_pretrain_gate(*, data_health_json: str | Path) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "split_hashes_present": all(
            _dict(data_health.get("effective_split_hashes")).get(key)
            for key in ("phase2v_train", "phase2v_val", "phase2v_holdout")
        ),
        "nonzero_transfer_controls": _dict(data_health.get("checks")).get(
            "phase2v_val_nonzero_control_floor"
        )
        is True
        and _dict(data_health.get("checks")).get("phase2v_holdout_nonzero_control_floor")
        is True,
        "sealed_not_used": _dict(data_health.get("checks")).get(
            "phase2v_no_sealed_reference_anywhere"
        )
        is True,
    }
    passed = all(checks.values())
    return {
        "audit_family": "phase2v_graded_transfer_pretrain_gate",
        "passed": passed,
        "ready_for_eval": passed,
        "ready_for_training": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": "run_phase2v_package_eval_or_smoke_training"
        if passed
        else "fix_phase2v_data_health",
        "checks": checks,
        "blocked_actions": []
        if passed
        else ["do_not_evaluate_or_train_phase2v_until_pretrain_gate_passes"],
        "effective_split_hashes": _dict(data_health.get("effective_split_hashes")),
        "inputs": {"data_health_json": str(Path(data_health_json))},
    }


def _metric(summary: dict[str, Any], control: str, metric: str) -> float | None:
    value = _dict(_dict(summary.get("metrics")).get(control)).get(metric)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_phase2v_eval_postflight(
    *,
    eval_summary_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    min_full_task_success: float = 0.85,
    min_full_minus_best_nonfull: float = 0.15,
    min_full_minus_no_nsi: float = 0.15,
    min_full_minus_native_head_only: float = 0.10,
    min_nonzero_controls: int = 3,
) -> dict[str, Any]:
    summary = _read_json(eval_summary_json)
    data_health = _read_json(data_health_json)
    pretrain = _read_json(pretrain_gate_json)
    full = _metric(summary, "full_package", "task_success")
    control_scores = {
        control: _metric(summary, control, "task_success") or 0.0
        for control in sorted(NON_FULL_CONTROLS)
    }
    nonzero = [control for control, value in control_scores.items() if value > 0.0]
    best = max(control_scores.values()) if control_scores else 0.0
    no_nsi = control_scores.get("no_nsi_latent")
    native = control_scores.get("native_head_only_no_cache")
    full_hallucination = _metric(summary, "full_package", "state_hallucination_rate")
    full_low_level = _metric(summary, "full_package", "low_level_qwen_calls")
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain.get("passed") is True,
        "eval_nonsealed": summary.get("sealed_data_used_for_training_or_tuning") is False,
        "row_level_predictions_recomputed": summary.get(
            "phase2v_row_level_predictions_recomputed"
        )
        is True,
        "required_controls_present": not _list(summary.get("missing_controls")),
        "nonzero_control_floor": len(nonzero) >= min_nonzero_controls,
        "best_nonfull_nonzero": best > 0.0,
        "full_task_success_min": isinstance(full, float) and full >= min_full_task_success,
        "full_beats_best_nonfull": isinstance(full, float)
        and full - best >= min_full_minus_best_nonfull,
        "full_beats_no_nsi": isinstance(full, float)
        and isinstance(no_nsi, float)
        and full - no_nsi >= min_full_minus_no_nsi,
        "full_beats_native_head_only": isinstance(full, float)
        and isinstance(native, float)
        and full - native >= min_full_minus_native_head_only,
        "full_no_state_hallucination": full_hallucination == 0.0,
        "full_low_level_qwen_calls_zero": full_low_level == 0.0,
    }
    passed = all(checks.values())
    blocked: list[str] = []
    if not passed:
        blocked.append("do_not_claim_phase2v_graded_transfer")
    if not checks["nonzero_control_floor"] or not checks["best_nonfull_nonzero"]:
        blocked.append("do_not_use_phase2v_all_zero_controls")
    if not checks["full_beats_best_nonfull"]:
        blocked.append("freeze_phase2v_mechanism_insufficiency")
    if not checks["row_level_predictions_recomputed"]:
        blocked.append("do_not_treat_phase2v_derived_summary_as_claim_bearing_eval")
    return {
        "audit_family": "phase2v_graded_transfer_eval_postflight",
        "passed": passed,
        "ready_for_evidence_synthesis": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": "build_phase2v_evidence_sufficiency_report"
        if passed
        else "freeze_phase2v_failure_audit",
        "checks": checks,
        "blocked_actions": sorted(set(blocked)),
        "nonzero_controls": nonzero,
        "metrics": {
            "full_task_success": full,
            "best_nonfull_task_success": best,
            "full_minus_best_nonfull_task_success": full - best
            if isinstance(full, float)
            else None,
            "no_nsi_task_success": no_nsi,
            "native_head_only_task_success": native,
            "full_minus_no_nsi_task_success": full - no_nsi
            if isinstance(full, float) and isinstance(no_nsi, float)
            else None,
            "full_minus_native_head_only_task_success": full - native
            if isinstance(full, float) and isinstance(native, float)
            else None,
            "full_state_hallucination_rate": full_hallucination,
            "full_low_level_qwen_calls": full_low_level,
        },
        "thresholds": {
            "min_full_task_success": min_full_task_success,
            "min_full_minus_best_nonfull": min_full_minus_best_nonfull,
            "min_full_minus_no_nsi": min_full_minus_no_nsi,
            "min_full_minus_native_head_only": min_full_minus_native_head_only,
            "min_nonzero_controls": min_nonzero_controls,
        },
        "inputs": {
            "eval_summary_json": str(Path(eval_summary_json)),
            "data_health_json": str(Path(data_health_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
        },
    }


def build_phase2v_sealed_block_gate(*, eval_postflight_json: str | Path) -> dict[str, Any]:
    postflight = _read_json(eval_postflight_json)
    return {
        "audit_family": "phase2v_sealed_block_gate",
        "passed": postflight.get("passed") is True,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": "do_not_run_sealed_from_phase2v_without_separate_preregistration",
        "blocked_actions": [
            "do_not_run_phase2v_sealed_eval",
            "do_not_tune_from_sealed_v3",
            "do_not_upgrade_to_production_autonomy_or_epoch_making_claim",
        ],
        "inputs": {"eval_postflight_json": str(Path(eval_postflight_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2V graded-transfer controls.")
    sub = parser.add_subparsers(dest="command", required=True)
    health = sub.add_parser("data-health")
    health.add_argument("--manifest-json", required=True)
    health.add_argument("--train-jsonl", required=True)
    health.add_argument("--val-jsonl", required=True)
    health.add_argument("--holdout-jsonl", required=True)
    health.add_argument("--output-json")
    health.add_argument("--no-fail", action="store_true")
    pretrain = sub.add_parser("pretrain-gate")
    pretrain.add_argument("--data-health-json", required=True)
    pretrain.add_argument("--output-json")
    pretrain.add_argument("--no-fail", action="store_true")
    independence = sub.add_parser("independence-audit")
    independence.add_argument("--manifest-json", required=True)
    independence.add_argument("--train-jsonl", required=True)
    independence.add_argument("--val-jsonl", required=True)
    independence.add_argument("--holdout-jsonl", required=True)
    independence.add_argument("--source-manifest-json")
    independence.add_argument("--output-json")
    independence.add_argument("--no-fail", action="store_true")
    postflight = sub.add_parser("eval-postflight")
    postflight.add_argument("--eval-summary-json", required=True)
    postflight.add_argument("--data-health-json", required=True)
    postflight.add_argument("--pretrain-gate-json", required=True)
    postflight.add_argument("--output-json")
    postflight.add_argument("--no-fail", action="store_true")
    sealed = sub.add_parser("sealed-block-gate")
    sealed.add_argument("--eval-postflight-json", required=True)
    sealed.add_argument("--output-json")
    sealed.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    if args.command == "data-health":
        report = build_phase2v_data_health(
            manifest_json=args.manifest_json,
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            holdout_jsonl=args.holdout_jsonl,
        )
    elif args.command == "pretrain-gate":
        report = build_phase2v_pretrain_gate(data_health_json=args.data_health_json)
    elif args.command == "independence-audit":
        report = build_phase2v_independence_audit(
            manifest_json=args.manifest_json,
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            holdout_jsonl=args.holdout_jsonl,
            source_manifest_json=args.source_manifest_json,
        )
    elif args.command == "eval-postflight":
        report = build_phase2v_eval_postflight(
            eval_summary_json=args.eval_summary_json,
            data_health_json=args.data_health_json,
            pretrain_gate_json=args.pretrain_gate_json,
        )
    else:
        report = build_phase2v_sealed_block_gate(
            eval_postflight_json=args.eval_postflight_json
        )
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
