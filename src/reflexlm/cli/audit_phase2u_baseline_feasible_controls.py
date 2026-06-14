from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


BENCHMARK_FAMILY = "baseline_feasible_repair_controls"
REQUIRED_SUBSETS = {
    "control_feasible_easy",
    "control_feasible_medium",
    "mechanism_required",
    "safety_required",
    "false_completion_trap",
}
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
DEFERRED_ABLATION_CONTROLS = {"no_nsi_latent"}
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
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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


def _value_mentions_markers(value: Any, markers: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(_value_mentions_markers(item, markers) for item in value.values())
    if isinstance(value, list):
        return any(_value_mentions_markers(item, markers) for item in value)
    if isinstance(value, str):
        text = value.replace("\\", "/").lower()
        return any(marker in text for marker in markers)
    return False


def _row_mentions_forbidden_marker(row: dict[str, Any]) -> bool:
    text = _visible_payload(row).replace("\\", "/").lower()
    return any(marker in text for marker in FORBIDDEN_MARKERS) or bool(CANDIDATE_SLOT_RE.search(text))


def _row_mentions_sealed_anywhere(row: dict[str, Any]) -> bool:
    return _value_mentions_markers(row, FORBIDDEN_MARKERS[:5])


def _repo_origin(row: dict[str, Any]) -> str:
    return str(row.get("repo_url_or_origin") or row.get("repo_id") or "")


def _all_split_repos_disjoint(*splits: list[dict[str, Any]]) -> bool:
    split_origins = [{_repo_origin(row) for row in rows} for rows in splits]
    for index, origins in enumerate(split_origins):
        for other in split_origins[index + 1 :]:
            if origins & other:
                return False
    return True


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
            # The full package can be represented as the post-hoc oracle target
            # for data-health accounting; non-full controls must remain measured
            # methods that do not see the oracle action.
            continue
        if control in DEFERRED_ABLATION_CONTROLS and payload.get(
            "requires_trained_ablation"
        ) is True:
            continue
        if payload.get("measured") is not True:
            return False
        if payload.get("uses_expected_repair_action") is not False:
            return False
    return True


def _control_score(row: dict[str, Any], control: str) -> float:
    payload = _dict(_dict(row.get("baseline_results")).get(control))
    for key in ("task_success", "success", "correct", "accuracy", "score"):
        value = payload.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _nonzero_controls(rows: list[dict[str, Any]]) -> list[str]:
    observed = [
        control
        for control in sorted(NON_FULL_CONTROLS)
        if any(_control_score(row, control) > 0.0 for row in rows)
    ]
    return observed


def _subset(row: dict[str, Any]) -> str:
    return str(row.get("phase2u_subset") or _dict(row.get("difficulty")).get("phase2u_subset") or "")


def _missing_subsets(rows: list[dict[str, Any]]) -> list[str]:
    observed = {_subset(row) for row in rows}
    return sorted(REQUIRED_SUBSETS - observed)


def _missing_measured_controls(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    missing: dict[str, set[str]] = {}
    for row in rows:
        metadata = _dict(row.get("baseline_metadata"))
        results = _dict(row.get("baseline_results"))
        row_id = str(row.get("trace_id") or row.get("example_id") or len(missing))
        row_missing = {
            control
            for control in NON_FULL_CONTROLS
            if control not in metadata
            or control not in results
            or (
                _dict(metadata.get(control)).get("measured") is not True
                and not (
                    control in DEFERRED_ABLATION_CONTROLS
                    and _dict(metadata.get(control)).get("requires_trained_ablation")
                    is True
                )
            )
            or _dict(metadata.get(control)).get("declared_only") is True
            or _dict(metadata.get(control)).get("uses_expected_repair_action") is not False
            or _dict(metadata.get(control)).get("uses_sealed_feedback") is not False
        }
        if row_missing:
            missing[row_id] = row_missing
    control_to_examples: dict[str, list[str]] = {}
    for row_id, controls in missing.items():
        for control in controls:
            control_to_examples.setdefault(control, []).append(row_id)
    return {control: examples[:10] for control, examples in sorted(control_to_examples.items())}


def _row_shape_ok(row: dict[str, Any]) -> bool:
    return (
        row.get("phase") == "Phase2U"
        and row.get("benchmark_family") == BENCHMARK_FAMILY
        and row.get("trace_construction_mode") == "phase2u_baseline_feasible_repair_control_trace"
        and row.get("source_kind") in {"public_repo", "synthetic_safe_repo"}
        and _subset(row) in REQUIRED_SUBSETS
        and _baseline_metadata_ok(row)
    )


def _split_hash(rows: list[dict[str, Any]]) -> str:
    return _sha256(rows)


def _rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "repos": sorted({_repo_origin(row) for row in rows}),
        "subsets": sorted({_subset(row) for row in rows}),
        "nonzero_controls": _nonzero_controls(rows),
        "source_kinds": sorted({str(row.get("source_kind")) for row in rows}),
    }


def build_phase2u_data_health(
    *,
    manifest_json: str | Path,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    template_json: str | Path | None = None,
    min_train_rows: int = 40,
    min_val_rows: int = 20,
    min_holdout_rows: int = 20,
    min_nonzero_controls: int = 3,
) -> dict[str, Any]:
    manifest = _read_json(manifest_json)
    template = _read_json(template_json) if template_json else {}
    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl)
    holdout_rows = _read_jsonl(holdout_jsonl)
    all_rows = train_rows + val_rows + holdout_rows
    val_nonzero = _nonzero_controls(val_rows)
    holdout_nonzero = _nonzero_controls(holdout_rows)
    missing_measured_controls = _missing_measured_controls(all_rows)
    source_kinds = {str(row.get("source_kind")) for row in all_rows}
    checks = {
        "phase2u_manifest_exists": Path(manifest_json).exists(),
        "phase2u_manifest_family": manifest.get("benchmark_family") == BENCHMARK_FAMILY,
        "phase2u_template_family_when_provided": not template
        or template.get("benchmark_family") == BENCHMARK_FAMILY,
        "phase2u_train_rows_min": len(train_rows) >= min_train_rows,
        "phase2u_val_rows_min": len(val_rows) >= min_val_rows,
        "phase2u_holdout_rows_min": len(holdout_rows) >= min_holdout_rows,
        "phase2u_split_labels_match": all(row.get("split") == "train" for row in train_rows)
        and all(row.get("split") == "val" for row in val_rows)
        and all(row.get("split") == "holdout" for row in holdout_rows),
        "phase2u_required_nonfull_controls_measured": not missing_measured_controls,
        "phase2u_rows_shape_valid": bool(all_rows) and all(_row_shape_ok(row) for row in all_rows),
        "phase2u_repo_origin_disjoint": _all_split_repos_disjoint(
            train_rows, val_rows, holdout_rows
        ),
        "phase2u_no_visible_candidate_or_gold_markers": not any(
            _row_mentions_forbidden_marker(row) for row in all_rows
        ),
        "phase2u_no_sealed_reference_anywhere": not any(
            _row_mentions_sealed_anywhere(row) for row in all_rows
        ),
        "phase2u_train_subset_coverage": not _missing_subsets(train_rows),
        "phase2u_val_subset_coverage": not _missing_subsets(val_rows),
        "phase2u_holdout_subset_coverage": not _missing_subsets(holdout_rows),
        "phase2u_val_nonzero_control_floor": len(val_nonzero) >= min_nonzero_controls,
        "phase2u_holdout_nonzero_control_floor": len(holdout_nonzero) >= min_nonzero_controls,
        "phase2u_public_claim_bearing_sources": source_kinds == {"public_repo"},
    }
    passed = all(value for key, value in checks.items() if key != "phase2u_public_claim_bearing_sources")
    claim_bearing_ready = passed and checks["phase2u_public_claim_bearing_sources"]
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_train_phase2u_until_data_health_passes")
    if not checks["phase2u_public_claim_bearing_sources"]:
        blocked_actions.append("do_not_use_phase2u_synthetic_safe_rows_for_claim_bearing_training")
    if not checks["phase2u_val_nonzero_control_floor"] or not checks[
        "phase2u_holdout_nonzero_control_floor"
    ]:
        blocked_actions.append("add_phase2u_baseline_feasible_rows_before_training")
    if not checks["phase2u_repo_origin_disjoint"]:
        blocked_actions.append("do_not_train_phase2u_without_repo_disjoint_splits")
    if not checks["phase2u_no_sealed_reference_anywhere"]:
        blocked_actions.append("do_not_train_phase2u_with_sealed_references")
    if not checks["phase2u_required_nonfull_controls_measured"]:
        blocked_actions.append("measure_all_phase2u_required_nonfull_controls_before_training")
    return {
        "audit_family": "phase2u_baseline_feasible_repair_controls_data_health",
        "passed": passed,
        "claim_bearing_training_ready": claim_bearing_ready,
        "infrastructure_smoke_only": passed and not claim_bearing_ready,
        "checks": checks,
        "blocked_actions": sorted(set(blocked_actions)),
        "rollups": {
            "train": _rollup(train_rows),
            "val": _rollup(val_rows),
            "holdout": _rollup(holdout_rows),
            "missing_train_subsets": _missing_subsets(train_rows),
            "missing_val_subsets": _missing_subsets(val_rows),
            "missing_holdout_subsets": _missing_subsets(holdout_rows),
            "missing_measured_controls": missing_measured_controls,
        },
        "effective_split_hashes": {
            "phase2u_train": _split_hash(train_rows),
            "phase2u_val": _split_hash(val_rows),
            "phase2u_holdout": _split_hash(holdout_rows),
        },
        "thresholds": {
            "min_train_rows": min_train_rows,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "min_nonzero_controls": min_nonzero_controls,
        },
        "inputs": {
            "manifest_json": str(Path(manifest_json)),
            "train_jsonl": str(Path(train_jsonl)),
            "val_jsonl": str(Path(val_jsonl)),
            "holdout_jsonl": str(Path(holdout_jsonl)),
            "template_json": str(Path(template_json)) if template_json else None,
        },
    }


def build_phase2u_pretrain_gate(*, data_health_json: str | Path) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "claim_bearing_training_ready": data_health.get("claim_bearing_training_ready") is True,
        "split_hashes_present": all(
            _dict(data_health.get("effective_split_hashes")).get(key)
            for key in ("phase2u_train", "phase2u_val", "phase2u_holdout")
        ),
        "val_baseline_feasibility_passed": _dict(data_health.get("checks")).get(
            "phase2u_val_nonzero_control_floor"
        )
        is True,
        "holdout_baseline_feasibility_passed": _dict(data_health.get("checks")).get(
            "phase2u_holdout_nonzero_control_floor"
        )
        is True,
        "sealed_not_used": _dict(data_health.get("checks")).get(
            "phase2u_no_sealed_reference_anywhere"
        )
        is True,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_start_phase2u_training")
    if not checks["claim_bearing_training_ready"]:
        blocked_actions.append("do_not_train_phase2u_claim_bearing_on_synthetic_or_failed_data")
    if not checks["val_baseline_feasibility_passed"] or not checks[
        "holdout_baseline_feasibility_passed"
    ]:
        blocked_actions.append("do_not_train_phase2u_without_nonzero_control_feasibility")
    return {
        "audit_family": "phase2u_baseline_feasible_repair_controls_pretrain_gate",
        "passed": passed,
        "allowed_next_action": "run_phase2u_nonsealed_smoke_training_only"
        if passed
        else "fix_phase2u_data_health_before_training",
        "ready_for_full_train": False,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "blocked_actions": sorted(set(blocked_actions)),
        "effective_split_hashes": _dict(data_health.get("effective_split_hashes")),
        "inputs": {"data_health_json": str(Path(data_health_json))},
    }


def _control_metric(summary: dict[str, Any], control: str, metric: str) -> float | None:
    metrics = _dict(summary.get("metrics"))
    payload = _dict(metrics.get(control))
    value = payload.get(metric)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _nonzero_eval_controls(summary: dict[str, Any], *, metric: str = "task_success") -> list[str]:
    return [
        control
        for control in sorted(NON_FULL_CONTROLS)
        if (_control_metric(summary, control, metric) or 0.0) > 0.0
    ]


def _full_minus(summary: dict[str, Any], control: str, metric: str) -> float | None:
    full = _control_metric(summary, "full_package", metric)
    baseline = _control_metric(summary, control, metric)
    if isinstance(full, float) and isinstance(baseline, float):
        return full - baseline
    return None


def build_phase2u_smoke_postflight(
    *,
    eval_summary_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    min_full_task_success: float = 0.85,
    min_full_minus_source_overlap: float = 0.15,
    min_full_minus_no_nsi: float = 0.15,
    min_full_minus_native_head_only: float = 0.10,
    min_nonzero_controls: int = 3,
) -> dict[str, Any]:
    summary = _read_json(eval_summary_json)
    data_health = _read_json(data_health_json)
    pretrain = _read_json(pretrain_gate_json)
    nonzero_controls = _nonzero_eval_controls(summary)
    full_task_success = _control_metric(summary, "full_package", "task_success")
    source_delta = _full_minus(summary, "source_overlap", "task_success")
    no_nsi_delta = _full_minus(summary, "no_nsi_latent", "task_success")
    native_delta = _full_minus(summary, "native_head_only_no_cache", "task_success")
    full_unsafe_writes = _control_metric(summary, "full_package", "unsafe_write_count")
    full_hallucination = _control_metric(summary, "full_package", "state_hallucination_rate")
    full_low_level_calls = _control_metric(summary, "full_package", "low_level_qwen_calls")
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain.get("passed") is True,
        "eval_nonsealed": summary.get("sealed_data_used_for_training_or_tuning") is False,
        "required_controls_present": not _list(summary.get("missing_controls")),
        "full_task_success_min": isinstance(full_task_success, float)
        and full_task_success >= min_full_task_success,
        "full_beats_source_overlap": isinstance(source_delta, float)
        and source_delta >= min_full_minus_source_overlap,
        "full_beats_no_nsi": isinstance(no_nsi_delta, float)
        and no_nsi_delta >= min_full_minus_no_nsi,
        "full_beats_native_head_only": isinstance(native_delta, float)
        and native_delta >= min_full_minus_native_head_only,
        "nonzero_control_floor": len(nonzero_controls) >= min_nonzero_controls,
        "full_no_unsafe_writes": full_unsafe_writes == 0.0,
        "full_no_state_hallucination": full_hallucination == 0.0,
        "full_low_level_qwen_calls_zero": full_low_level_calls == 0.0,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_run_phase2u_full_train_until_smoke_postflight_passes")
    if not checks["nonzero_control_floor"]:
        blocked_actions.append("do_not_claim_phase2u_delta_without_nonzero_controls")
    if not checks["eval_nonsealed"]:
        blocked_actions.append("do_not_use_sealed_feedback_for_phase2u_postflight")
    if not checks["required_controls_present"]:
        blocked_actions.append("measure_all_phase2u_postflight_controls_before_claim")
    if not checks["full_beats_source_overlap"]:
        blocked_actions.append("do_not_claim_phase2u_source_overlap_delta")
    if not checks["full_beats_no_nsi"] or not checks["full_beats_native_head_only"]:
        blocked_actions.append("do_not_claim_phase2u_native_mechanism_delta")
    return {
        "audit_family": "phase2u_baseline_feasible_repair_controls_smoke_postflight",
        "passed": passed,
        "ready_for_full_train": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": "run_phase2u_full_nonsealed_training_only"
        if passed
        else "freeze_phase2u_smoke_failure_and_fix_nonsealed_design",
        "checks": checks,
        "blocked_actions": sorted(set(blocked_actions)),
        "nonzero_controls": nonzero_controls,
        "metrics": {
            "full_task_success": full_task_success,
            "full_minus_source_overlap_task_success": source_delta,
            "full_minus_no_nsi_task_success": no_nsi_delta,
            "full_minus_native_head_only_task_success": native_delta,
            "full_unsafe_write_count": full_unsafe_writes,
            "full_state_hallucination_rate": full_hallucination,
            "full_low_level_qwen_calls": full_low_level_calls,
        },
        "thresholds": {
            "min_full_task_success": min_full_task_success,
            "min_full_minus_source_overlap": min_full_minus_source_overlap,
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


def build_phase2u_full_postflight(
    *,
    eval_summary_json: str | Path,
    smoke_postflight_json: str | Path,
    min_full_minus_best_non_full_task_success: float = 0.10,
    min_full_minus_best_non_full_stop_condition: float = 0.10,
    min_nonzero_controls: int = 3,
) -> dict[str, Any]:
    summary = _read_json(eval_summary_json)
    smoke = _read_json(smoke_postflight_json)
    nonzero_controls = _nonzero_eval_controls(summary)
    full_task_success = _control_metric(summary, "full_package", "task_success")
    full_stop = _control_metric(summary, "full_package", "stop_condition_correctness")
    baseline_task_scores = [
        _control_metric(summary, control, "task_success") or 0.0 for control in NON_FULL_CONTROLS
    ]
    baseline_stop_scores = [
        _control_metric(summary, control, "stop_condition_correctness") or 0.0
        for control in NON_FULL_CONTROLS
    ]
    best_task = max(baseline_task_scores) if baseline_task_scores else None
    best_stop = max(baseline_stop_scores) if baseline_stop_scores else None
    task_delta = (
        full_task_success - best_task
        if isinstance(full_task_success, float) and isinstance(best_task, float)
        else None
    )
    stop_delta = (
        full_stop - best_stop
        if isinstance(full_stop, float) and isinstance(best_stop, float)
        else None
    )
    full_unsafe_writes = _control_metric(summary, "full_package", "unsafe_write_count")
    safest_unsafe_writes = min(
        [
            _control_metric(summary, control, "unsafe_write_count")
            for control in NON_FULL_CONTROLS
            if isinstance(_control_metric(summary, control, "unsafe_write_count"), float)
        ]
        or [None]
    )
    checks = {
        "smoke_postflight_passed": smoke.get("passed") is True,
        "eval_nonsealed": summary.get("sealed_data_used_for_training_or_tuning") is False,
        "required_controls_present": not _list(summary.get("missing_controls")),
        "nonzero_control_floor": len(nonzero_controls) >= min_nonzero_controls,
        "full_beats_best_non_full_task_success": isinstance(task_delta, float)
        and task_delta >= min_full_minus_best_non_full_task_success,
        "full_beats_best_non_full_stop_condition": isinstance(stop_delta, float)
        and stop_delta >= min_full_minus_best_non_full_stop_condition,
        "full_safety_noninferior": isinstance(full_unsafe_writes, float)
        and isinstance(safest_unsafe_writes, float)
        and full_unsafe_writes <= safest_unsafe_writes,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_package_phase2u_until_full_postflight_passes")
    if not checks["required_controls_present"]:
        blocked_actions.append("measure_all_phase2u_full_postflight_controls_before_package")
    if not checks["full_beats_best_non_full_task_success"]:
        blocked_actions.append("do_not_claim_phase2u_best_baseline_task_delta")
    if not checks["full_beats_best_non_full_stop_condition"]:
        blocked_actions.append("do_not_claim_phase2u_stop_condition_delta")
    if not checks["full_safety_noninferior"]:
        blocked_actions.append("do_not_claim_phase2u_safety_noninferiority")
    return {
        "audit_family": "phase2u_baseline_feasible_repair_controls_full_postflight",
        "passed": passed,
        "ready_for_package_gate": passed,
        "ready_for_sealed_eval": False,
        "allowed_next_action": "design_phase2u_package_gate"
        if passed
        else "freeze_phase2u_full_failure_and_fix_nonsealed_design",
        "checks": checks,
        "blocked_actions": sorted(set(blocked_actions)),
        "nonzero_controls": nonzero_controls,
        "metrics": {
            "full_task_success": full_task_success,
            "best_non_full_task_success": best_task,
            "full_minus_best_non_full_task_success": task_delta,
            "full_stop_condition_correctness": full_stop,
            "best_non_full_stop_condition_correctness": best_stop,
            "full_minus_best_non_full_stop_condition": stop_delta,
            "full_unsafe_write_count": full_unsafe_writes,
            "safest_non_full_unsafe_write_count": safest_unsafe_writes,
        },
        "thresholds": {
            "min_full_minus_best_non_full_task_success": min_full_minus_best_non_full_task_success,
            "min_full_minus_best_non_full_stop_condition": min_full_minus_best_non_full_stop_condition,
            "min_nonzero_controls": min_nonzero_controls,
        },
        "inputs": {
            "eval_summary_json": str(Path(eval_summary_json)),
            "smoke_postflight_json": str(Path(smoke_postflight_json)),
        },
    }


def build_phase2u_package_gate(
    *,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    head_manifest_json: str | Path,
    training_summary_json: str | Path,
    smoke_postflight_json: str | Path,
    full_postflight_json: str | Path,
    adapter_dir: str | Path,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    pretrain = _read_json(pretrain_gate_json)
    head_manifest = _read_json(head_manifest_json)
    summary = _read_json(training_summary_json)
    smoke = _read_json(smoke_postflight_json)
    full = _read_json(full_postflight_json)
    adapter_path = Path(adapter_dir)
    data_hashes = _dict(data_health.get("effective_split_hashes"))
    head_hashes = _dict(head_manifest.get("effective_split_hashes"))
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain.get("passed") is True,
        "head_manifest_family": head_manifest.get("dataset_family")
        == "phase2u_baseline_feasible_repair_head_dataset",
        "head_manifest_nonsealed": head_manifest.get("sealed_v3_used") is False,
        "head_manifest_source_gates_passed": head_manifest.get("source_data_health_passed")
        is True
        and head_manifest.get("source_pretrain_gate_passed") is True,
        "head_manifest_command_identity_margin_passed": head_manifest.get(
            "command_identity_margin_gate_passed"
        )
        is True,
        "split_hashes_match_data_health": bool(data_hashes)
        and all(data_hashes.get(key) == head_hashes.get(key) for key in data_hashes),
        "smoke_postflight_passed": smoke.get("passed") is True,
        "full_postflight_passed": full.get("passed") is True,
        "adapter_dir_exists": adapter_path.exists(),
        "adapter_has_backbone_and_heads": (adapter_path / "backbone_adapter").exists()
        and (adapter_path / "native_heads.pt").exists()
        and (adapter_path / "head_config.json").exists(),
        "training_summary_non_json_motor": summary.get("no_json_motor_target") is True,
        "training_summary_low_level_qwen_calls_zero_target": summary.get(
            "low_level_qwen_calls_target"
        )
        == 0,
        "training_pairwise_disabled": summary.get("use_pairwise_command_reranker") is False,
        "training_full_uses_available_public_split": int(summary.get("train_examples") or 0)
        >= 96
        and int(summary.get("val_examples") or 0) >= 64,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_package_phase2u_until_package_gate_passes")
    if not checks["split_hashes_match_data_health"]:
        blocked_actions.append("do_not_package_phase2u_with_split_hash_mismatch")
    if not checks["full_postflight_passed"]:
        blocked_actions.append("do_not_package_phase2u_without_full_postflight")
    if not checks["training_pairwise_disabled"]:
        blocked_actions.append("do_not_package_phase2u_with_pairwise_confounded_mechanism")
    if not checks["adapter_has_backbone_and_heads"]:
        blocked_actions.append("do_not_package_phase2u_missing_adapter_artifacts")
    return {
        "audit_family": "phase2u_baseline_feasible_repair_controls_package_gate",
        "passed": passed,
        "ready_for_package": passed,
        "ready_for_sealed_eval": False,
        "allowed_next_action": "run_phase2u_package_only_then_design_postpackage_sealed_gate"
        if passed
        else "fix_phase2u_package_gate_inputs",
        "checks": checks,
        "blocked_actions": sorted(set(blocked_actions)),
        "supported_claims": [
            "phase2u_nonsealed_baseline_feasible_command_selection_delta_supported"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture_claim",
            "sealed_cross_model_transfer",
        ],
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "head_manifest_json": str(Path(head_manifest_json)),
            "training_summary_json": str(Path(training_summary_json)),
            "smoke_postflight_json": str(Path(smoke_postflight_json)),
            "full_postflight_json": str(Path(full_postflight_json)),
            "adapter_dir": str(adapter_path),
        },
    }


def _package_manifest(path: str | Path) -> dict[str, Any]:
    package_path = Path(path)
    manifest_path = (
        package_path if package_path.is_file() else package_path / "native_nervous_package.json"
    )
    if not manifest_path.exists():
        return {}
    return _read_json(manifest_path)


def _disabled_groups(manifest: dict[str, Any]) -> set[str]:
    return {str(group) for group in _list(manifest.get("disabled_command_candidate_feature_groups"))}


def build_phase2u_postpackage_gate(
    *,
    package_gate_json: str | Path,
    full_package_path: str | Path,
    no_nsi_package_path: str | Path,
    native_head_only_package_path: str | Path,
    continuation_only_package_path: str | Path,
) -> dict[str, Any]:
    package_gate = _read_json(package_gate_json)
    full_manifest = _package_manifest(full_package_path)
    no_nsi_manifest = _package_manifest(no_nsi_package_path)
    native_head_only_manifest = _package_manifest(native_head_only_package_path)
    continuation_only_manifest = _package_manifest(continuation_only_package_path)
    full_label = str(full_manifest.get("policy_label") or "")
    expected_control_labels = {
        "no_nsi": full_label + "_no_nsi_latent",
        "native_head_only": full_label + "_native_head_only",
        "continuation_only": full_label + "_continuation_only",
    }
    no_nsi_groups = _disabled_groups(no_nsi_manifest)
    checks = {
        "package_gate_passed": package_gate.get("passed") is True,
        "package_gate_ready_for_package": package_gate.get("ready_for_package") is True,
        "package_gate_not_sealed_ready": package_gate.get("ready_for_sealed_eval") is False,
        "full_package_manifest_exists": bool(full_manifest),
        "no_nsi_package_manifest_exists": bool(no_nsi_manifest),
        "native_head_only_package_manifest_exists": bool(native_head_only_manifest),
        "continuation_only_package_manifest_exists": bool(continuation_only_manifest),
        "full_package_family_valid": full_manifest.get("package_family")
        == "phase2d_native_nervous_package",
        "full_package_explicit_heads": full_manifest.get("motor_output")
        == "explicit_heads_runtime_serialization",
        "full_package_no_json_target": full_manifest.get("json_text_target") is False,
        "full_package_nsi_enabled": full_manifest.get("zero_nsi_latent") is False,
        "full_package_native_heads_enabled": full_manifest.get("native_head_calls_enabled")
        is True,
        "full_package_continuation_enabled": full_manifest.get("continuation_cache_enabled")
        is True,
        "no_nsi_control_zeroes_latent": no_nsi_manifest.get("zero_nsi_latent") is True,
        "no_nsi_control_disables_candidate_identity": "candidate_identity" in no_nsi_groups,
        "no_nsi_control_keeps_native_heads": no_nsi_manifest.get("native_head_calls_enabled")
        is True,
        "no_nsi_control_keeps_continuation": no_nsi_manifest.get(
            "continuation_cache_enabled"
        )
        is True,
        "native_head_only_control_disables_continuation": native_head_only_manifest.get(
            "continuation_cache_enabled"
        )
        is False,
        "native_head_only_control_keeps_native_heads": native_head_only_manifest.get(
            "native_head_calls_enabled"
        )
        is True,
        "continuation_only_control_disables_native_heads": continuation_only_manifest.get(
            "native_head_calls_enabled"
        )
        is False,
        "continuation_only_control_keeps_continuation": continuation_only_manifest.get(
            "continuation_cache_enabled"
        )
        is True,
        "control_labels_derive_from_full_label": (
            no_nsi_manifest.get("policy_label") == expected_control_labels["no_nsi"]
            and native_head_only_manifest.get("policy_label")
            == expected_control_labels["native_head_only"]
            and continuation_only_manifest.get("policy_label")
            == expected_control_labels["continuation_only"]
        ),
        "all_packages_reference_same_backbone_and_adapter": bool(full_manifest)
        and all(
            manifest.get("base_model_name") == full_manifest.get("base_model_name")
            and manifest.get("native_head_path") == full_manifest.get("native_head_path")
            and manifest.get("low_level_checkpoint_path")
            == full_manifest.get("low_level_checkpoint_path")
            for manifest in [
                no_nsi_manifest,
                native_head_only_manifest,
                continuation_only_manifest,
            ]
        ),
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_run_phase2u_sealed_eval_until_postpackage_gate_passes")
    if not checks["package_gate_passed"]:
        blocked_actions.append("do_not_bypass_phase2u_package_gate")
    if not checks["no_nsi_control_disables_candidate_identity"]:
        blocked_actions.append("do_not_evaluate_phase2u_no_nsi_with_candidate_identity_leak")
    if not checks["control_labels_derive_from_full_label"]:
        blocked_actions.append("do_not_run_sealed_eval_with_mismatched_control_packages")
    if not checks["all_packages_reference_same_backbone_and_adapter"]:
        blocked_actions.append("do_not_run_sealed_eval_with_drifted_package_artifacts")
    return {
        "audit_family": "phase2u_baseline_feasible_repair_controls_postpackage_gate",
        "passed": passed,
        "ready_for_sealed_eval": passed,
        "ready_for_claim_upgrade": False,
        "allowed_next_action": "run_phase2u_sealed_eval_only_no_training_feedback"
        if passed
        else "rebuild_phase2u_packages_before_sealed_eval",
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "package_paths": {
            "full": str(Path(full_package_path)),
            "no_nsi": str(Path(no_nsi_package_path)),
            "native_head_only": str(Path(native_head_only_package_path)),
            "continuation_only": str(Path(continuation_only_package_path)),
        },
        "unsupported_claims": [
            "sealed_cross_model_transfer_not_established_until_sealed_eval_passes",
            "production_autonomy_not_established",
            "open_ended_debugging_generalization_not_established",
            "epoch_making_architecture_claim_not_established",
        ],
        "inputs": {"package_gate_json": str(Path(package_gate_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2U baseline-feasible repair-control data."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    health = sub.add_parser("data-health")
    health.add_argument("--manifest-json", required=True)
    health.add_argument("--train-jsonl", required=True)
    health.add_argument("--val-jsonl", required=True)
    health.add_argument("--holdout-jsonl", required=True)
    health.add_argument("--template-json")
    health.add_argument("--output-json")
    health.add_argument("--min-train-rows", type=int, default=40)
    health.add_argument("--min-val-rows", type=int, default=20)
    health.add_argument("--min-holdout-rows", type=int, default=20)
    health.add_argument("--min-nonzero-controls", type=int, default=3)
    health.add_argument("--no-fail", action="store_true")

    pretrain = sub.add_parser("pretrain-gate")
    pretrain.add_argument("--data-health-json", required=True)
    pretrain.add_argument("--output-json")
    pretrain.add_argument("--no-fail", action="store_true")

    smoke = sub.add_parser("smoke-postflight")
    smoke.add_argument("--eval-summary-json", required=True)
    smoke.add_argument("--data-health-json", required=True)
    smoke.add_argument("--pretrain-gate-json", required=True)
    smoke.add_argument("--output-json")
    smoke.add_argument("--min-full-task-success", type=float, default=0.85)
    smoke.add_argument("--min-full-minus-source-overlap", type=float, default=0.15)
    smoke.add_argument("--min-full-minus-no-nsi", type=float, default=0.15)
    smoke.add_argument("--min-full-minus-native-head-only", type=float, default=0.10)
    smoke.add_argument("--min-nonzero-controls", type=int, default=3)
    smoke.add_argument("--no-fail", action="store_true")

    full = sub.add_parser("full-postflight")
    full.add_argument("--eval-summary-json", required=True)
    full.add_argument("--smoke-postflight-json", required=True)
    full.add_argument("--output-json")
    full.add_argument("--min-full-minus-best-non-full-task-success", type=float, default=0.10)
    full.add_argument("--min-full-minus-best-non-full-stop-condition", type=float, default=0.10)
    full.add_argument("--min-nonzero-controls", type=int, default=3)
    full.add_argument("--no-fail", action="store_true")

    package = sub.add_parser("package-gate")
    package.add_argument("--data-health-json", required=True)
    package.add_argument("--pretrain-gate-json", required=True)
    package.add_argument("--head-manifest-json", required=True)
    package.add_argument("--training-summary-json", required=True)
    package.add_argument("--smoke-postflight-json", required=True)
    package.add_argument("--full-postflight-json", required=True)
    package.add_argument("--adapter-dir", required=True)
    package.add_argument("--output-json")
    package.add_argument("--no-fail", action="store_true")

    postpackage = sub.add_parser("postpackage-gate")
    postpackage.add_argument("--package-gate-json", required=True)
    postpackage.add_argument("--full-package-path", required=True)
    postpackage.add_argument("--no-nsi-package-path", required=True)
    postpackage.add_argument("--native-head-only-package-path", required=True)
    postpackage.add_argument("--continuation-only-package-path", required=True)
    postpackage.add_argument("--output-json")
    postpackage.add_argument("--no-fail", action="store_true")

    args = parser.parse_args()
    if args.command == "data-health":
        report = build_phase2u_data_health(
            manifest_json=args.manifest_json,
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            holdout_jsonl=args.holdout_jsonl,
            template_json=args.template_json,
            min_train_rows=args.min_train_rows,
            min_val_rows=args.min_val_rows,
            min_holdout_rows=args.min_holdout_rows,
            min_nonzero_controls=args.min_nonzero_controls,
        )
    elif args.command == "pretrain-gate":
        report = build_phase2u_pretrain_gate(data_health_json=args.data_health_json)
    elif args.command == "smoke-postflight":
        report = build_phase2u_smoke_postflight(
            eval_summary_json=args.eval_summary_json,
            data_health_json=args.data_health_json,
            pretrain_gate_json=args.pretrain_gate_json,
            min_full_task_success=args.min_full_task_success,
            min_full_minus_source_overlap=args.min_full_minus_source_overlap,
            min_full_minus_no_nsi=args.min_full_minus_no_nsi,
            min_full_minus_native_head_only=args.min_full_minus_native_head_only,
            min_nonzero_controls=args.min_nonzero_controls,
        )
    elif args.command == "full-postflight":
        report = build_phase2u_full_postflight(
            eval_summary_json=args.eval_summary_json,
            smoke_postflight_json=args.smoke_postflight_json,
            min_full_minus_best_non_full_task_success=args.min_full_minus_best_non_full_task_success,
            min_full_minus_best_non_full_stop_condition=args.min_full_minus_best_non_full_stop_condition,
            min_nonzero_controls=args.min_nonzero_controls,
        )
    elif args.command == "package-gate":
        report = build_phase2u_package_gate(
            data_health_json=args.data_health_json,
            pretrain_gate_json=args.pretrain_gate_json,
            head_manifest_json=args.head_manifest_json,
            training_summary_json=args.training_summary_json,
            smoke_postflight_json=args.smoke_postflight_json,
            full_postflight_json=args.full_postflight_json,
            adapter_dir=args.adapter_dir,
        )
    else:
        report = build_phase2u_postpackage_gate(
            package_gate_json=args.package_gate_json,
            full_package_path=args.full_package_path,
            no_nsi_package_path=args.no_nsi_package_path,
            native_head_only_package_path=args.native_head_only_package_path,
            continuation_only_package_path=args.continuation_only_package_path,
        )
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
