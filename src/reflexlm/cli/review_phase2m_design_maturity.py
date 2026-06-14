from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_BASELINES = {
    "source_overlap",
    "native_head_only",
    "continuation_only",
    "prompt_only",
    "react",
}
CLAIM_BEARING_SOURCE_KINDS = {"public_repo"}
CANDIDATE_MARKER_RE = re.compile(r"candidate[_-]?(\d+)", re.IGNORECASE)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _load_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    candidate = Path(path)
    if not candidate.exists():
        return []
    return [
        json.loads(line)
        for line in candidate.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _candidate_commands(row: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for candidate in row.get("command_candidates", []):
        if isinstance(candidate, str):
            commands.append(candidate)
        elif isinstance(candidate, dict) and candidate.get("command") is not None:
            commands.append(str(candidate["command"]))
    return commands


def _runtime_evidence_text(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "current_visible_text": row.get("current_visible_text"),
            "runtime_visible_evidence": row.get("runtime_visible_evidence"),
        },
        ensure_ascii=False,
        sort_keys=True,
    ).lower()


def _baseline_metadata_ok(row: dict[str, Any]) -> bool:
    metadata = row.get("baseline_metadata")
    if not isinstance(metadata, dict):
        return False
    for baseline in REQUIRED_BASELINES:
        payload = metadata.get(baseline)
        if not isinstance(payload, dict):
            return False
        if payload.get("measured") is not True:
            return False
        if not payload.get("method"):
            return False
    return True


def _expected_candidate_marker_visible(row: dict[str, Any]) -> bool:
    expected = str(row.get("expected_command") or "")
    evidence = _runtime_evidence_text(row)
    expected_markers = {match.group(1) for match in CANDIDATE_MARKER_RE.finditer(expected)}
    return bool(expected_markers) and any(
        re.search(rf"candidate[_-]?{re.escape(marker)}", evidence)
        for marker in expected_markers
    )


def _baseline_accuracy(data_health: dict[str, Any], split: str, baseline: str) -> float | None:
    value = (
        data_health.get("rollups", {})
        .get("baselines", {})
        .get(split, {})
        .get(baseline, {})
        .get("accuracy")
    )
    return float(value) if isinstance(value, (int, float)) else None


def build_phase2m_design_maturity_review(
    *,
    data_health_json: str | Path,
    head_manifest_json: str | Path | None = None,
    head_state_baseline_json: str | Path | None = None,
) -> dict[str, Any]:
    data_health = _load_json(data_health_json)
    inputs = data_health.get("inputs", {})
    all_rows = (
        _load_jsonl(inputs.get("train_jsonl"))
        + _load_jsonl(inputs.get("val_jsonl"))
        + _load_jsonl(inputs.get("holdout_jsonl"))
    )
    source_kinds = sorted({str(row.get("source_kind")) for row in all_rows if row.get("source_kind")})
    direct_marker_rows = [
        str(row.get("trace_id"))
        for row in all_rows
        if _expected_candidate_marker_visible(row)
    ]
    baseline_metadata_missing_rows = [
        str(row.get("trace_id"))
        for row in all_rows
        if not _baseline_metadata_ok(row)
    ]
    head_manifest = _load_json(head_manifest_json) if head_manifest_json else {}
    head_state_baseline = _load_json(head_state_baseline_json) if head_state_baseline_json else {}
    source_overlap_val = _baseline_accuracy(data_health, "val", "source_overlap")
    native_val = _baseline_accuracy(data_health, "val", "native_head_only")

    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "head_dataset_exists": (
            bool(head_manifest)
            and head_manifest.get("dataset_family")
            == "phase2m_external_generalization_head_dataset"
            and head_manifest.get("sealed_v3_used") is False
            and head_manifest.get("json_text_target") is False
        ),
        "graded_dimensions_present": all(
            data_health.get("checks", {}).get(key) is True
            for key in (
                "phase2m_evidence_density_coverage",
                "phase2m_candidate_count_coverage",
                "phase2m_continuation_depth_coverage",
                "phase2m_ambiguity_class_coverage",
                "phase2m_trace_type_coverage",
            )
        ),
        "repo_disjoint_holdout": data_health.get("checks", {}).get(
            "phase2m_repo_disjoint_holdout"
        )
        is True,
        "sealed_not_used": data_health.get("checks", {}).get(
            "phase2m_no_sealed_reference_anywhere"
        )
        is True,
        "has_claim_bearing_public_repo_trace": bool(
            set(source_kinds) & CLAIM_BEARING_SOURCE_KINDS
        ),
        "baselines_are_measured_not_declared_only": bool(all_rows)
        and not baseline_metadata_missing_rows,
        "runtime_evidence_avoids_direct_candidate_slot_marker": bool(all_rows)
        and not direct_marker_rows,
        "baseline_pressure_not_degenerate": (
            isinstance(source_overlap_val, float)
            and isinstance(native_val, float)
            and source_overlap_val < 0.85
            and native_val < 0.85
            and (source_overlap_val > 0.0 or native_val > 0.0)
        ),
        "head_state_source_overlap_pressure_not_solved": (
            True
            if not head_state_baseline
            else head_state_baseline.get("checks", {}).get(
                "phase2m_head_state_source_overlap_val_below_threshold"
            )
            is True
        ),
    }
    ready_for_plumbing_smoke = all(
        checks[key]
        for key in (
            "data_health_passed",
            "head_dataset_exists",
            "graded_dimensions_present",
            "repo_disjoint_holdout",
            "sealed_not_used",
        )
    )
    ready_for_claim_bearing_training = ready_for_plumbing_smoke and all(
        checks[key]
        for key in (
            "has_claim_bearing_public_repo_trace",
            "baselines_are_measured_not_declared_only",
            "runtime_evidence_avoids_direct_candidate_slot_marker",
            "baseline_pressure_not_degenerate",
            "head_state_source_overlap_pressure_not_solved",
        )
    )
    blocked_actions: list[str] = []
    if not ready_for_plumbing_smoke:
        blocked_actions.append("do_not_run_phase2m_smoke_until_plumbing_ready")
    if not ready_for_claim_bearing_training:
        blocked_actions.append("do_not_treat_phase2m_smoke_as_claim_bearing_evidence")
        blocked_actions.append("do_not_start_phase2m_full_training_or_package")
    if not checks["has_claim_bearing_public_repo_trace"]:
        blocked_actions.append("collect_public_repo_readonly_traces_before_claim_training")
    if not checks["baselines_are_measured_not_declared_only"]:
        blocked_actions.append("measure_baselines_with_code_before_claim_training")
    if not checks["runtime_evidence_avoids_direct_candidate_slot_marker"]:
        blocked_actions.append("remove_direct_candidate_slot_markers_from_runtime_evidence")
    if not checks["head_state_source_overlap_pressure_not_solved"]:
        blocked_actions.append("revise_phase2m_head_prompt_or_trace_design_before_training")

    return {
        "review_family": "phase2m_design_maturity_review",
        "passed": ready_for_claim_bearing_training,
        "ready_for_plumbing_smoke": ready_for_plumbing_smoke,
        "ready_for_claim_bearing_training": ready_for_claim_bearing_training,
        "allowed_next_action": (
            "run_phase2m_plumbing_smoke_only_not_claim_evidence"
            if ready_for_plumbing_smoke and not ready_for_claim_bearing_training
            else "run_phase2m_claim_bearing_smoke_training"
            if ready_for_claim_bearing_training
            else "revise_phase2m_design_before_training"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "observations": {
            "source_kinds": source_kinds,
            "source_overlap_val_accuracy": source_overlap_val,
            "native_head_only_val_accuracy": native_val,
            "head_state_source_overlap_val_accuracy": (
                head_state_baseline.get("rollups", {})
                .get("source_overlap", {})
                .get("val", {})
                .get("accuracy")
                if head_state_baseline
                else None
            ),
            "baseline_metadata_missing_examples": baseline_metadata_missing_rows[:12],
            "direct_candidate_marker_examples": direct_marker_rows[:12],
        },
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "head_manifest_json": str(Path(head_manifest_json)) if head_manifest_json else None,
            "head_state_baseline_json": (
                str(Path(head_state_baseline_json)) if head_state_baseline_json else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Review Phase2M design maturity before training.")
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--head-manifest-json")
    parser.add_argument("--head-state-baseline-json")
    parser.add_argument("--output-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2m_design_maturity_review(
        data_health_json=args.data_health_json,
        head_manifest_json=args.head_manifest_json,
        head_state_baseline_json=args.head_state_baseline_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
