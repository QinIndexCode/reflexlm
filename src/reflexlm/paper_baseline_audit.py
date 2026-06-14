from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ZERO_CATEGORIES = {
    "expected_zero_due_to_missing_capability",
    "not_evaluable_for_control",
    "valid_zero_failure",
    "suspicious_zero_requires_redesign",
}


@dataclass(frozen=True)
class EvidenceSource:
    phase: str
    evaluation_scope: str
    source_kind: str
    path: str
    sanity_path: str | None = None


DEFAULT_EVIDENCE_SOURCES = [
    EvidenceSource(
        phase="Phase2M-v2",
        evaluation_scope="sealed_v3_semantic_required",
        source_kind="baseline_table",
        path="artifacts/reports/phase2m_v2_external_trace_v3_semantic_required/"
        "phase2m_v2_external_trace_v3_exact_baseline_table.json",
        sanity_path="artifacts/reports/phase2m_v2_claim_bearing/"
        "phase2m_v2_public_relationkey_full_postflight.json",
    ),
    EvidenceSource(
        phase="Phase2Q",
        evaluation_scope="sealed_v3_semantic_required",
        source_kind="baseline_table",
        path="artifacts/reports/phase2q_external_trace_v3_semantic_required/"
        "phase2q_external_trace_v3_exact_baseline_table.json",
        sanity_path="artifacts/reports/phase2q_public_trace_breadth/"
        "phase2q_public_trace_breadth_full_summary.json",
    ),
    EvidenceSource(
        phase="Phase2R",
        evaluation_scope="sealed_v3_semantic_required",
        source_kind="baseline_table",
        path="artifacts/reports/phase2r_external_trace_v3_semantic_required/"
        "phase2r_external_trace_v3_exact_baseline_table.json",
        sanity_path="artifacts/reports/phase2r_dynamic_public_trace/"
        "phase2r_dynamic_public_trace_full_summary.json",
    ),
    EvidenceSource(
        phase="Phase2L",
        evaluation_scope="sealed_v3_transfer_failure",
        source_kind="eval_family",
        path="artifacts/reports/phase2l_external_trace_v3_semantic_required",
        sanity_path="artifacts/reports/phase2l_counterfactual_continuation/"
        "phase2l_full1024_checkpointed_postflight.json",
    ),
    EvidenceSource(
        phase="Phase2P",
        evaluation_scope="sealed_cross_model_transfer_aggregate",
        source_kind="aggregate_summary",
        path="artifacts/reports/phase2p_sealed_cross_model_transfer/"
        "phase2p_multiseed_cross_model_transfer_summary.json",
        sanity_path="artifacts/reports/phase2n_multimodel_multiseed_reproduction/"
        "phase2n_qwen2_5_7b_reference_multiseed_summary.json",
    ),
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _nested_mean(payload: dict[str, Any], key: str) -> float | None:
    aggregate = payload.get("metrics", {}).get("aggregate", {})
    value = aggregate.get(key)
    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
        return float(value["mean"])
    if isinstance(payload.get(key), (int, float)):
        return float(payload[key])
    return None


def _rows_from_baseline_table(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows = []
    for row in payload.get("rows", []):
        rows.append(
            {
                "policy": row.get("policy", "unknown"),
                "completion": _as_float(row.get("completion")),
                "positives_over_episodes": row.get("positives/episodes"),
                "gate_status": row.get("gate_status"),
                "eval_json": row.get("eval_json"),
            }
        )
    return rows


def _rows_from_phase2l_eval_family(path: Path) -> list[dict[str, Any]]:
    files = {
        "prompt-only 7B": "phase2l_prompt_only_sealed_v3_eval.json",
        "ReAct 7B": "phase2l_react_sealed_v3_eval.json",
        "no-NSI latent": "phase2l_no_nsi_sealed_v3_eval.json",
        "native-head-only": "phase2l_native_head_only_sealed_v3_eval.json",
        "continuation-only": "phase2l_continuation_only_sealed_v3_eval.json",
        "Phase2L full package": "phase2l_full_sealed_v3_eval.json",
    }
    rows = []
    for policy, filename in files.items():
        eval_path = path / filename
        payload = _load_json(eval_path)
        completion = _nested_mean(payload, "task_completion_rate")
        rows.append(
            {
                "policy": policy,
                "completion": completion,
                "positives_over_episodes": None,
                "gate_status": "sealed_transfer_failure" if policy.endswith("full package") else "control",
                "eval_json": str(eval_path),
            }
        )
    return rows


def _rows_from_phase2p_aggregate(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    aggregate = payload.get("aggregate", {})
    return [
        {
            "policy": "Phase2P full package aggregate minimum",
            "completion": _as_float(aggregate.get("full_completion_min")),
            "positives_over_episodes": None,
            "gate_status": "sealed_cross_model_transfer_gate",
            "eval_json": str(path),
        },
        {
            "policy": "prompt-only 7B aggregate maximum",
            "completion": _as_float(aggregate.get("prompt_completion_max")),
            "positives_over_episodes": None,
            "gate_status": "text_baseline",
            "eval_json": str(path),
        },
        {
            "policy": "ReAct 7B aggregate maximum",
            "completion": _as_float(aggregate.get("react_completion_max")),
            "positives_over_episodes": None,
            "gate_status": "text_baseline",
            "eval_json": str(path),
        },
        {
            "policy": "no-NSI latent aggregate maximum",
            "completion": _as_float(aggregate.get("no_nsi_completion_max")),
            "positives_over_episodes": None,
            "gate_status": "mechanism_ablation",
            "eval_json": str(path),
        },
        {
            "policy": "native-head-only aggregate maximum",
            "completion": _as_float(aggregate.get("native_head_only_completion_max")),
            "positives_over_episodes": None,
            "gate_status": "mechanism_ablation",
            "eval_json": str(path),
        },
        {
            "policy": "continuation-only aggregate maximum",
            "completion": _as_float(aggregate.get("continuation_only_completion_max")),
            "positives_over_episodes": None,
            "gate_status": "mechanism_ablation",
            "eval_json": str(path),
        },
    ]


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normal_policy(policy: str) -> str:
    return policy.lower().replace("_", "-")


def _zero_category(policy: str, completion: float | None) -> str | None:
    if completion is None or completion != 0.0:
        return None
    policy_key = _normal_policy(policy)
    if "full package" in policy_key:
        return "valid_zero_failure"
    if "native-head-only" in policy_key or "continuation-only" in policy_key:
        return "expected_zero_due_to_missing_capability"
    if "no-nsi" in policy_key:
        return "expected_zero_due_to_missing_capability"
    if "prompt-only" in policy_key or "react" in policy_key:
        return "valid_zero_failure"
    return "suspicious_zero_requires_redesign"


def _sanity_metrics(root: Path, source: EvidenceSource) -> dict[str, Any]:
    if not source.sanity_path:
        return {"available": False}
    payload = _load_json(root / source.sanity_path)
    if not payload:
        return {"available": False, "path": source.sanity_path}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    buckets = [
        payload.get("metrics", {}),
        payload.get("full_nonsealed", {}),
        payload.get("nonsealed", {}),
        payload.get("metrics", {}).get("nonsealed", {}) if isinstance(payload.get("metrics"), dict) else {},
        payload,
    ]
    result: dict[str, Any] = {
        "available": True,
        "path": source.sanity_path,
        "native_head_only_completion": None,
        "source_overlap_accuracy": None,
        "full_minus_native_head_only": None,
        "val_command_slot_accuracy": None,
    }
    if "native_head_only_completion_mean" in metrics:
        result["native_head_only_completion"] = _as_float(metrics.get("native_head_only_completion_mean"))
        result["full_minus_native_head_only"] = _as_float(metrics.get("full_minus_native_head_only_mean"))
        result["val_command_slot_accuracy"] = _as_float(metrics.get("val_command_slot_accuracy_mean"))
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        result["native_head_only_completion"] = result["native_head_only_completion"] or _as_float(
            bucket.get("native_head_only_completion")
        )
        result["source_overlap_accuracy"] = result["source_overlap_accuracy"] or _as_float(
            bucket.get("source_overlap_val_accuracy")
        ) or _as_float(bucket.get("source_overlap_validation_accuracy"))
        result["full_minus_native_head_only"] = result["full_minus_native_head_only"] or _as_float(
            bucket.get("full_minus_native_head_only")
        )
        result["val_command_slot_accuracy"] = result["val_command_slot_accuracy"] or _as_float(
            bucket.get("val_command_slot_accuracy")
        )
    return result


def _load_rows(root: Path, source: EvidenceSource) -> list[dict[str, Any]]:
    path = root / source.path
    if source.source_kind == "baseline_table":
        return _rows_from_baseline_table(path)
    if source.source_kind == "eval_family":
        return _rows_from_phase2l_eval_family(path)
    if source.source_kind == "aggregate_summary":
        return _rows_from_phase2p_aggregate(path)
    raise ValueError(f"Unknown evidence source kind: {source.source_kind}")


def build_baseline_zero_audit(root: str | Path = ".") -> dict[str, Any]:
    repo_root = Path(root)
    phases: list[dict[str, Any]] = []
    unexplained_zero_count = 0
    claim_blockers: list[str] = []

    for source in DEFAULT_EVIDENCE_SOURCES:
        raw_rows = _load_rows(repo_root, source)
        sanity = _sanity_metrics(repo_root, source)
        rows = []
        zero_rows = []
        for row in raw_rows:
            completion = row.get("completion")
            zero_category = _zero_category(row.get("policy", ""), completion)
            if zero_category == "suspicious_zero_requires_redesign":
                unexplained_zero_count += 1
            if zero_category:
                zero_rows.append(row.get("policy", "unknown"))
            rows.append(
                {
                    **row,
                    "zero_category": zero_category,
                    "zero_claim_use": _zero_claim_use(zero_category, sanity),
                }
            )
        native_zero = any(
            row.get("completion") == 0.0 and "native-head-only" in _normal_policy(row.get("policy", ""))
            for row in raw_rows
        )
        no_nsi_zero = any(
            row.get("completion") == 0.0 and "no-nsi" in _normal_policy(row.get("policy", ""))
            for row in raw_rows
        )
        full_zero = any(
            row.get("completion") == 0.0 and "full package" in _normal_policy(row.get("policy", ""))
            for row in raw_rows
        )
        sanity_native_nonzero = (sanity.get("native_head_only_completion") or 0.0) > 0.0
        phase_claim_blockers = []
        if native_zero and no_nsi_zero and not sanity_native_nonzero:
            phase_claim_blockers.append("add_graded_sanity_subset_before_using_zero_controls_as_delta_evidence")
        if full_zero:
            phase_claim_blockers.append("do_not_use_this_phase_as_positive_sealed_transfer_evidence")
        if any(row.get("zero_category") == "suspicious_zero_requires_redesign" for row in rows):
            phase_claim_blockers.append("redesign_or_explain_suspicious_zero_control")
        claim_blockers.extend(f"{source.phase}:{blocker}" for blocker in phase_claim_blockers)
        phases.append(
            {
                "phase": source.phase,
                "evaluation_scope": source.evaluation_scope,
                "source": source.path,
                "sanity_evidence": sanity,
                "zero_rows": zero_rows,
                "native_and_no_nsi_both_zero": native_zero and no_nsi_zero,
                "full_zero": full_zero,
                "claim_blockers": phase_claim_blockers,
                "rows": rows,
            }
        )

    return {
        "audit_family": "paper_b_baseline_zero_interpretability",
        "passed": unexplained_zero_count == 0,
        "unexplained_zero_count": unexplained_zero_count,
        "claim_blockers": sorted(set(claim_blockers)),
        "interpretation": {
            "zero_controls_are_not_automatic_stronger_evidence": True,
            "zeros_require_capability_or_evaluability_classification": True,
            "broad_generalization_claims_blocked": True,
            "sealed_v3_used_for_training_or_tuning": False,
        },
        "zero_categories": sorted(ZERO_CATEGORIES),
        "phases": phases,
    }


def _zero_claim_use(zero_category: str | None, sanity: dict[str, Any]) -> str:
    if zero_category is None:
        return "nonzero_measured_control"
    if zero_category == "suspicious_zero_requires_redesign":
        return "blocks_claim_use_until_redesigned"
    if zero_category == "not_evaluable_for_control":
        return "do_not_use_as_performance_delta"
    if zero_category == "valid_zero_failure":
        return "usable_as_failure_evidence_not_architecture_proof"
    if zero_category == "expected_zero_due_to_missing_capability":
        if (sanity.get("native_head_only_completion") or 0.0) > 0.0:
            return "bounded_mechanism_stress_only_with_sanity_subset"
        return "requires_graded_sanity_subset_before_delta_claim"
    return "blocks_claim_use_until_reviewed"


def write_baseline_zero_audit(
    *,
    output_json: str | Path,
    output_md: str | Path | None = None,
    root: str | Path = ".",
) -> dict[str, Any]:
    report = build_baseline_zero_audit(root=root)
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if output_md is not None:
        md_path = Path(output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_to_markdown(report), encoding="utf-8")
    return report


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper B baseline-zero interpretability audit",
        "",
        f"- Passed: `{str(report['passed']).lower()}`",
        f"- Unexplained zero count: `{report['unexplained_zero_count']}`",
        f"- Broad generalization claims blocked: `{str(report['interpretation']['broad_generalization_claims_blocked']).lower()}`",
        "",
        "This audit treats zero-valued controls as evidence that must be classified, not as automatic proof of a stronger architecture claim.",
        "",
        "Required zero categories: "
        + ", ".join(f"`{category}`" for category in report["zero_categories"])
        + ".",
        "",
        "## Phase summary",
        "",
        "| Phase | Scope | Native and no-NSI both zero | Full zero | Claim blockers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for phase in report["phases"]:
        blockers = ", ".join(phase["claim_blockers"]) if phase["claim_blockers"] else "none"
        lines.append(
            "| {phase} | {scope} | {native_no_nsi} | {full_zero} | {blockers} |".format(
                phase=phase["phase"],
                scope=phase["evaluation_scope"],
                native_no_nsi=str(phase["native_and_no_nsi_both_zero"]).lower(),
                full_zero=str(phase["full_zero"]).lower(),
                blockers=blockers,
            )
        )
    lines.extend(["", "## Row classifications", ""])
    for phase in report["phases"]:
        lines.extend(
            [
                f"### {phase['phase']} - {phase['evaluation_scope']}",
                "",
                "| Policy | Completion | Zero category | Claim use |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for row in phase["rows"]:
            category = row["zero_category"] or "not_zero"
            completion = "NA" if row["completion"] is None else f"{row['completion']:.6g}"
            lines.append(f"| {row['policy']} | {completion} | {category} | {row['zero_claim_use']} |")
        lines.append("")
    return "\n".join(lines)
