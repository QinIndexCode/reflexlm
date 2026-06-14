from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def build_phase2at_evidence_sufficiency_report(
    *,
    data_health_json: str | Path,
    pretrain_readiness_json: str | Path,
    multiseed_smoke_json: str | Path,
    cross_model_smoke_json: str | Path | None = None,
    package_schema_gate_json: str | Path | None = None,
    package_runtime_smoke_audit_json: str | Path | None = None,
    package_runtime_delta_gate_json: str | Path | None = None,
    min_holdout_rows: int = 32,
    larger_holdout_row_threshold: int = 96,
    min_unique_seeds: int = 3,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    readiness = _read_json(pretrain_readiness_json)
    multiseed = _read_json(multiseed_smoke_json)
    cross_model = _read_json(cross_model_smoke_json) if cross_model_smoke_json else None
    package_schema_gate = (
        _read_json(package_schema_gate_json) if package_schema_gate_json else None
    )
    package_runtime_smoke_audit = (
        _read_json(package_runtime_smoke_audit_json)
        if package_runtime_smoke_audit_json
        else None
    )
    package_runtime_delta_gate = (
        _read_json(package_runtime_delta_gate_json)
        if package_runtime_delta_gate_json
        else None
    )
    split_counts = _dict(_dict(data_health.get("metrics")).get("split_counts"))
    holdout_rows = int(split_counts.get("holdout") or 0)
    larger_holdout_met = holdout_rows >= larger_holdout_row_threshold
    cross_model_passed = cross_model is not None and cross_model.get("passed") is True
    runtime_smoke_passed = (
        package_runtime_smoke_audit is not None
        and package_runtime_smoke_audit.get("passed") is True
    )
    runtime_row_count = int(
        _dict(_dict(package_runtime_smoke_audit or {}).get("metrics")).get("row_count")
        or 0
    )
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_readiness_passed": readiness.get("passed") is True,
        "ready_for_training": readiness.get("ready_for_training") is True,
        "multiseed_smoke_passed": multiseed.get("passed") is True,
        "holdout_row_minimum_met": holdout_rows >= min_holdout_rows,
        "unique_seed_minimum_met": int(
            _dict(multiseed.get("metrics")).get("unique_seed_count") or 0
        )
        >= min_unique_seeds,
        "unsupported_claims_retained": all(
            claim in set(multiseed.get("unsupported_claims") or [])
            for claim in [
                "learned_freeform_patch_generation",
                "sealed_cross_model_transfer",
                "production_autonomy",
                "open_ended_debugging_generalization",
                "epoch_making_architecture",
            ]
        ),
    }
    cross_model_checks: dict[str, bool] = {}
    if cross_model is not None:
        cross_model_checks = {
            "cross_model_smoke_passed": cross_model.get("passed") is True,
            "cross_model_has_two_or_more_models": int(
                _dict(cross_model.get("metrics")).get("model_count") or 0
            )
            >= 2,
            "cross_model_boundary_retained": "sealed_cross_model_transfer"
            in set(cross_model.get("unsupported_claims") or []),
        }
        checks.update(cross_model_checks)
    if package_schema_gate is not None:
        checks.update(
            {
                "package_schema_gate_passed": package_schema_gate.get("passed") is True,
                "package_schema_boundary_retained": all(
                    claim in set(package_schema_gate.get("unsupported_claims") or [])
                    for claim in [
                        "freeform_patch_generation",
                        "production_autonomy",
                        "open_ended_debugging_generalization",
                        "epoch_making_architecture",
                    ]
                ),
            }
        )
    if package_runtime_smoke_audit is not None:
        runtime_unsupported_actions = set(
            package_runtime_smoke_audit.get("blocked_actions") or []
        )
        checks.update(
            {
                "package_runtime_smoke_audit_passed": package_runtime_smoke_audit.get(
                    "passed"
                )
                is True,
                "package_runtime_smoke_uses_bounded_symbolic_boundary": package_runtime_smoke_audit.get(
                    "claim_boundary"
                )
                == "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
                "package_runtime_smoke_blocks_freeform_and_openended": {
                    "do_not_claim_freeform_model_generated_patch_repair",
                    "do_not_claim_open_ended_debugging_generalization",
                }.issubset(runtime_unsupported_actions),
            }
        )
    if package_runtime_delta_gate is not None:
        checks.update(
            {
                "package_runtime_delta_gate_passed": package_runtime_delta_gate.get(
                    "passed"
                )
                is True,
                "package_runtime_delta_boundary_retained": all(
                    claim in set(package_runtime_delta_gate.get("unsupported_claims") or [])
                    for claim in [
                        "learned_freeform_patch_generation",
                        "sealed_cross_model_transfer",
                        "production_autonomy",
                        "open_ended_debugging_generalization",
                        "epoch_making_architecture",
                    ]
                ),
            }
        )
    passed = all(checks.values())
    next_required_evidence = [
        "claim_bearing_package_release_only_after_full_nonsealed_delta_gate",
    ]
    if runtime_row_count < 24:
        next_required_evidence.insert(
            0, "claim_bearing_runtime_evaluation_beyond_single_row_symbolic_smoke"
        )
    if not larger_holdout_met:
        next_required_evidence.insert(
            0, "larger_nonsealed_public_repo_origin_disjoint_descriptor_holdout"
        )
    if not cross_model_passed:
        next_required_evidence.insert(1, "cross_model_descriptor_smoke_after_same_model_seed_gate")
    if not runtime_smoke_passed:
        next_required_evidence.append("bounded_package_runtime_smoke_after_schema_gate")
    if (
        package_runtime_delta_gate is None
        or package_runtime_delta_gate.get("passed") is not True
    ):
        next_required_evidence.append(
            "nonsealed_task_where_loaded_package_beats_no_policy_control"
        )
    if larger_holdout_met and cross_model_passed:
        claim_boundary = (
            "Phase2AT currently supports bounded patch-candidate descriptor-head "
            "learning on a non-sealed repo-origin-disjoint descriptor split, with "
            "same-model three-seed stability and initial same-split cross-model "
            "smoke. Package/runtime evidence is limited to bounded symbolic "
            "structural smoke when provided. This is not learned freeform patch "
            "generation, sealed transfer, production autonomy, open-ended "
            "debugging, or an epoch-making architecture."
        )
    else:
        claim_boundary = (
            "Phase2AT currently supports only bounded patch-candidate descriptor "
            "learning on an early non-sealed smoke. It is evidence for "
            "descriptor-head plumbing and seed stability, not learned freeform "
            "patch generation, sealed transfer, production autonomy, open-ended "
            "debugging, or an epoch-making architecture."
        )
    return {
        "artifact_family": "phase2at_evidence_sufficiency_report",
        "passed": passed,
        "claim_boundary": claim_boundary,
        "checks": checks,
        "supported_claims": [
            "phase2at_nonsealed_data_ready_for_bounded_descriptor_training",
            "phase2at_same_model_three_seed_descriptor_smoke_stable",
            *(
                ["phase2at_initial_same_split_cross_model_descriptor_smoke_supported"]
                if cross_model is not None and cross_model.get("passed") is True
                else []
            ),
            *(
                ["phase2at_package_schema_ready_for_learned_bounded_candidate_eval"]
                if package_schema_gate is not None
                and package_schema_gate.get("passed") is True
                else []
            ),
            *(
                ["phase2at_package_runtime_bounded_symbolic_smoke_passed"]
                if package_runtime_smoke_audit is not None
                and package_runtime_smoke_audit.get("passed") is True
                else []
            ),
            *(
                ["phase2at_package_runtime_delta_supported"]
                if package_runtime_delta_gate is not None
                and package_runtime_delta_gate.get("passed") is True
                else []
            ),
        ]
        if passed
        else [],
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "blocked_actions": [
            "do_not_treat_phase2at_schema_ready_package_as_claim_bearing_generation_without_full_nonsealed_delta",
            "do_not_use_phase2at_smoke_as_sealed_transfer_evidence",
            "do_not_claim_epoch_making_architecture_from_phase2at",
        ],
        "next_required_evidence": next_required_evidence,
        "metrics": {
            "split_counts": split_counts,
            "data_best_non_full_baseline_accuracy": _dict(data_health.get("metrics")).get(
                "best_non_full_baseline_accuracy"
            ),
            "multiseed_metric_summary": _dict(_dict(multiseed.get("metrics")).get("metric_summary")),
            "unique_seeds": _dict(multiseed.get("metrics")).get("unique_seeds"),
            "unique_seed_count": _dict(multiseed.get("metrics")).get("unique_seed_count"),
            "larger_holdout_row_threshold": larger_holdout_row_threshold,
            "larger_holdout_row_threshold_met": larger_holdout_met,
            "evidence_scale": (
                "repo_origin_disjoint_rows64_or_larger"
                if larger_holdout_met
                else "early_small_smoke"
            ),
            "cross_model_min_metrics": _dict(_dict(cross_model or {}).get("metrics")).get(
                "min_metrics"
            )
            if cross_model is not None
            else None,
            "cross_model_models": _dict(_dict(cross_model or {}).get("metrics")).get("models")
            if cross_model is not None
            else None,
            "package_schema_gate_passed": (
                package_schema_gate.get("passed") is True
                if package_schema_gate is not None
                else None
            ),
            "package_schema_gate_metrics": _dict(
                _dict(package_schema_gate or {}).get("metrics")
            )
            if package_schema_gate is not None
            else None,
            "package_runtime_smoke_audit_passed": (
                package_runtime_smoke_audit.get("passed") is True
                if package_runtime_smoke_audit is not None
                else None
            ),
            "package_runtime_smoke_metrics": _dict(
                _dict(package_runtime_smoke_audit or {}).get("metrics")
            )
            if package_runtime_smoke_audit is not None
            else None,
            "package_runtime_delta_gate_passed": (
                package_runtime_delta_gate.get("passed") is True
                if package_runtime_delta_gate is not None
                else None
            ),
            "package_runtime_delta_metrics": _dict(
                _dict(package_runtime_delta_gate or {}).get("metrics")
            )
            if package_runtime_delta_gate is not None
            else None,
        },
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "pretrain_readiness_json": str(Path(pretrain_readiness_json)),
            "multiseed_smoke_json": str(Path(multiseed_smoke_json)),
            "cross_model_smoke_json": str(Path(cross_model_smoke_json))
            if cross_model_smoke_json
            else None,
            "package_schema_gate_json": str(Path(package_schema_gate_json))
            if package_schema_gate_json
            else None,
            "package_runtime_smoke_audit_json": str(Path(package_runtime_smoke_audit_json))
            if package_runtime_smoke_audit_json
            else None,
            "package_runtime_delta_gate_json": str(Path(package_runtime_delta_gate_json))
            if package_runtime_delta_gate_json
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AT evidence sufficiency report.")
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--pretrain-readiness-json", required=True)
    parser.add_argument("--multiseed-smoke-json", required=True)
    parser.add_argument("--cross-model-smoke-json")
    parser.add_argument("--package-schema-gate-json")
    parser.add_argument("--package-runtime-smoke-audit-json")
    parser.add_argument("--package-runtime-delta-gate-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-holdout-rows", type=int, default=32)
    parser.add_argument("--larger-holdout-row-threshold", type=int, default=96)
    parser.add_argument("--min-unique-seeds", type=int, default=3)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=args.data_health_json,
        pretrain_readiness_json=args.pretrain_readiness_json,
        multiseed_smoke_json=args.multiseed_smoke_json,
        cross_model_smoke_json=args.cross_model_smoke_json,
        package_schema_gate_json=args.package_schema_gate_json,
        package_runtime_smoke_audit_json=args.package_runtime_smoke_audit_json,
        package_runtime_delta_gate_json=args.package_runtime_delta_gate_json,
        min_holdout_rows=args.min_holdout_rows,
        larger_holdout_row_threshold=args.larger_holdout_row_threshold,
        min_unique_seeds=args.min_unique_seeds,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
