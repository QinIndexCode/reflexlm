from __future__ import annotations

import argparse
import json
import re
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


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key) is True) / len(rows)


def _execution_summary(path: str | Path, package_json: str | Path | None = None) -> dict[str, Any]:
    rows = _read_jsonl(path)
    package = _read_json(package_json) if package_json else {}
    first = rows[0] if rows else {}
    policy_label = first.get("native_policy_label") or package.get("policy_label")
    seed_match = re.search(r"seed(\d+)", str(policy_label or path))
    return {
        "path": str(Path(path)),
        "package_json": str(Path(package_json)) if package_json else None,
        "policy_label": policy_label,
        "base_model_name": package.get("base_model_name"),
        "seed": int(seed_match.group(1)) if seed_match else None,
        "row_count": len(rows),
        "success_rate": _rate(rows, "success"),
        "selection_accuracy": _rate(rows, "patch_candidate_selected_correctly"),
        "policy_loaded": all(row.get("policy_loaded") is True for row in rows),
        "claim_boundary": first.get("claim_boundary"),
    }


def build_phase2aa_runtime_replication_report(
    *,
    data_health_json: str | Path,
    full_delta_gate_jsons: list[str | Path],
    full_execution_jsonls: list[str | Path],
    full_package_jsons: list[str | Path] | None = None,
    control_execution_jsonl: str | Path,
    no_nsi_execution_jsonl: str | Path | None = None,
    retry_control_execution_jsonl: str | Path | None = None,
    symbolic_runtime_delta_gate_json: str | Path | None = None,
    min_rows: int = 256,
    min_model_count: int = 2,
    min_seed_count_for_any_model: int = 3,
    min_full_success_rate: float = 0.85,
    min_full_minus_control: float = 0.15,
    min_full_minus_no_nsi: float = 0.15,
    require_full_minus_retry_control: bool = False,
    min_full_minus_retry_control: float = 0.15,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    full_delta_gates = [_read_json(path) for path in full_delta_gate_jsons]
    package_jsons = full_package_jsons or []
    full_runs = [
        _execution_summary(
            path,
            package_jsons[index] if index < len(package_jsons) else None,
        )
        for index, path in enumerate(full_execution_jsonls)
    ]
    control = _execution_summary(control_execution_jsonl)
    no_nsi = _execution_summary(no_nsi_execution_jsonl) if no_nsi_execution_jsonl else None
    retry_control = (
        _execution_summary(retry_control_execution_jsonl)
        if retry_control_execution_jsonl
        else None
    )
    symbolic_delta = (
        _read_json(symbolic_runtime_delta_gate_json)
        if symbolic_runtime_delta_gate_json
        else None
    )

    full_success_rates = [float(run["success_rate"]) for run in full_runs]
    full_selection_rates = [float(run["selection_accuracy"]) for run in full_runs]
    model_names = {
        str(run.get("base_model_name") or run.get("policy_label") or "")
        for run in full_runs
        if run.get("base_model_name") or run.get("policy_label")
    }
    seeds_by_model: dict[str, list[int]] = {}
    for run in full_runs:
        model_key = str(run.get("base_model_name") or run.get("policy_label") or "")
        seed = run.get("seed")
        if model_key and isinstance(seed, int):
            seeds_by_model.setdefault(model_key, [])
            if seed not in seeds_by_model[model_key]:
                seeds_by_model[model_key].append(seed)
    seeds_by_model = {
        model: sorted(seeds) for model, seeds in sorted(seeds_by_model.items())
    }
    max_seed_count_for_model = max((len(seeds) for seeds in seeds_by_model.values()), default=0)
    min_full_success = min(full_success_rates or [0.0])
    min_full_selection = min(full_selection_rates or [0.0])
    control_success = float(control["success_rate"])
    no_nsi_success = float(_dict(no_nsi).get("success_rate") or 0.0) if no_nsi else None
    retry_control_success = (
        float(_dict(retry_control).get("success_rate") or 0.0)
        if retry_control
        else None
    )
    min_full_minus_control_value = min_full_success - control_success
    min_full_minus_no_nsi_value = (
        min_full_success - no_nsi_success if no_nsi_success is not None else None
    )
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "runtime_artifacts_resolved": _dict(data_health.get("checks")).get(
            "required_runtime_artifacts_available"
        )
        is True,
        "all_full_delta_gates_passed": bool(full_delta_gates)
        and all(gate.get("passed") is True for gate in full_delta_gates),
        "all_full_runs_policy_loaded": bool(full_runs)
        and all(run.get("policy_loaded") is True for run in full_runs),
        "all_full_runs_row_minimum_met": bool(full_runs)
        and all(int(run.get("row_count") or 0) >= min_rows for run in full_runs),
        "model_count_minimum_met": len(model_names) >= min_model_count,
        "independent_seed_count_minimum_met": max_seed_count_for_model
        >= min_seed_count_for_any_model,
        "min_full_success_rate_met": min_full_success >= min_full_success_rate,
        "min_full_selection_accuracy_met": min_full_selection >= min_full_success_rate,
        "full_minus_control_delta_met": min_full_minus_control_value >= min_full_minus_control,
        "no_nsi_control_present": no_nsi is not None,
        "full_minus_no_nsi_delta_met": (
            min_full_minus_no_nsi_value is not None
            and min_full_minus_no_nsi_value >= min_full_minus_no_nsi
        ),
        "symbolic_runtime_tie_not_misclaimed": (
            symbolic_delta is None or symbolic_delta.get("passed") is not True
        ),
        "retry_control_not_claimed_as_beaten": (
            retry_control is None or retry_control_success >= min_full_success
        ),
        "full_minus_retry_control_delta_met_if_required": (
            not require_full_minus_retry_control
            or (
                retry_control_success is not None
                and min_full_success - retry_control_success >= min_full_minus_retry_control
            )
        ),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2aa_runtime_replication_report",
        "passed": passed,
        "claim_boundary": (
            "This report supports bounded patch-candidate selection replication "
            "across loaded NativeNervousPolicyPackage variants on the same "
            "non-sealed public-repo holdout. It remains a bounded candidate "
            "selection claim and does not establish freeform patch generation, "
            "sealed transfer, production autonomy, open-ended debugging, or an "
            "epoch-making architecture."
        ),
        "checks": checks,
        "supported_claims": [
            "bounded_patch_candidate_selection_runtime_delta_replicated_across_models"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
            *(
                ["verification_retry_or_identity_heuristic_control_beaten"]
                if retry_control is not None
                else []
            ),
        ],
        "next_required_evidence": [
            *(
                ["independent_seed_packages_for_same_runtime_delta"]
                if max_seed_count_for_model < min_seed_count_for_any_model
                else []
            ),
            "repo_origin_disjoint_external_holdout_not_reused_in_package_selection",
            "nonsealed_runtime_task_where_symbolic_parser_only_control_cannot_solve",
            *(
                ["task_family_where_identity_first_retry_control_does_not_tie_full"]
                if retry_control is not None and retry_control_success >= min_full_success
                else []
            ),
        ],
        "metrics": {
            "model_count": len(model_names),
            "models": sorted(model_names),
            "seeds_by_model": seeds_by_model,
            "max_seed_count_for_model": max_seed_count_for_model,
            "full_runs": full_runs,
            "control_run": control,
            "no_nsi_run": no_nsi,
            "retry_control_run": retry_control,
            "min_full_success_rate": min_full_success,
            "min_full_selection_accuracy": min_full_selection,
            "control_success_rate": control_success,
            "full_minus_control_success_rate_min": min_full_minus_control_value,
            "no_nsi_success_rate": no_nsi_success,
            "full_minus_no_nsi_success_rate_min": min_full_minus_no_nsi_value,
            "retry_control_success_rate": retry_control_success,
            "full_minus_retry_control_success_rate_min": (
                min_full_success - retry_control_success
                if retry_control_success is not None
                else None
            ),
            "require_full_minus_retry_control": require_full_minus_retry_control,
            "symbolic_runtime_delta_metrics": _dict(_dict(symbolic_delta or {}).get("metrics")),
        },
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "full_delta_gate_jsons": [str(Path(path)) for path in full_delta_gate_jsons],
            "full_execution_jsonls": [str(Path(path)) for path in full_execution_jsonls],
            "full_package_jsons": [str(Path(path)) for path in package_jsons],
            "control_execution_jsonl": str(Path(control_execution_jsonl)),
            "no_nsi_execution_jsonl": str(Path(no_nsi_execution_jsonl))
            if no_nsi_execution_jsonl
            else None,
            "retry_control_execution_jsonl": str(Path(retry_control_execution_jsonl))
            if retry_control_execution_jsonl
            else None,
            "symbolic_runtime_delta_gate_json": str(Path(symbolic_runtime_delta_gate_json))
            if symbolic_runtime_delta_gate_json
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AA runtime replication evidence report."
    )
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--full-delta-gate-json", action="append", required=True)
    parser.add_argument("--full-execution-jsonl", action="append", required=True)
    parser.add_argument("--full-package-json", action="append")
    parser.add_argument("--control-execution-jsonl", required=True)
    parser.add_argument("--no-nsi-execution-jsonl")
    parser.add_argument("--retry-control-execution-jsonl")
    parser.add_argument("--symbolic-runtime-delta-gate-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=256)
    parser.add_argument("--min-model-count", type=int, default=2)
    parser.add_argument("--min-seed-count-for-any-model", type=int, default=3)
    parser.add_argument("--min-full-success-rate", type=float, default=0.85)
    parser.add_argument("--min-full-minus-control", type=float, default=0.15)
    parser.add_argument("--min-full-minus-no-nsi", type=float, default=0.15)
    parser.add_argument("--require-full-minus-retry-control", action="store_true")
    parser.add_argument("--min-full-minus-retry-control", type=float, default=0.15)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2aa_runtime_replication_report(
        data_health_json=args.data_health_json,
        full_delta_gate_jsons=args.full_delta_gate_json,
        full_execution_jsonls=args.full_execution_jsonl,
        full_package_jsons=args.full_package_json,
        control_execution_jsonl=args.control_execution_jsonl,
        no_nsi_execution_jsonl=args.no_nsi_execution_jsonl,
        retry_control_execution_jsonl=args.retry_control_execution_jsonl,
        symbolic_runtime_delta_gate_json=args.symbolic_runtime_delta_gate_json,
        min_rows=args.min_rows,
        min_model_count=args.min_model_count,
        min_seed_count_for_any_model=args.min_seed_count_for_any_model,
        min_full_success_rate=args.min_full_success_rate,
        min_full_minus_control=args.min_full_minus_control,
        min_full_minus_no_nsi=args.min_full_minus_no_nsi,
        require_full_minus_retry_control=args.require_full_minus_retry_control,
        min_full_minus_retry_control=args.min_full_minus_retry_control,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
