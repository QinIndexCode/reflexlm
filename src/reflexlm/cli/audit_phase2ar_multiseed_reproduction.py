from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _seed_pair(report: dict[str, Any]) -> tuple[int | None, int | None]:
    contract = report.get("metrics", {}).get("training_contract", {})
    if not isinstance(contract, dict):
        return None, None
    primary = contract.get("primary_seed")
    reproduction = contract.get("reproduction_seed")
    return (
        int(primary) if isinstance(primary, int) else None,
        int(reproduction) if isinstance(reproduction, int) else None,
    )


def audit_phase2ar_multiseed_reproduction(
    *,
    reproduction_audit_jsons: list[str | Path],
    min_unique_seeds: int = 3,
    min_success_rate: float = 1.0,
) -> dict[str, Any]:
    reports = [_read_json(path) for path in reproduction_audit_jsons]
    seeds: set[int] = set()
    success_rates: list[float] = []
    row_counts: list[int] = []
    contract_mismatches: list[dict[str, Any]] = []
    failed_reports: list[str] = []
    for path, report in zip(reproduction_audit_jsons, reports):
        metrics = report.get("metrics", {})
        primary_seed, reproduction_seed = _seed_pair(report)
        for seed in (primary_seed, reproduction_seed):
            if seed is not None:
                seeds.add(seed)
        success_rates.append(float(metrics.get("reproduction_success_rate") or 0.0))
        row_counts.append(int(metrics.get("row_count") or 0))
        contract = metrics.get("training_contract", {})
        if not isinstance(contract, dict) or not (
            contract.get("same_contract_except_seed_and_names")
            and contract.get("split_hashes_match")
            and contract.get("seed_changed")
        ):
            contract_mismatches.append({"path": str(path), "training_contract": contract})
        if report.get("passed") is not True:
            failed_reports.append(str(path))

    checks = {
        "all_reproduction_audits_passed": not failed_reports,
        "unique_seed_minimum_met": len(seeds) >= min_unique_seeds,
        "all_success_rates_met": all(rate >= min_success_rate for rate in success_rates),
        "row_counts_consistent": len(set(row_counts)) == 1 and bool(row_counts),
        "training_contracts_match_except_seed": not contract_mismatches,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2ar_multiseed_reproduction_audit",
        "passed": passed,
        "claim_boundary": (
            "Phase2AR multi-seed audit supports same-model, same-split seed "
            "reproduction only. It does not support cross-model transfer, "
            "sealed transfer, production autonomy, or open-ended repair."
        ),
        "checks": checks,
        "metrics": {
            "unique_seeds": sorted(seeds),
            "unique_seed_count": len(seeds),
            "success_rates": success_rates,
            "row_counts": row_counts,
            "failed_reports": failed_reports,
            "contract_mismatches": contract_mismatches,
        },
        "supported_claims": ["phase2ar_three_seed_same_model_reproduction_supported"]
        if passed
        else [],
        "unsupported_claims": [
            "cross_model_reproduction",
            "sealed_cross_model_transfer",
            "epoch_making_architecture",
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
        ],
        "blocked_actions": [
            "do_not_claim_cross_model_reproduction_from_same_model_seed_runs",
            "do_not_claim_epoch_making_architecture_from_phase2ar_alone",
        ],
        "inputs": {
            "reproduction_audit_jsons": [str(Path(path)) for path in reproduction_audit_jsons],
            "min_unique_seeds": min_unique_seeds,
            "min_success_rate": min_success_rate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AR multi-seed reproduction evidence.")
    parser.add_argument("--reproduction-audit-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-unique-seeds", type=int, default=3)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    args = parser.parse_args()
    report = audit_phase2ar_multiseed_reproduction(
        reproduction_audit_jsons=args.reproduction_audit_json,
        min_unique_seeds=args.min_unique_seeds,
        min_success_rate=args.min_success_rate,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
