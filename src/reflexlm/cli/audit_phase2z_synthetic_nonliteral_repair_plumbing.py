from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def audit_phase2z_synthetic_nonliteral_repair_plumbing(
    *,
    execution_results_jsonl: str | Path,
    min_rows: int = 8,
    min_success_rate: float = 0.75,
    require_multifile: bool = True,
) -> dict[str, Any]:
    rows = _read_jsonl(execution_results_jsonl)
    success_count = sum(1 for row in rows if row.get("success") is True)
    success_rate = success_count / len(rows) if rows else 0.0
    nonclaim_rows = sum(
        1
        for row in rows
        if row.get("claim_bearing_execution_evidence") is False
        and row.get("recorded_patch_artifact_used") is True
        and row.get("oracle_trace_used") is True
        and row.get("sealed_feedback_used") is False
    )
    nonliteral_rows = sum(
        1
        for row in rows
        if isinstance(row.get("patch_stats"), dict)
        and row["patch_stats"].get("nonliteral_structure_present") is True
    )
    multifile_rows = sum(
        1
        for row in rows
        if isinstance(row.get("patch_stats"), dict)
        and row["patch_stats"].get("multi_file") is True
    )
    checks = {
        "rows_minimum_met": len(rows) >= min_rows,
        "success_rate_minimum_met": success_rate >= min_success_rate,
        "all_rows_non_claim_plumbing": bool(rows) and nonclaim_rows == len(rows),
        "nonliteral_structure_present": nonliteral_rows > 0,
        "multifile_patch_present": (multifile_rows > 0) if require_multifile else True,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2z_synthetic_nonliteral_repair_plumbing_audit",
        "passed": passed,
        "claim_bearing_execution_evidence": False,
        "claim_boundary": "synthetic_nonliteral_plumbing_only_not_model_patch_generation",
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "success_count": success_count,
            "success_rate": success_rate,
            "nonclaim_rows": nonclaim_rows,
            "nonliteral_rows": nonliteral_rows,
            "multifile_rows": multifile_rows,
        },
        "blocked_actions": [
            "do_not_use_phase2z_plumbing_as_open_ended_debugging_claim",
            "do_not_use_recorded_patch_artifact_as_model_generated_nonliteral_patch",
            "collect_public_repo_origin_nonliteral_patch_traces_before_claim_upgrade",
        ],
        "inputs": {"execution_results_jsonl": str(Path(execution_results_jsonl))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2Z synthetic-safe nonliteral repair plumbing results."
    )
    parser.add_argument("--execution-results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--min-success-rate", type=float, default=0.75)
    parser.add_argument("--allow-no-multifile", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2z_synthetic_nonliteral_repair_plumbing(
        execution_results_jsonl=args.execution_results_jsonl,
        min_rows=args.min_rows,
        min_success_rate=args.min_success_rate,
        require_multifile=not args.allow_no_multifile,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
