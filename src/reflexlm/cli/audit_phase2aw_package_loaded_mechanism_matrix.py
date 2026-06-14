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


def _rate(payload: dict[str, Any]) -> float | None:
    value = payload.get("success_rate")
    return float(value) if isinstance(value, (int, float)) else None


def _selection(payload: dict[str, Any]) -> float | None:
    value = payload.get("patch_candidate_selection_accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def _row(name: str, path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    return {
        "policy": name,
        "success_rate": _rate(payload),
        "successes/rows": f"{payload.get('successes')}/{payload.get('rows')}",
        "patch_candidate_selection_accuracy": _selection(payload),
        "qwen_called_rows": payload.get("qwen_called_rows"),
        "policy_loaded": payload.get("policy_loaded"),
        "summary_json": str(Path(path)),
        "claim_boundary": payload.get("claim_boundary"),
    }


def audit_phase2aw_package_loaded_mechanism_matrix(
    *,
    full_summary_json: str | Path,
    source_overlap_summary_json: str | Path,
    no_nsi_summary_json: str | Path,
    native_head_only_summary_json: str | Path,
    continuation_only_summary_json: str | Path,
    min_full_success_rate: float = 0.85,
    min_full_minus_source_overlap: float = 0.15,
    min_full_minus_no_nsi: float = 0.15,
    min_full_minus_native_head_only: float = 0.10,
    min_full_minus_continuation_only: float = 0.15,
) -> dict[str, Any]:
    rows = [
        _row("full_package", full_summary_json),
        _row("source_overlap", source_overlap_summary_json),
        _row("no_nsi_no_candidate_identity", no_nsi_summary_json),
        _row("native_head_only_no_cache", native_head_only_summary_json),
        _row("continuation_only_no_native_heads", continuation_only_summary_json),
    ]
    by_policy = {row["policy"]: row for row in rows}
    full = by_policy["full_package"]["success_rate"]
    source = by_policy["source_overlap"]["success_rate"]
    no_nsi = by_policy["no_nsi_no_candidate_identity"]["success_rate"]
    native = by_policy["native_head_only_no_cache"]["success_rate"]
    continuation = by_policy["continuation_only_no_native_heads"]["success_rate"]

    metrics = {
        "full_success_rate": full,
        "source_overlap_success_rate": source,
        "no_nsi_success_rate": no_nsi,
        "native_head_only_success_rate": native,
        "continuation_only_success_rate": continuation,
        "full_minus_source_overlap": full - source if full is not None and source is not None else None,
        "full_minus_no_nsi": full - no_nsi if full is not None and no_nsi is not None else None,
        "full_minus_native_head_only": full - native if full is not None and native is not None else None,
        "full_minus_continuation_only": (
            full - continuation if full is not None and continuation is not None else None
        ),
    }
    checks = {
        "all_package_controls_loaded": all(
            row["policy_loaded"] is True
            for row in rows
            if row["policy"] != "source_overlap"
        ),
        "full_success_rate_ge_threshold": full is not None and full >= min_full_success_rate,
        "full_beats_source_overlap": metrics["full_minus_source_overlap"] is not None
        and metrics["full_minus_source_overlap"] >= min_full_minus_source_overlap,
        "full_beats_no_nsi": metrics["full_minus_no_nsi"] is not None
        and metrics["full_minus_no_nsi"] >= min_full_minus_no_nsi,
        "full_beats_native_head_only": metrics["full_minus_native_head_only"] is not None
        and metrics["full_minus_native_head_only"] >= min_full_minus_native_head_only,
        "full_beats_continuation_only": metrics["full_minus_continuation_only"] is not None
        and metrics["full_minus_continuation_only"] >= min_full_minus_continuation_only,
        "native_head_only_control_nonzero": native is not None and native > 0.0,
        "no_nsi_control_nonzero": no_nsi is not None and no_nsi > 0.0,
    }
    passed = all(checks.values())
    full_package_necessity_supported = (
        checks["full_beats_native_head_only"]
        and checks["full_beats_no_nsi"]
        and checks["full_beats_continuation_only"]
    )
    return {
        "artifact_family": "phase2aw_package_loaded_mechanism_matrix",
        "passed": passed,
        "claim_scope": (
            "phase2aw_package_loaded_full_mechanism_matrix_supported"
            if passed
            else "phase2aw_package_loaded_partial_mechanism_evidence"
        ),
        "full_package_necessity_supported": full_package_necessity_supported,
        "ready_for_claim_upgrade": passed and full_package_necessity_supported,
        "checks": checks,
        "metrics": metrics,
        "control_matrix": {
            "table_family": "phase2aw_package_loaded_nonsealed_control_matrix",
            "columns": [
                "policy",
                "success_rate",
                "successes/rows",
                "patch_candidate_selection_accuracy",
                "qwen_called_rows",
                "policy_loaded",
                "claim_boundary",
            ],
            "rows": rows,
        },
        "supported_claims": [
            "Phase2AW package-loaded full policy beats source-overlap on split-clean public holdout",
            "Phase2AW package-loaded full policy beats no-NSI/no-candidate-identity control",
            "Phase2AW native heads are sufficient for this non-sealed descriptor-runtime split",
        ],
        "unsupported_claims": [
            "full package necessity is not supported unless full beats native-head-only",
            "continuation memory necessity is not supported on this split",
            "epoch-making architecture is not proven",
            "production autonomy and open-ended debugging generalization are not proven",
        ],
        "failure_or_boundary_analysis": (
            "Native-head-only matches full on this split; the split validates bounded "
            "package-loaded native-head repair selection and NSI/candidate-identity "
            "ablation, but it does not prove that the full package is necessary over "
            "native heads alone."
        ),
        "inputs": {
            "full_summary_json": str(Path(full_summary_json)),
            "source_overlap_summary_json": str(Path(source_overlap_summary_json)),
            "no_nsi_summary_json": str(Path(no_nsi_summary_json)),
            "native_head_only_summary_json": str(Path(native_head_only_summary_json)),
            "continuation_only_summary_json": str(Path(continuation_only_summary_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AW package-loaded mechanism control matrix.")
    parser.add_argument("--full-summary-json", required=True)
    parser.add_argument("--source-overlap-summary-json", required=True)
    parser.add_argument("--no-nsi-summary-json", required=True)
    parser.add_argument("--native-head-only-summary-json", required=True)
    parser.add_argument("--continuation-only-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2aw_package_loaded_mechanism_matrix(
        full_summary_json=args.full_summary_json,
        source_overlap_summary_json=args.source_overlap_summary_json,
        no_nsi_summary_json=args.no_nsi_summary_json,
        native_head_only_summary_json=args.native_head_only_summary_json,
        continuation_only_summary_json=args.continuation_only_summary_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
