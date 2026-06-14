from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2af_hardened_structural_sidecar_split import (
    _read_candidate_sources,
    _shortcut_key,
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2af_shortcut_bucket_gap_report(
    *,
    jsonls: list[str | Path],
    output_json: str | Path | None = None,
    require_tie_residual_feasible_rows: bool = True,
    min_identity_wrong_source_correct_rows: int = 1,
) -> dict[str, Any]:
    rows = _read_candidate_sources(
        jsonls,
        require_tie_residual_feasible=require_tie_residual_feasible_rows,
    )
    buckets = Counter(_shortcut_key(row) for row in rows)
    bucket_counts = {
        f"source_{source}_identity_{identity}": int(count)
        for (source, identity), count in sorted(buckets.items())
    }
    identity_wrong_source_correct = buckets.get((1, 0), 0)
    checks = {
        "candidate_rows_present": len(rows) > 0,
        "tie_residual_feasible_filter_enabled": require_tie_residual_feasible_rows,
        "identity_wrong_source_correct_bucket_present": identity_wrong_source_correct
        >= min_identity_wrong_source_correct_rows,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2af_shortcut_bucket_gap_report",
        "passed": passed,
        "claim_bearing_training_allowed": passed,
        "checks": checks,
        "metrics": {
            "candidate_rows": len(rows),
            "bucket_counts": bucket_counts,
            "identity_wrong_source_correct_rows": int(identity_wrong_source_correct),
            "min_identity_wrong_source_correct_rows": min_identity_wrong_source_correct_rows,
        },
        "diagnosis": {
            "primary_issue": (
                "source data pool lacks rows where runtime identity is wrong while source evidence is correct; "
                "training on this pool cannot prove a model mechanism beyond the runtime-identity shortcut"
            )
            if not checks["identity_wrong_source_correct_bucket_present"]
            else "source data pool contains identity-failure feasible rows for residual mechanism pressure",
            "non_goal": (
                "Do not manufacture this bucket from sealed failures or gold slot markers; collect or construct "
                "non-sealed runtime-visible evidence that genuinely challenges the identity shortcut."
            ),
        },
        "blocked_actions": []
        if passed
        else [
            "do_not_train_claim_bearing_phase2af_adapter",
            "do_not_package_phase2af",
            "do_not_claim_model_beats_runtime_identity",
        ],
        "allowed_next_action": (
            "build_phase2af_or_phase2ah_claim_bearing_split"
            if passed
            else "collect_or_construct_nonsealed_identity_wrong_source_correct_rows"
        ),
        "inputs": {
            "jsonls": [str(Path(path)) for path in jsonls],
            "require_tie_residual_feasible_rows": require_tie_residual_feasible_rows,
        },
    }
    if output_json is not None:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether Phase2AF sources can challenge runtime-identity shortcuts."
    )
    parser.add_argument("--jsonl", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--allow-unresolved-tie-rows", action="store_true")
    parser.add_argument("--min-identity-wrong-source-correct-rows", type=int, default=1)
    args = parser.parse_args()
    report = build_phase2af_shortcut_bucket_gap_report(
        jsonls=args.jsonl,
        output_json=args.output_json,
        require_tie_residual_feasible_rows=not args.allow_unresolved_tie_rows,
        min_identity_wrong_source_correct_rows=args.min_identity_wrong_source_correct_rows,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
