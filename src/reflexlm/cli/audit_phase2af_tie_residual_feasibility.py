from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import command_candidate_source_overlap_rows


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


def _num(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _slot(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _repo_id(row: dict[str, Any]) -> str:
    trace = _dict(row.get("source_trace"))
    return str(trace.get("repo_id") or row.get("repo_id") or "<unknown>")


def _source_overlap_unique_best(
    state_prompt: str,
    candidates: list[str],
) -> tuple[int | None, float]:
    rows = command_candidate_source_overlap_rows(state_prompt, candidates)
    scores = [row[1] for row in rows[: len(candidates)]]
    if not scores:
        return None, 0.0
    best = max(scores)
    if best <= 0.0 or scores.count(best) != 1:
        return None, 0.0
    second = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0.0
    return scores.index(best), float(best - second)


def _candidate_identity_margin(row: dict[str, Any]) -> float:
    return _num(_dict(row.get("nsi_reference")).get("command_identity_margin"))


def _row_feasibility(row: dict[str, Any], index: int) -> dict[str, Any]:
    candidates = [str(item) for item in row.get("candidate_commands") or [] if item]
    label = _slot(row.get("command_slot"))
    identity_margin = _candidate_identity_margin(row)
    source_pred, source_margin = _source_overlap_unique_best(str(row.get("state_prompt") or ""), candidates)
    identity_tie = identity_margin <= 0.0
    source_unique = source_pred is not None
    source_correct = source_pred == label if source_unique and label is not None else False
    unresolved = identity_tie and not source_unique
    return {
        "row_index": index,
        "example_id": row.get("example_id") or row.get("episode_id"),
        "repo_id": _repo_id(row),
        "command_slot": label,
        "candidate_count": len(candidates),
        "identity_margin": identity_margin,
        "identity_tie": identity_tie,
        "source_overlap_unique_best_slot": source_pred,
        "source_overlap_margin": source_margin,
        "source_overlap_unique_best_correct": source_correct,
        "unresolved_identity_tie": unresolved,
    }


def audit_phase2af_tie_residual_feasibility(
    *,
    head_jsonl: str | Path,
    output_json: str | Path,
    split_name: str = "eval",
) -> dict[str, Any]:
    rows = _read_jsonl(head_jsonl)
    row_reports = [_row_feasibility(row, index) for index, row in enumerate(rows)]
    identity_ties = [row for row in row_reports if row["identity_tie"]]
    unresolved = [row for row in row_reports if row["unresolved_identity_tie"]]
    source_disambiguated = [
        row for row in identity_ties if row["source_overlap_unique_best_slot"] is not None
    ]
    source_disambiguated_correct = [
        row for row in source_disambiguated if row["source_overlap_unique_best_correct"]
    ]
    repos = Counter(row["repo_id"] for row in unresolved)
    slots = Counter(str(row["command_slot"]) for row in unresolved)
    passed = len(unresolved) == 0
    report = {
        "artifact_family": "phase2af_tie_residual_feasibility_audit",
        "split_name": split_name,
        "passed": passed,
        "claim_bearing_training_allowed": passed,
        "metrics": {
            "row_count": len(rows),
            "identity_tie_rows": len(identity_ties),
            "source_disambiguated_identity_tie_rows": len(source_disambiguated),
            "source_disambiguated_identity_tie_correct_rows": len(source_disambiguated_correct),
            "unresolved_identity_tie_rows": len(unresolved),
            "unresolved_identity_tie_rate": len(unresolved) / len(rows) if rows else 0.0,
        },
        "failure_distribution": {
            "unresolved_by_repo": dict(sorted(repos.items())),
            "unresolved_by_slot": dict(sorted(slots.items())),
        },
        "diagnosis": {
            "primary_issue": (
                "Some rows have zero command-identity margin and no independent source-overlap unique best cue; "
                "these rows are not claim-bearing for the current sidecar feature set."
            )
            if unresolved
            else "Every command-identity tie row has an independent visible cue that can be audited.",
            "boundary": (
                "Passing this audit is not sufficient for a claim; it only prevents unresolvable tie rows "
                "from being mistaken for learned mechanism evidence."
            ),
        },
        "blocked_actions": []
        if passed
        else [
            "do_not_train_claim_bearing_phase2af_adapter_on_this_split",
            "do_not_package_phase2af",
            "do_not_run_sealed_phase2af",
            "do_not_claim_hardened_structural_sidecar_mechanism",
        ],
        "unresolved_rows": unresolved,
        "inputs": {"head_jsonl": str(Path(head_jsonl))},
    }
    _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AF identity-tie residual feasibility.")
    parser.add_argument("--head-jsonl", required=True)
    parser.add_argument("--split-name", default="eval")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = audit_phase2af_tie_residual_feasibility(
        head_jsonl=args.head_jsonl,
        split_name=args.split_name,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
