from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MARKER_RE = re.compile(
    r"\b(?:candidate_[0-9]+|gold(?:en)?_?(?:slot|label|answer)|sealed[_ -]?v?[0-9]*|"
    r"expected_patch_candidate_slot)\b",
    re.IGNORECASE,
)


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


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _token(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "/").strip().lower()


def _tokens_from_value(value: Any) -> set[str]:
    if isinstance(value, list):
        return {_token(item) for item in value if _token(item)}
    if isinstance(value, dict):
        return {_token(item) for item in value.values() if _token(item)}
    token = _token(value)
    return {token} if token else set()


def _row_text_for_marker_scan(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(row.get("current_visible_text") or ""),
            json.dumps(row.get("runtime_visible_evidence") or {}, ensure_ascii=False, sort_keys=True),
            json.dumps(row.get("repair_candidates") or [], ensure_ascii=False, sort_keys=True),
        ]
    )


def _expected_slot(row: dict[str, Any]) -> int | None:
    expected = str(row.get("expected_repair_action") or "")
    if not expected:
        return None
    for index, candidate in enumerate(_list(row.get("repair_candidates"))):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("repair_action") or "") == expected:
            return index
    return None


def _runtime_probe_tokens(row: dict[str, Any]) -> set[str]:
    runtime = _dict(row.get("runtime_visible_evidence"))
    tokens: set[str] = set()
    tokens.update(_tokens_from_value(runtime.get("verification_probe_tokens")))
    tokens.update(_tokens_from_value(runtime.get("structural_probe_hashes")))
    expected_literal_hash = _token(runtime.get("expected_literal_hash"))
    if expected_literal_hash:
        tokens.add(f"literal:{expected_literal_hash}")
    location = _dict(runtime.get("target_location"))
    path = _token(location.get("path"))
    line = _token(location.get("line"))
    col = _token(location.get("col"))
    if path and line and col:
        tokens.add(f"loc:{path}:{line}:{col}")
    return {token for token in tokens if token}


def _candidate_probe_tokens(candidate: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    tokens.update(_tokens_from_value(candidate.get("verification_probe_tokens")))
    tokens.update(_tokens_from_value(candidate.get("verification_probe_token")))
    structural_hash = _token(candidate.get("structural_probe_hash"))
    if structural_hash:
        tokens.add(structural_hash)
    literal_hash = _token(candidate.get("target_literal_hash"))
    if literal_hash:
        tokens.add(f"literal:{literal_hash}")
    path = _token(candidate.get("edit_scope"))
    line = _token(candidate.get("target_line"))
    col = _token(candidate.get("target_col"))
    if path and line and col:
        tokens.add(f"loc:{path}:{line}:{col}")
    return {token for token in tokens if token}


def _probe_prediction(row: dict[str, Any]) -> tuple[int | None, list[float], list[list[str]]]:
    runtime_tokens = _runtime_probe_tokens(row)
    scores: list[float] = []
    overlaps: list[list[str]] = []
    for candidate in _list(row.get("repair_candidates")):
        if not isinstance(candidate, dict):
            scores.append(0.0)
            overlaps.append([])
            continue
        overlap = sorted(_candidate_probe_tokens(candidate) & runtime_tokens)
        overlaps.append(overlap)
        scores.append(float(len(overlap)))
    if not scores:
        return None, scores, overlaps
    best = max(scores)
    if best <= 0.0 or scores.count(best) != 1:
        return None, scores, overlaps
    return scores.index(best), scores, overlaps


def _has_sealed_reference(row: dict[str, Any]) -> bool:
    text = json.dumps(row, ensure_ascii=False, sort_keys=True).lower()
    return "sealed_v3_used\": true" in text or "sealed_feedback_used\": true" in text


def _audit_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    expected = _expected_slot(row)
    prediction, scores, overlaps = _probe_prediction(row)
    marker_leak = bool(MARKER_RE.search(_row_text_for_marker_scan(row)))
    sealed_ref = _has_sealed_reference(row)
    unresolved = prediction is None
    correct = prediction == expected if prediction is not None and expected is not None else False
    return {
        "row_index": index,
        "trace_id": row.get("trace_id") or row.get("example_id"),
        "repo_id": row.get("repo_id"),
        "expected_slot": expected,
        "probe_prediction": prediction,
        "probe_scores": scores,
        "probe_overlaps": overlaps,
        "marker_leak": marker_leak,
        "sealed_reference": sealed_ref,
        "unresolved_probe": unresolved,
        "probe_correct": correct,
    }


def audit_phase2ag_verifiable_candidate_sidecar(
    *,
    jsonl: str | Path,
    output_json: str | Path,
    split_name: str = "eval",
) -> dict[str, Any]:
    rows = _read_jsonl(jsonl)
    row_reports = [_audit_row(row, index) for index, row in enumerate(rows)]
    unresolved = [row for row in row_reports if row["unresolved_probe"]]
    incorrect = [row for row in row_reports if not row["probe_correct"]]
    marker_leaks = [row for row in row_reports if row["marker_leak"]]
    sealed_refs = [row for row in row_reports if row["sealed_reference"]]
    repos = Counter(str(row["repo_id"]) for row in unresolved)
    passed = bool(rows) and not unresolved and not incorrect and not marker_leaks and not sealed_refs
    report = {
        "artifact_family": "phase2ag_verifiable_candidate_sidecar_audit",
        "split_name": split_name,
        "passed": passed,
        "claim_bearing_training_allowed": passed,
        "metrics": {
            "row_count": len(rows),
            "unresolved_probe_rows": len(unresolved),
            "incorrect_probe_rows": len(incorrect),
            "marker_leak_rows": len(marker_leaks),
            "sealed_reference_rows": len(sealed_refs),
            "probe_accuracy": (
                (len(rows) - len(incorrect)) / len(rows)
                if rows
                else 0.0
            ),
        },
        "failure_distribution": {
            "unresolved_by_repo": dict(sorted(repos.items())),
        },
        "diagnosis": {
            "primary_issue": (
                "Every row has a unique non-label candidate probe that matches runtime-visible evidence."
                if passed
                else "One or more rows lack a unique, correct, non-label candidate probe or contain forbidden markers."
            ),
            "boundary": (
                "This audit only validates candidate verifiability. It does not prove learned-head advantage, "
                "sealed transfer, production autonomy, or open-ended debugging."
            ),
        },
        "blocked_actions": []
        if passed
        else [
            "do_not_train_claim_bearing_phase2ag_adapter",
            "do_not_package_phase2ag",
            "do_not_run_sealed_phase2ag",
            "do_not_claim_verifiable_candidate_sidecar_mechanism",
        ],
        "unresolved_rows": unresolved,
        "incorrect_rows": incorrect,
        "marker_leak_rows": marker_leaks,
        "sealed_reference_rows": sealed_refs,
        "inputs": {"jsonl": str(Path(jsonl))},
    }
    _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AG verifiable candidate sidecar rows.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--split-name", default="eval")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = audit_phase2ag_verifiable_candidate_sidecar(
        jsonl=args.jsonl,
        split_name=args.split_name,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
