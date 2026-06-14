from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TEXT_MEMBERSHIP_RE = re.compile(r"assert\s+(?:'[^']*'|\"[^\"]*\")\s+in\s+text")
AST_ATTRIBUTE_RE = re.compile(r"node\.attr\s*==\s*(?:'[^']*'|\"[^\"]*\")")
FORBIDDEN_MARKER_RE = re.compile(
    r"(?i)(sealed[_-]?v?3|candidate[_-]?\d+|slot[_-]?\d+|gold[_-]?(label|slot|hint))"
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


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


def _patch_kinds_for_row(dataset_root: Path, row: dict[str, Any]) -> set[str]:
    artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    generated_rel = str(artifact_paths.get("generated_test") or "")
    if not generated_rel:
        return set()
    generated_path = dataset_root / generated_rel
    if not generated_path.exists():
        return set()
    source = generated_path.read_text(encoding="utf-8", errors="replace")
    kinds: set[str] = set()
    if TEXT_MEMBERSHIP_RE.search(source):
        kinds.add("text_membership")
    if AST_ATTRIBUTE_RE.search(source):
        kinds.add("ast_attribute_restoration")
    return kinds


def _row_has_forbidden_marker(dataset_root: Path, row: dict[str, Any]) -> bool:
    payload = json.dumps(row, ensure_ascii=False)
    if FORBIDDEN_MARKER_RE.search(payload):
        return True
    artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    for key in ("generated_test", "patch_diff"):
        rel = str(artifact_paths.get(key) or "")
        path = dataset_root / rel
        if rel and path.exists() and FORBIDDEN_MARKER_RE.search(
            path.read_text(encoding="utf-8", errors="replace")
        ):
            return True
    return False


def audit_phase2ar_diverse_symbolic_patch_benchmark(
    *,
    dataset_root: str | Path,
    min_rows_per_split: int = 8,
    required_patch_kinds: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root)
    required = set(required_patch_kinds or ["text_membership", "ast_attribute_restoration"])
    manifest = _read_json(root / "manifest.json")
    split_rows = {
        split: _read_jsonl(root / f"{split}.raw.jsonl")
        for split in ("train", "val", "holdout")
    }
    repos_by_split = {
        split: {str(row.get("repo_id") or "") for row in rows}
        for split, rows in split_rows.items()
    }
    patch_kinds_by_split: dict[str, Counter[str]] = {}
    forbidden_markers = 0
    sealed_flags = []
    for split, rows in split_rows.items():
        counter: Counter[str] = Counter()
        for row in rows:
            for kind in _patch_kinds_for_row(root, row):
                counter[kind] += 1
            if _row_has_forbidden_marker(root, row):
                forbidden_markers += 1
            normalization = row.get("normalization")
            if isinstance(normalization, dict):
                sealed_flags.append(normalization.get("sealed_feedback_absent") is True)
            else:
                sealed_flags.append(False)
        patch_kinds_by_split[split] = counter

    repo_disjoint = (
        repos_by_split["train"].isdisjoint(repos_by_split["val"])
        and repos_by_split["train"].isdisjoint(repos_by_split["holdout"])
        and repos_by_split["val"].isdisjoint(repos_by_split["holdout"])
    )
    checks = {
        "manifest_present": (root / "manifest.json").exists(),
        "stratified_target_selection": manifest.get("target_selection_policy")
        == "stratified_repair_mode",
        "rows_per_split_minimum_met": all(
            len(rows) >= min_rows_per_split for rows in split_rows.values()
        ),
        "repo_origin_disjoint": repo_disjoint,
        "sealed_feedback_absent": bool(sealed_flags) and all(sealed_flags),
        "forbidden_candidate_gold_slot_markers_absent": forbidden_markers == 0,
        "holdout_required_patch_kinds_present": required.issubset(
            set(patch_kinds_by_split["holdout"])
        ),
        "val_required_patch_kinds_present": required.issubset(set(patch_kinds_by_split["val"])),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2ar_diverse_symbolic_patch_benchmark_audit",
        "passed": passed,
        "claim_boundary": (
            "nonsealed_public_repo_diverse_bounded_symbolic_patch_benchmark_only"
        ),
        "checks": checks,
        "metrics": {
            "rows_by_split": {split: len(rows) for split, rows in split_rows.items()},
            "repos_by_split": {split: sorted(repos) for split, repos in repos_by_split.items()},
            "patch_kinds_by_split": {
                split: dict(sorted(counter.items()))
                for split, counter in patch_kinds_by_split.items()
            },
            "forbidden_marker_rows": forbidden_markers,
        },
        "supported_claims": [
            "phase2ar_nonsealed_diverse_symbolic_patch_benchmark_ready"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "inputs": {"dataset_root": str(root)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AR diverse bounded symbolic patch benchmark."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows-per-split", type=int, default=8)
    args = parser.parse_args()
    report = audit_phase2ar_diverse_symbolic_patch_benchmark(
        dataset_root=args.dataset_root,
        min_rows_per_split=args.min_rows_per_split,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
