from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


STRUCTURAL_CHANGE_RE = re.compile(
    r"^\s*(from\s+\S+\s+import\s+|import\s+\S+|def\s+\w+|class\s+\w+|"
    r"return\s+.+\.\w+\(|.+\.\w+\(.+\)|.+\[[^\]]+\]|raise\s+\w+|with\s+.+:|"
    r"for\s+.+\s+in\s+.+:|if\s+.+\s+(and|or)\s+.+:)"
)


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


def _changed_lines(patch_text: str, prefix: str) -> list[str]:
    marker = "+++" if prefix == "+" else "---"
    return [
        line[1:]
        for line in patch_text.splitlines()
        if line.startswith(prefix) and not line.startswith(marker)
    ]


def classify_patch_text(patch_text: str) -> dict[str, Any]:
    changed_files = [
        line[6:].strip()
        for line in patch_text.splitlines()
        if line.startswith("+++ b/")
    ]
    added = _changed_lines(patch_text, "+")
    removed = _changed_lines(patch_text, "-")
    structural_added = [line for line in added if STRUCTURAL_CHANGE_RE.search(line)]
    structural_removed = [line for line in removed if STRUCTURAL_CHANGE_RE.search(line)]
    single_line_literal_like = (
        len(changed_files) == 1
        and len(added) == 1
        and len(removed) == 1
        and not structural_added
        and not structural_removed
    )
    structural = len(changed_files) >= 2 or bool(structural_added or structural_removed)
    return {
        "changed_file_count": len(changed_files),
        "changed_files": changed_files,
        "added_line_count": len(added),
        "removed_line_count": len(removed),
        "multi_file": len(changed_files) >= 2,
        "single_line_literal_like": single_line_literal_like,
        "structural_nonliteral_candidate": structural and not single_line_literal_like,
        "structural_added_examples": structural_added[:3],
        "structural_removed_examples": structural_removed[:3],
    }


def audit_phase2z_public_nonliteral_trace_gap(
    *,
    dataset_root: str | Path,
    split_jsonl: list[str | Path],
    min_rows: int = 64,
    min_structural_nonliteral_rows: int = 8,
    min_multifile_rows: int = 1,
) -> dict[str, Any]:
    root = Path(dataset_root)
    rows: list[dict[str, Any]] = []
    for path in split_jsonl:
        rows.extend(_read_jsonl(path))

    patch_rows = 0
    missing_patch_rows = 0
    public_rows = 0
    structural_rows = 0
    multifile_rows = 0
    literal_like_rows = 0
    source_kinds = Counter()
    split_counts = Counter()
    examples: list[dict[str, Any]] = []
    for row in rows:
        source_kind = str(row.get("source_kind") or "")
        source_kinds[source_kind] += 1
        split_counts[str(row.get("split") or "")] += 1
        public_rows += int(source_kind == "public_repo")
        paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
        patch_rel = str(paths.get("patch_diff") or "")
        patch_path = root / patch_rel
        if not patch_rel or not patch_path.exists():
            missing_patch_rows += 1
            continue
        patch_rows += 1
        stats = classify_patch_text(patch_path.read_text(encoding="utf-8", errors="replace"))
        structural_rows += int(bool(stats["structural_nonliteral_candidate"]))
        multifile_rows += int(bool(stats["multi_file"]))
        literal_like_rows += int(bool(stats["single_line_literal_like"]))
        if stats["structural_nonliteral_candidate"] and len(examples) < 8:
            examples.append(
                {
                    "trace_id": row.get("trace_id"),
                    "split": row.get("split"),
                    "repo": row.get("repo_id") or row.get("repo_url_or_origin"),
                    "patch_stats": stats,
                    "patch_path": str(patch_path),
                }
            )

    checks = {
        "row_minimum_met": len(rows) >= min_rows,
        "all_rows_public_repo": bool(rows) and public_rows == len(rows),
        "patch_artifacts_present": bool(rows) and missing_patch_rows == 0,
        "structural_nonliteral_minimum_met": structural_rows >= min_structural_nonliteral_rows,
        "multifile_minimum_met": multifile_rows >= min_multifile_rows,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2z_public_nonliteral_trace_gap_audit",
        "passed": passed,
        "claim_ready_public_nonliteral_patch_generation": passed,
        "claim_boundary": (
            "public_repo_origin_nonliteral_patch_trace_ready"
            if passed
            else "public_repo_origin_nonliteral_patch_gap"
        ),
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "patch_rows": patch_rows,
            "missing_patch_rows": missing_patch_rows,
            "public_rows": public_rows,
            "structural_nonliteral_rows": structural_rows,
            "multifile_rows": multifile_rows,
            "single_line_literal_like_rows": literal_like_rows,
            "source_kind_distribution": dict(sorted(source_kinds.items())),
            "split_distribution": dict(sorted(split_counts.items())),
        },
        "structural_nonliteral_examples": examples,
        "blocked_actions": []
        if passed
        else [
            "do_not_claim_public_nonliteral_patch_generation",
            "do_not_train_phase2z_claim_bearing_nonliteral_without_public_structural_rows",
            "collect_repo_origin_disjoint_public_nonliteral_multifile_traces",
        ],
        "inputs": {
            "dataset_root": str(root),
            "split_jsonl": [str(Path(path)) for path in split_jsonl],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether public repair traces contain enough structural nonliteral patches."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-jsonl", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--min-structural-nonliteral-rows", type=int, default=8)
    parser.add_argument("--min-multifile-rows", type=int, default=1)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2z_public_nonliteral_trace_gap(
        dataset_root=args.dataset_root,
        split_jsonl=args.split_jsonl,
        min_rows=args.min_rows,
        min_structural_nonliteral_rows=args.min_structural_nonliteral_rows,
        min_multifile_rows=args.min_multifile_rows,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
