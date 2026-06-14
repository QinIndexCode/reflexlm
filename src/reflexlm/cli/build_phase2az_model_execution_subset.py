from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _repo(row: dict[str, Any]) -> str:
    manifest = row.get("source_task_manifest")
    if isinstance(manifest, dict):
        return str(manifest.get("repo_origin") or "")
    return str(row.get("repo_origin") or "")


def _repo_id_from_origin(origin: str) -> str:
    parsed = urlparse(origin)
    path = parsed.path if parsed.scheme else origin
    return path.strip("/").removesuffix(".git").replace("/", "_").replace("-", "_").lower()


def _clone_present(repo_origin: str, clone_root: Path | None) -> bool:
    if clone_root is None:
        return False
    return (clone_root / _repo_id_from_origin(repo_origin)).is_dir()


def _pair_id(row: dict[str, Any]) -> str:
    manifest = row.get("source_task_manifest")
    if isinstance(manifest, dict):
        return str(manifest.get("pair_id") or "")
    return str(row.get("phase2ax_pair_id") or "")


def build_phase2az_model_execution_subset(
    *,
    head_jsonl: str | Path,
    tasks_jsonl: str | Path,
    output_head_jsonl: str | Path,
    output_tasks_jsonl: str | Path,
    report_json: str | Path,
    max_repos: int = 3,
    pairs_per_repo: int = 1,
    min_repos: int = 2,
    clone_root: str | Path | None = None,
    require_clone_present: bool = False,
) -> dict[str, Any]:
    head_rows = _read_jsonl(head_jsonl)
    task_rows = _read_jsonl(tasks_jsonl)
    task_by_id = {str(row.get("task_id") or ""): row for row in task_rows}
    repo_pair_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    clones = Path(clone_root) if clone_root is not None else None
    clone_filtered_repos: list[str] = []
    for row in head_rows:
        repo = _repo(row)
        pair_id = _pair_id(row)
        if not repo or not pair_id:
            continue
        if require_clone_present and not _clone_present(repo, clones):
            if repo not in clone_filtered_repos:
                clone_filtered_repos.append(repo)
            continue
        repo_pair_rows.setdefault(repo, {}).setdefault(pair_id, []).append(row)

    selected_head_rows: list[dict[str, Any]] = []
    selected_repos: list[str] = []
    for repo in sorted(repo_pair_rows):
        if len(selected_repos) >= max_repos:
            break
        complete_pairs = [
            sorted(rows, key=lambda item: str(item.get("episode_id") or ""))
            for _, rows in sorted(repo_pair_rows[repo].items())
            if {int(item.get("command_slot", -1)) for item in rows} >= {0, 1}
        ]
        if not complete_pairs:
            continue
        selected_repos.append(repo)
        for pair_rows in complete_pairs[:pairs_per_repo]:
            selected_head_rows.extend(pair_rows)

    selected_task_ids = [str(row.get("episode_id") or "") for row in selected_head_rows]
    selected_task_rows = [
        task_by_id[task_id] for task_id in selected_task_ids if task_id in task_by_id
    ]
    slot_counts: dict[str, int] = {}
    for row in selected_head_rows:
        slot = str(row.get("command_slot"))
        slot_counts[slot] = slot_counts.get(slot, 0) + 1
    checks = {
        "min_repos_met": len(selected_repos) >= min_repos,
        "head_rows_present": bool(selected_head_rows),
        "tasks_matched_all_head_rows": len(selected_task_rows) == len(selected_head_rows),
        "slot0_present": slot_counts.get("0", 0) > 0,
        "slot1_present": slot_counts.get("1", 0) > 0,
        "clone_requirement_satisfied": (not require_clone_present)
        or (clones is not None and bool(selected_repos)),
    }
    report = {
        "artifact_family": "phase2az_model_execution_subset",
        "passed": all(checks.values()),
        "checks": checks,
        "head_rows": len(selected_head_rows),
        "task_rows": len(selected_task_rows),
        "repo_count": len(selected_repos),
        "repos": selected_repos,
        "clone_filtered_repos": sorted(clone_filtered_repos),
        "slot_counts": slot_counts,
        "outputs": {
            "head_jsonl": str(Path(output_head_jsonl)),
            "tasks_jsonl": str(Path(output_tasks_jsonl)),
        },
        "inputs": {
            "head_jsonl": str(Path(head_jsonl)),
            "tasks_jsonl": str(Path(tasks_jsonl)),
            "clone_root": str(Path(clone_root)) if clone_root is not None else None,
            "require_clone_present": require_clone_present,
        },
        "claim_boundary": "phase2az_subset_only_not_execution_or_claim_evidence",
    }
    _write_jsonl(output_head_jsonl, selected_head_rows)
    _write_jsonl(output_tasks_jsonl, selected_task_rows)
    _write_json(report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a repo-diverse Phase2AZ model-predicted execution subset."
    )
    parser.add_argument("--head-jsonl", required=True)
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--output-head-jsonl", required=True)
    parser.add_argument("--output-tasks-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--max-repos", type=int, default=3)
    parser.add_argument("--pairs-per-repo", type=int, default=1)
    parser.add_argument("--min-repos", type=int, default=2)
    parser.add_argument("--clone-root")
    parser.add_argument("--require-clone-present", action="store_true")
    args = parser.parse_args()
    report = build_phase2az_model_execution_subset(
        head_jsonl=args.head_jsonl,
        tasks_jsonl=args.tasks_jsonl,
        output_head_jsonl=args.output_head_jsonl,
        output_tasks_jsonl=args.output_tasks_jsonl,
        report_json=args.report_json,
        max_repos=args.max_repos,
        pairs_per_repo=args.pairs_per_repo,
        min_repos=args.min_repos,
        clone_root=args.clone_root,
        require_clone_present=args.require_clone_present,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
