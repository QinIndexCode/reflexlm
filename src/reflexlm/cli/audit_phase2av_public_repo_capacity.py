from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.collect_phase2m_public_repo_traces import _clone_or_get_repo, _run_git
from reflexlm.cli.collect_phase2z_public_structural_repair_traces import (
    _discover_structural_targets,
    _order_structural_targets,
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def audit_phase2av_public_repo_capacity(
    *,
    repo_specs_json: str | Path,
    clone_root: str | Path,
    no_clone: bool = False,
    target_selection_policy: str = "behavioral_diverse_descriptor",
    min_non_import_targets_per_repo: int = 2,
    min_total_non_import_targets: int = 48,
) -> dict[str, Any]:
    specs = _read_json(repo_specs_json)
    if not isinstance(specs, list):
        raise ValueError("repo_specs_json must contain a list of repo specs")
    clone_root_path = Path(clone_root)
    reports: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    split_totals: dict[str, Counter[str]] = {}
    clone_failures: list[dict[str, str]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        repo_id = str(spec.get("repo_id") or spec.get("repo_url") or spec.get("url") or "repo")
        split = str(spec.get("split") or "train")
        try:
            repo = _clone_or_get_repo(spec, clone_root=clone_root_path, no_clone=no_clone)
            commit_hash = _run_git(repo, ["rev-parse", "HEAD"])
            targets = _order_structural_targets(
                _discover_structural_targets(
                    repo,
                    target_selection_policy=target_selection_policy,
                ),
                target_selection_policy=target_selection_policy,
            )
        except Exception as exc:
            clone_failures.append({"repo_id": repo_id, "reason": type(exc).__name__})
            continue
        counts = Counter(target.repair_mode for target in targets)
        totals.update(counts)
        split_totals.setdefault(split, Counter()).update(counts)
        non_import = sum(
            count for mode, count in counts.items() if mode != "behavioral_import_restoration"
        )
        reports.append(
            {
                "repo_id": repo_id,
                "split": split,
                "repo_path": str(repo),
                "repo_url_or_origin": str(spec.get("repo_url") or spec.get("url") or ""),
                "commit_hash": commit_hash,
                "eligible_targets": len(targets),
                "eligible_repair_mode_counts": dict(sorted(counts.items())),
                "non_import_target_count": non_import,
                "meets_non_import_threshold": non_import >= min_non_import_targets_per_repo,
            }
        )
    total_non_import = sum(
        count for mode, count in totals.items() if mode != "behavioral_import_restoration"
    )
    passed = total_non_import >= min_total_non_import_targets and any(
        report["meets_non_import_threshold"] for report in reports
    )
    return {
        "artifact_family": "phase2av_public_repo_capacity_audit",
        "passed": passed,
        "ready_for_phase2av_candidate_collection": passed,
        "claim_boundary": (
            "Public repo capacity audit only. It does not authorize training, "
            "packaging, sealed evaluation, freeform patch generation, production "
            "autonomy, or epoch-making architecture claims."
        ),
        "target_selection_policy": target_selection_policy,
        "repo_count": len(reports),
        "clone_failure_count": len(clone_failures),
        "clone_failures": clone_failures[:20],
        "eligible_repair_mode_totals": dict(sorted(totals.items())),
        "eligible_repair_mode_totals_by_split": {
            split: dict(sorted(counter.items())) for split, counter in sorted(split_totals.items())
        },
        "non_import_target_total": total_non_import,
        "recommended_repos": [
            report for report in reports if report["meets_non_import_threshold"]
        ],
        "repo_reports": reports,
        "thresholds": {
            "min_non_import_targets_per_repo": min_non_import_targets_per_repo,
            "min_total_non_import_targets": min_total_non_import_targets,
        },
        "blocking_reasons": []
        if passed
        else [
            "public_repo_capacity_below_non_import_descriptor_threshold",
            "collect_more_public_repo_specs_before_full_scale_phase2av",
        ],
        "unsupported_claims": [
            "phase2av_full_training_ready",
            "phase2av_package_ready",
            "sealed_cross_model_transfer",
            "freeform_patch_generation",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "inputs": {
            "repo_specs_json": str(Path(repo_specs_json)),
            "clone_root": str(clone_root_path),
            "no_clone": no_clone,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AV public repo descriptor capacity.")
    parser.add_argument("--repo-specs-json", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--target-selection-policy", default="behavioral_diverse_descriptor")
    parser.add_argument("--min-non-import-targets-per-repo", type=int, default=2)
    parser.add_argument("--min-total-non-import-targets", type=int, default=48)
    parser.add_argument("--no-clone", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2av_public_repo_capacity(
        repo_specs_json=args.repo_specs_json,
        clone_root=args.clone_root,
        no_clone=args.no_clone,
        target_selection_policy=args.target_selection_policy,
        min_non_import_targets_per_repo=args.min_non_import_targets_per_repo,
        min_total_non_import_targets=args.min_total_non_import_targets,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
