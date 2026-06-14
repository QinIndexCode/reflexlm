from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.run_phase2cd_public_repo_post_verification_control import (
    _execution_rows,
    _split_repo_disjoint,
)
from reflexlm.cli.run_phase2ce_single_policy_live_patch_verify_stop_loop import (
    _train_verification_matcher,
)
from reflexlm.llm.native_nervous_package import PACKAGE_MANIFEST_NAME


def build_phase2cg_unified_verification_package(
    *,
    base_package_path: str | Path,
    historical_execution_jsonl: str | Path,
    tasks_jsonl: str | Path,
    cortex_model_path: str | Path,
    cortex_device: str,
    cortex_dtype: str,
    output_package_dir: str | Path,
    output_report_json: str | Path,
    recency_decay: float = 0.25,
) -> dict[str, Any]:
    base_package = Path(base_package_path)
    base_manifest_path = (
        base_package
        if base_package.is_file()
        else base_package / PACKAGE_MANIFEST_NAME
    )
    manifest = json.loads(base_manifest_path.read_text(encoding="utf-8-sig"))
    rows = _execution_rows(historical_execution_jsonl, tasks_jsonl)
    train_rows, holdout_rows = _split_repo_disjoint(rows)
    train_repos = {str(row["repo_origin"]) for row in train_rows}
    holdout_repos = {str(row["repo_origin"]) for row in holdout_rows}
    matcher = _train_verification_matcher(
        train_rows=train_rows,
        cortex_model_path=cortex_model_path,
        cortex_device=cortex_device,
        cortex_dtype=cortex_dtype,
    )
    output_dir = Path(output_package_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "verification_cortex.pt"
    matcher.matcher.save(checkpoint_path)
    manifest.update(
        {
            "policy_label": "phase2cg_unified_package_verification_cortex",
            "verification_cortex_path": str(checkpoint_path),
            "verification_cortex_model_name": str(cortex_model_path),
            "verification_cortex_recency_decay": float(recency_decay),
            "verification_control_source": "package_internal_verification_cortex",
        }
    )
    output_manifest = output_dir / PACKAGE_MANIFEST_NAME
    output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    checks = {
        "actual_execution_rows_present": len(rows) >= 12,
        "train_holdout_repos_disjoint": train_repos.isdisjoint(holdout_repos),
        "minimum_train_repos_met": len(train_repos) >= 3,
        "minimum_holdout_repos_met": len(holdout_repos) >= 3,
        "verification_checkpoint_written": checkpoint_path.is_file(),
        "unified_package_manifest_written": output_manifest.is_file(),
        "matcher_training_accuracy_complete": (
            matcher.matcher.training_summary is not None
            and matcher.matcher.training_summary.training_top1_accuracy == 1.0
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cg_unified_verification_package_build",
        "passed": passed,
        "checks": checks,
        "train_repos": sorted(train_repos),
        "holdout_repos": sorted(holdout_repos),
        "historical_rows": len(rows),
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "package_manifest": str(output_manifest),
        "verification_cortex_checkpoint": str(checkpoint_path),
        "verification_cortex_metadata": matcher.metadata(),
        "claim_boundary": (
            "single deployment package with distinct patch-selection and "
            "post-verification cortical experts; not a monolithic 7B-head claim"
        ),
    }
    output_report = Path(output_report_json)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a native nervous package with an internal verification cortical expert."
    )
    parser.add_argument("--base-package-path", required=True)
    parser.add_argument("--historical-execution-jsonl", required=True)
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--cortex-model-path", required=True)
    parser.add_argument("--cortex-device", default="cpu")
    parser.add_argument("--cortex-dtype", default="auto")
    parser.add_argument("--output-package-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--recency-decay", type=float, default=0.25)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2cg_unified_verification_package(
        base_package_path=args.base_package_path,
        historical_execution_jsonl=args.historical_execution_jsonl,
        tasks_jsonl=args.tasks_jsonl,
        cortex_model_path=args.cortex_model_path,
        cortex_device=args.cortex_device,
        cortex_dtype=args.cortex_dtype,
        output_package_dir=args.output_package_dir,
        output_report_json=args.output_report_json,
        recency_decay=args.recency_decay,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
