from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2t_architecture_iteration import (
    REQUIRED_FACTOR_LEVELS,
    REQUIRED_GRADED_FACTORS,
    REQUIRED_TASK_FAMILIES,
)


VALID_SPLITS = {"train", "val", "holdout"}
FORBIDDEN_MARKERS = (
    "external_trace_v3_semantic_required",
    "phase2i_external_trace",
    "phase2g_external_trace",
    "sealed_failure",
    "sealed_feedback",
    "hidden_hint",
    "gold_label",
    "gold_command",
    "correct_patch",
    "expected_patch",
    "correct_command_hint",
)
CANDIDATE_SLOT_RE = re.compile(r"(?i)\bcandidate[_-]?\d+\b|\bslot[_-]?\d+\b")
HEX_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
REQUIRED_REPAIR_LOOP_CONTRACT_FLAGS = {
    "sandbox_compatible",
    "source_repo_readonly",
    "patches_required",
    "tests_required",
    "rollback_required",
    "stop_required",
    "safety_pressure_included",
    "modern_baseline_measurable",
}
REQUIRED_PROVENANCE_FLAGS = {
    "public_repo",
    "license_metadata_present",
    "commit_pinned",
    "repo_origin_recorded",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_set(value: Any) -> set[str]:
    return {str(item) for item in _as_list(value) if str(item)}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _bool_dict_has_true_flags(value: Any, required: set[str]) -> bool:
    payload = _dict(value)
    return all(payload.get(key) is True for key in required)


def _repo_origin(spec: dict[str, Any]) -> str:
    raw = str(spec.get("repo_url") or spec.get("url") or spec.get("repo_origin") or "")
    normalized = raw.strip().lower().removesuffix(".git").rstrip("/")
    return normalized


def _split(spec: dict[str, Any]) -> str:
    return str(spec.get("split") or "").lower()


def _commit_hash(spec: dict[str, Any]) -> str:
    return str(spec.get("commit_hash") or spec.get("commit") or "")


def _license(spec: dict[str, Any]) -> str:
    return str(spec.get("license") or spec.get("license_or_synthetic_origin") or "").strip()


def _source_kind(spec: dict[str, Any]) -> str:
    return str(spec.get("source_kind") or "public_repo")


def _task_families(spec: dict[str, Any]) -> set[str]:
    return _string_set(spec.get("task_families"))


def _factor_values(spec: dict[str, Any]) -> dict[str, set[str]]:
    raw = _dict(spec.get("factor_levels"))
    return {key: {str(item) for item in _as_list(raw.get(key))} for key in REQUIRED_GRADED_FACTORS}


def _mentions_forbidden_marker(spec: dict[str, Any]) -> bool:
    text = _canonical_json(spec).replace("\\", "/").lower()
    return any(marker in text for marker in FORBIDDEN_MARKERS) or bool(CANDIDATE_SLOT_RE.search(text))


def _public_url_ok(origin: str) -> bool:
    return origin.startswith("https://github.com/") or origin.startswith("https://gitlab.com/")


def _per_spec_report(spec: dict[str, Any]) -> dict[str, Any]:
    split = _split(spec)
    origin = _repo_origin(spec)
    commit_hash = _commit_hash(spec)
    license_name = _license(spec)
    provenance = _dict(spec.get("provenance"))
    repair_contract = _dict(spec.get("repair_loop_contract"))
    task_families = _task_families(spec)
    factors = _factor_values(spec)
    checks = {
        "split_valid": split in VALID_SPLITS,
        "source_kind_public": _source_kind(spec) == "public_repo",
        "repo_origin_public_and_recorded": bool(origin) and _public_url_ok(origin),
        "commit_pinned": bool(HEX_RE.match(commit_hash)),
        "license_metadata_present": bool(license_name),
        "provenance_flags_present": _bool_dict_has_true_flags(
            provenance, REQUIRED_PROVENANCE_FLAGS
        ),
        "repair_loop_contract_present": _bool_dict_has_true_flags(
            repair_contract, REQUIRED_REPAIR_LOOP_CONTRACT_FLAGS
        ),
        "task_family_known": bool(task_families) and task_families.issubset(
            REQUIRED_TASK_FAMILIES
        ),
        "factor_levels_known": all(
            values and values.issubset(REQUIRED_FACTOR_LEVELS[factor])
            for factor, values in factors.items()
        ),
        "no_forbidden_or_sealed_markers": not _mentions_forbidden_marker(spec),
    }
    return {
        "repo_id": str(spec.get("repo_id") or origin),
        "split": split,
        "repo_origin": origin,
        "commit_hash": commit_hash,
        "license": license_name,
        "task_families": sorted(task_families),
        "factor_levels": {key: sorted(values) for key, values in factors.items()},
        "checks": checks,
        "passed": all(checks.values()),
    }


def _split_repo_disjoint(reports: list[dict[str, Any]]) -> bool:
    by_origin: dict[str, set[str]] = defaultdict(set)
    for report in reports:
        origin = str(report.get("repo_origin") or "")
        if origin:
            by_origin[origin].add(str(report.get("split") or ""))
    return all(len(splits) == 1 for splits in by_origin.values())


def _coverage(reports: list[dict[str, Any]]) -> tuple[set[str], dict[str, set[str]]]:
    task_families: set[str] = set()
    factor_values: dict[str, set[str]] = {factor: set() for factor in REQUIRED_GRADED_FACTORS}
    for report in reports:
        task_families.update(str(item) for item in report.get("task_families") or [])
        for factor, values in _dict(report.get("factor_levels")).items():
            factor_values.setdefault(factor, set()).update(str(item) for item in values or [])
    return task_families, factor_values


def _split_coverage(
    reports: list[dict[str, Any]],
    split: str,
) -> tuple[set[str], dict[str, set[str]]]:
    return _coverage([report for report in reports if report.get("split") == split])


def _missing_factor_levels(factor_values: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        factor: sorted(required - factor_values.get(factor, set()))
        for factor, required in REQUIRED_FACTOR_LEVELS.items()
        if required - factor_values.get(factor, set())
    }


def build_phase2t_public_repair_loop_spec_manifest(
    *,
    repo_specs_json: str | Path,
    min_train_repos: int = 3,
    min_val_repos: int = 2,
    min_holdout_repos: int = 2,
) -> dict[str, Any]:
    specs = _read_json(repo_specs_json)
    if not isinstance(specs, list):
        raise TypeError("repo_specs_json must contain a JSON list of repo specs")
    reports = [_per_spec_report(spec) for spec in specs if isinstance(spec, dict)]
    split_counts = Counter(str(report.get("split") or "") for report in reports)
    task_families, factor_values = _coverage(reports)
    missing_task_families = sorted(REQUIRED_TASK_FAMILIES - task_families)
    missing_factor_levels = _missing_factor_levels(factor_values)
    split_missing_task_families: dict[str, list[str]] = {}
    split_missing_factor_levels: dict[str, dict[str, list[str]]] = {}
    for split in ("train", "val", "holdout"):
        split_task_families, split_factor_values = _split_coverage(reports, split)
        split_missing_task_families[split] = sorted(
            REQUIRED_TASK_FAMILIES - split_task_families
        )
        split_missing_factor_levels[split] = _missing_factor_levels(split_factor_values)
    repo_origins = [str(report.get("repo_origin") or "") for report in reports]
    checks = {
        "repo_specs_json_exists": Path(repo_specs_json).exists(),
        "repo_specs_are_list": isinstance(specs, list),
        "all_specs_pass_shape_checks": bool(reports) and all(report["passed"] for report in reports),
        "train_repo_count_minimum_met": split_counts["train"] >= min_train_repos,
        "val_repo_count_minimum_met": split_counts["val"] >= min_val_repos,
        "holdout_repo_count_minimum_met": split_counts["holdout"] >= min_holdout_repos,
        "repo_origins_unique": len([origin for origin in repo_origins if origin])
        == len(set(origin for origin in repo_origins if origin)),
        "repo_origins_split_disjoint": _split_repo_disjoint(reports),
        "task_family_coverage_complete": not missing_task_families,
        "graded_factor_coverage_complete": not missing_factor_levels,
        "train_task_family_coverage_complete": not split_missing_task_families["train"],
        "train_graded_factor_coverage_complete": not split_missing_factor_levels["train"],
        "val_task_family_coverage_complete": not split_missing_task_families["val"],
        "val_graded_factor_coverage_complete": not split_missing_factor_levels["val"],
        "holdout_task_family_coverage_complete": not split_missing_task_families["holdout"],
        "holdout_graded_factor_coverage_complete": not split_missing_factor_levels["holdout"],
        "no_sealed_or_forbidden_markers": not any(
            not report["checks"]["no_forbidden_or_sealed_markers"] for report in reports
        ),
    }

    blocked_actions: list[str] = []
    if not all(checks.values()):
        blocked_actions.append("do_not_collect_phase2t_traces_until_specs_pass")
    if not checks["all_specs_pass_shape_checks"]:
        blocked_actions.append("revise_phase2t_repo_specs_shape_or_contract")
    if not (
        checks["train_repo_count_minimum_met"]
        and checks["val_repo_count_minimum_met"]
        and checks["holdout_repo_count_minimum_met"]
    ):
        blocked_actions.append("add_more_repo_origin_disjoint_phase2t_specs")
    if not checks["repo_origins_unique"] or not checks["repo_origins_split_disjoint"]:
        blocked_actions.append("do_not_reuse_repo_origin_across_phase2t_splits")
    if not checks["task_family_coverage_complete"]:
        blocked_actions.append("add_phase2t_task_family_coverage_before_collection")
    if not checks["graded_factor_coverage_complete"]:
        blocked_actions.append("add_phase2t_graded_factor_coverage_before_collection")
    if not (
        checks["train_task_family_coverage_complete"]
        and checks["val_task_family_coverage_complete"]
        and checks["holdout_task_family_coverage_complete"]
    ):
        blocked_actions.append("add_phase2t_splitwise_task_family_coverage_before_collection")
    if not (
        checks["train_graded_factor_coverage_complete"]
        and checks["val_graded_factor_coverage_complete"]
        and checks["holdout_graded_factor_coverage_complete"]
    ):
        blocked_actions.append("add_phase2t_splitwise_graded_factor_coverage_before_collection")
    if not checks["no_sealed_or_forbidden_markers"]:
        blocked_actions.append("remove_sealed_gold_candidate_or_expected_patch_markers")

    passed = all(checks.values())
    return {
        "collector_family": "phase2t_public_repair_loop_spec_collector",
        "passed": passed,
        "claim_bearing_collection_candidate": passed,
        "claim_bearing_training_ready": False,
        "allowed_next_action": (
            "run_phase2t_dynamic_repair_trace_collection"
            if passed
            else "revise_phase2t_public_repair_loop_specs"
        ),
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "thresholds": {
            "min_train_repos": min_train_repos,
            "min_val_repos": min_val_repos,
            "min_holdout_repos": min_holdout_repos,
            "required_task_families": sorted(REQUIRED_TASK_FAMILIES),
            "required_factor_levels": {
                key: sorted(values) for key, values in REQUIRED_FACTOR_LEVELS.items()
            },
            "required_repair_loop_contract_flags": sorted(REQUIRED_REPAIR_LOOP_CONTRACT_FLAGS),
            "required_provenance_flags": sorted(REQUIRED_PROVENANCE_FLAGS),
        },
        "rollups": {
            "split_counts": dict(sorted(split_counts.items())),
            "repo_origins": sorted(repo_origins),
            "task_families": sorted(task_families),
            "missing_task_families": missing_task_families,
            "factor_levels": {
                key: sorted(values) for key, values in sorted(factor_values.items())
            },
            "missing_factor_levels": missing_factor_levels,
            "split_missing_task_families": split_missing_task_families,
            "split_missing_factor_levels": split_missing_factor_levels,
        },
        "repos": reports,
        "spec_hash": _sha256(specs),
        "inputs": {"repo_specs_json": str(Path(repo_specs_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Phase2T public repair-loop repository specs before trace collection."
    )
    parser.add_argument("--repo-specs-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--min-train-repos", type=int, default=3)
    parser.add_argument("--min-val-repos", type=int, default=2)
    parser.add_argument("--min-holdout-repos", type=int, default=2)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    manifest = build_phase2t_public_repair_loop_spec_manifest(
        repo_specs_json=args.repo_specs_json,
        min_train_repos=args.min_train_repos,
        min_val_repos=args.min_val_repos,
        min_holdout_repos=args.min_holdout_repos,
    )
    if args.output_json:
        _write_json(args.output_json, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not manifest["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
