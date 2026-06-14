from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2au_policy_required_runtime_tasks import (
    BENCHMARK_FAMILY,
    REQUIRED_DIFFICULTY_AXES,
    _test_has_parser_oracle,
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


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_test_paths(row: dict[str, Any], dataset_root: Path) -> list[Path]:
    artifacts = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    raw_values: list[Any] = []
    for key in ("generated_test", "generated_tests", "test_artifacts"):
        value = artifacts.get(key)
        if isinstance(value, list):
            raw_values.extend(value)
        elif value:
            raw_values.append(value)
    if row.get("materialized_test_target"):
        raw_values.append(row["materialized_test_target"])

    paths: list[Path] = []
    seen: set[str] = set()
    for value in raw_values:
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        resolved = path if path.is_absolute() else dataset_root / path
        key = str(resolved)
        if key not in seen:
            paths.append(resolved)
            seen.add(key)
    return paths


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _contract_supports_conversion(row: dict[str, Any]) -> bool:
    contract = row.get("runtime_visible_contract")
    if isinstance(contract, dict):
        return (
            contract.get("no_candidate_slot_marker") is True
            and contract.get("no_gold_hint") is True
            and contract.get("no_sealed_feedback") is True
        )
    normalization = row.get("normalization") if isinstance(row.get("normalization"), dict) else {}
    return (
        row.get("source_kind") == "public_repo"
        and row.get("synthetic_fault_injected_in_sandbox_only") is True
        and normalization.get("sealed_feedback_absent") is True
        and normalization.get("preserved_runtime_visible_evidence") is True
    )


def _expected_policy_supports_conversion(row: dict[str, Any]) -> bool:
    policy = row.get("expected_policy")
    if isinstance(policy, dict):
        return (
            policy.get("patch_proposal") == 1
            and policy.get("patch_operation") is not None
            and policy.get("patch_template") is not None
        )
    runtime = row.get("repair_runtime") if isinstance(row.get("repair_runtime"), dict) else {}
    return (
        isinstance(row.get("repair_candidates"), list)
        and bool(row.get("repair_candidates"))
        and isinstance(row.get("expected_repair_action"), str)
        and runtime.get("post_patch_tests_recorded") is True
        and runtime.get("rollback_recorded") is True
    )


def _normalized_difficulty_axes(row: dict[str, Any]) -> list[str]:
    axes = {str(axis) for axis in row.get("difficulty_axes", [])}
    difficulty = row.get("difficulty") if isinstance(row.get("difficulty"), dict) else {}
    runtime = row.get("repair_runtime") if isinstance(row.get("repair_runtime"), dict) else {}
    evidence = (
        row.get("runtime_visible_evidence")
        if isinstance(row.get("runtime_visible_evidence"), dict)
        else {}
    )
    candidate_count = difficulty.get("candidate_count")
    if isinstance(candidate_count, int) and candidate_count >= 2:
        axes.add("ambiguous_nonliteral_semantic")
    changed_files = evidence.get("changed_files")
    if isinstance(changed_files, list) and len({str(path) for path in changed_files}) > 1:
        axes.add("multi_file_interaction")
    if runtime.get("rollback_recorded") is True:
        axes.add("stateful_verification")
    if (
        runtime.get("rollback_failure_recorded") is True
        or row.get("synthetic_fault_injected_in_sandbox_only") is True
    ):
        axes.add("negative_constraint")
    return sorted(axes)


def _normalized_expected_policy(row: dict[str, Any]) -> dict[str, Any]:
    policy = row.get("expected_policy")
    if isinstance(policy, dict):
        return dict(policy)
    evidence = (
        row.get("runtime_visible_evidence")
        if isinstance(row.get("runtime_visible_evidence"), dict)
        else {}
    )
    repair_modes = evidence.get("repair_modes")
    if isinstance(repair_modes, list) and repair_modes:
        template = "+".join(sorted({str(mode) for mode in repair_modes}))
    else:
        template = "runtime_visible_behavioral_patch"
    return {
        "patch_proposal": 1,
        "patch_operation": "apply_patch_and_rerun_tests",
        "patch_template": template,
        "bounded_edit_scope": 1,
        "rollback_safety": 1,
        "test_selection": 1,
        "verification_state": 1,
        "progress_monitor": 1,
        "stop_condition": 1,
    }


def _repair_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("repair_candidates")
    if not isinstance(candidates, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        action = str(candidate.get("repair_action") or "").strip()
        if not action:
            continue
        sanitized.append(
            {
                "repair_action": action,
                "intent": str(candidate.get("intent") or "apply_patch_and_rerun_tests"),
                "description": str(candidate.get("description") or ""),
                "edit_scope": str(candidate.get("edit_scope") or "bounded_public_source_patch"),
                "structural_probe_hash": str(candidate.get("structural_probe_hash") or ""),
                "target_symbol": str(candidate.get("target_symbol") or ""),
            }
        )
    return sanitized


def _candidate_policy_commands(row: dict[str, Any], test_rels: list[str]) -> list[str]:
    commands: list[str] = []
    verify = test_rels[0] if test_rels else "<missing_generated_test>"
    for candidate in _repair_candidates(row):
        action = candidate["repair_action"]
        probe = candidate.get("structural_probe_hash") or ""
        symbol = candidate.get("target_symbol") or ""
        commands.append(
            f"phase2au_apply_candidate --repair-action {action} "
            f"structural_probe_hash={probe} target_symbol={symbol} "
            f"--verify \"python -m pytest -q {verify} --maxfail=1\""
        )
    return commands


def _normalized_contract(row: dict[str, Any]) -> dict[str, Any]:
    contract = row.get("runtime_visible_contract")
    if isinstance(contract, dict):
        return dict(contract)
    return {
        "no_candidate_slot_marker": True,
        "no_gold_hint": True,
        "no_sealed_feedback": True,
        "public_repo_origin": row.get("source_kind") == "public_repo",
        "sandbox_fault_only": row.get("synthetic_fault_injected_in_sandbox_only") is True,
    }


def _runtime_identity(row: dict[str, Any]) -> dict[str, Any]:
    evidence = (
        row.get("runtime_visible_evidence")
        if isinstance(row.get("runtime_visible_evidence"), dict)
        else {}
    )
    hashes = [
        str(item)
        for item in evidence.get("structural_probe_hashes", [])
        if str(item).strip()
    ]
    return {
        "command_identity_tokens": hashes,
        "identity_source": "runtime_visible_structural_probe_hashes",
        "sealed_feedback_used": False,
    }


def _row_id(row: dict[str, Any]) -> str:
    for key in ("task_id", "trace_id", "materialization_source_trace_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "<missing-task-id>"


def _convert_row(
    row: dict[str, Any],
    *,
    index: int,
    split: str,
    dataset_root: Path,
    test_paths: list[Path],
) -> dict[str, Any]:
    test_rels = [_relative_path(path, dataset_root) for path in test_paths]
    candidate_policy_commands = _candidate_policy_commands(row, test_rels)
    contract = _normalized_contract(row)
    contract.update(
        {
            "policy_required_runtime_delta": True,
            "no_policy_symbolic_control_expected_to_fail": True,
            "no_direct_text_membership_or_ast_attr_oracle": True,
            "real_generated_tests_present": True,
        }
    )
    converted = {
        "task_id": f"phase2au:{split}:{index:05d}",
        "benchmark_family": BENCHMARK_FAMILY,
        "split": split,
        "repo_origin": row.get("repo_origin") or row.get("repo_url_or_origin"),
        "repo_commit": row.get("repo_commit") or row.get("commit_hash"),
        "problem_statement": row.get("problem_statement")
        or "Resolve a policy-required public-repo runtime repair task using only runtime-visible evidence.",
        "evaluation_commands": [
            f"python -m pytest -q {test_rel} --maxfail=1" for test_rel in test_rels
        ],
        "candidate_policy_commands": candidate_policy_commands,
        "repair_candidates": _repair_candidates(row),
        "expected_repair_action": row.get("expected_repair_action"),
        "artifact_paths": {"generated_tests": test_rels},
        "allowed_write_scope": row.get("allowed_write_scope")
        or (
            row.get("runtime_visible_evidence", {}).get("changed_files")
            if isinstance(row.get("runtime_visible_evidence"), dict)
            else None
        ),
        "difficulty_axes": _normalized_difficulty_axes(row),
        "runtime_visible_contract": contract,
        "runtime_visible_identity": _runtime_identity(row),
        "expected_policy": _normalized_expected_policy(row),
        "sealed_feedback_used": row.get("sealed_feedback_used", False),
        "source": {
            "source_task_id": row.get("task_id") or row.get("trace_id"),
            "source_benchmark_family": row.get("benchmark_family"),
            "source_task_spec_sha256": row.get("task_spec_sha256") or row.get("trace_hash"),
        },
    }
    converted["task_spec_sha256"] = _sha256(
        {
            "source": converted["source"],
            "artifact_paths": converted["artifact_paths"],
            "expected_policy": converted["expected_policy"],
            "runtime_visible_contract": converted["runtime_visible_contract"],
            "candidate_policy_commands": converted["candidate_policy_commands"],
            "expected_repair_action": converted["expected_repair_action"],
            "runtime_visible_identity": converted["runtime_visible_identity"],
        }
    )
    return converted


def build_phase2au_policy_required_runtime_tasks(
    *,
    input_tasks_jsonl: str | Path,
    dataset_root: str | Path,
    output_jsonl: str | Path,
    split: str,
    min_rows: int = 64,
) -> dict[str, Any]:
    rows = _read_jsonl(input_tasks_jsonl)
    root = Path(dataset_root)
    output_root = Path(output_jsonl).parent
    converted: list[dict[str, Any]] = []
    reject_counts: dict[str, int] = {}
    rejected_examples: dict[str, list[str]] = {}

    def record_reject(row: dict[str, Any], reason: str) -> None:
        reject_counts[reason] = reject_counts.get(reason, 0) + 1
        bucket = rejected_examples.setdefault(reason, [])
        if len(bucket) < 10:
            bucket.append(_row_id(row))

    for row in rows:
        axes = set(_normalized_difficulty_axes(row))
        test_paths = _candidate_test_paths(row, root)
        row_rejects: list[str] = []
        if not axes.intersection(REQUIRED_DIFFICULTY_AXES):
            row_rejects.append("missing_required_policy_difficulty_axes")
        if not _contract_supports_conversion(row):
            row_rejects.append("missing_leakage_control_contract")
        if not _expected_policy_supports_conversion(row):
            row_rejects.append("missing_explicit_learned_patch_policy_heads")
        if len(_repair_candidates(row)) < 2:
            row_rejects.append("missing_nontrivial_repair_candidates")
        elif str(row.get("expected_repair_action") or "") not in {
            candidate["repair_action"] for candidate in _repair_candidates(row)
        }:
            row_rejects.append("expected_repair_action_missing_from_candidates")
        if not test_paths or not all(path.exists() and path.is_file() for path in test_paths):
            row_rejects.append("missing_real_generated_test_files")
        elif any(_test_has_parser_oracle(path) for path in test_paths):
            row_rejects.append("parser_oracle_generated_tests")
        if row_rejects:
            for reason in row_rejects:
                record_reject(row, reason)
            continue
        for path in test_paths:
            destination = output_root / _relative_path(path, root)
            if path.resolve() != destination.resolve():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        converted.append(
            _convert_row(
                row,
                index=len(converted),
                split=split,
                dataset_root=root,
                test_paths=test_paths,
            )
        )

    _write_jsonl(output_jsonl, converted)
    passed = len(converted) >= min_rows
    return {
        "artifact_family": "phase2au_policy_required_runtime_task_builder",
        "passed": passed,
        "source_row_count": len(rows),
        "converted_row_count": len(converted),
        "reject_counts": dict(sorted(reject_counts.items())),
        "rejected_examples": rejected_examples,
        "output_jsonl": str(Path(output_jsonl)),
        "input_tasks_jsonl": str(Path(input_tasks_jsonl)),
        "dataset_root": str(Path(dataset_root)),
        "claim_boundary": (
            "phase2au_candidate_split_ready_for_task_gate"
            if passed
            else "phase2au_candidate_split_gap_not_training_ready"
        ),
        "blocked_actions": []
        if passed
        else [
            "do_not_train_phase2au",
            "do_not_package_or_claim_runtime_delta",
            "collect_or_materialize_non_parser_oracle_public_runtime_tasks_first",
        ],
        "thresholds": {"min_rows": min_rows},
        "task_split_sha256": _sha256(converted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AU policy-required runtime tasks from existing non-sealed candidates."
    )
    parser.add_argument("--input-tasks-jsonl", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2au_policy_required_runtime_tasks(
        input_tasks_jsonl=args.input_tasks_jsonl,
        dataset_root=args.dataset_root,
        output_jsonl=args.output_jsonl,
        split=args.split,
        min_rows=args.min_rows,
    )
    _write_json(args.summary_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
