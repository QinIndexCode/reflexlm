from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


BENCHMARK_FAMILY = "phase2av_graded_descriptor_runtime"
CLAIM_BOUNDARY = "phase2av_pretrain_gate_before_learned_descriptor_runtime_claim"


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


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _artifact_test_paths(row: dict[str, Any], source_root: Path) -> list[Path]:
    artifacts = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    raw_values: list[Any] = []
    for key in ("generated_test", "generated_tests", "test_artifacts"):
        value = artifacts.get(key)
        if isinstance(value, list):
            raw_values.extend(value)
        elif value:
            raw_values.append(value)
    paths: list[Path] = []
    seen: set[str] = set()
    for value in raw_values:
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        resolved = path if path.is_absolute() else source_root / path
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            paths.append(resolved)
    return paths


def _repair_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    values = row.get("repair_candidates")
    if not isinstance(values, list):
        return []
    candidates: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        action = str(value.get("repair_action") or "").strip()
        if not action:
            continue
        candidates.append(
            {
                "repair_action": action,
                "intent": str(value.get("intent") or "apply_patch_and_rerun_tests"),
                "edit_scope": str(value.get("edit_scope") or "bounded_public_source_patch"),
                "structural_probe_hash": str(value.get("structural_probe_hash") or ""),
                "target_symbol": str(value.get("target_symbol") or ""),
                "description": str(value.get("description") or ""),
            }
        )
    return candidates


def _candidate_policy_commands(row: dict[str, Any], test_rels: list[str]) -> list[str]:
    verify = test_rels[0] if test_rels else "<missing_generated_test>"
    commands: list[str] = []
    for candidate in _repair_candidates(row):
        commands.append(
            "phase2av_apply_descriptor_candidate "
            f"--repair-action {candidate['repair_action']} "
            f"operation=<learned_patch_operation> template=<learned_patch_template> "
            f"structural_probe_hash={candidate['structural_probe_hash']} "
            f"target_symbol={candidate['target_symbol']} "
            f"--verify \"python -m pytest -q {verify} --maxfail=1\""
        )
    return commands


def _difficulty_axes(row: dict[str, Any]) -> list[str]:
    axes = set(str(axis) for axis in row.get("difficulty_axes", []) if str(axis).strip())
    difficulty = row.get("difficulty") if isinstance(row.get("difficulty"), dict) else {}
    if int(difficulty.get("candidate_count") or 0) >= 2:
        axes.add("ambiguous_nonliteral_semantic")
    if difficulty.get("repair_depth") in {"two_edits", "stale_state_refresh"}:
        axes.add("multi_step_or_stale_descriptor")
    if difficulty.get("evidence_density") in {"medium", "high"}:
        axes.add("graded_evidence_density")
    axes.add("learned_patch_descriptor_required")
    return sorted(axes)


def _target(row: dict[str, Any]) -> dict[str, Any]:
    target = row.get("learned_patch_candidate_target")
    return dict(target) if isinstance(target, dict) else {}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def _operation_exception_consistency(row: dict[str, Any]) -> tuple[bool, str | None]:
    target = _target(row)
    operation = str(target.get("operation") or "")
    template = str(target.get("after_fragment_template_id") or "")
    evidence_text = _json_text(row.get("runtime_visible_evidence") or {})
    has_name_error = any(
        marker in evidence_text
        for marker in ("nameerror", "importerror", "modulenotfounderror", "is not defined")
    )
    has_attribute_error = any(
        marker in evidence_text
        for marker in ("attributeerror", "has no attribute")
    )
    if has_name_error and has_attribute_error:
        return False, "mixed_name_and_attribute_exception_evidence"
    if operation == "replace_attribute" or template == "call_attribute_restoration":
        if has_name_error:
            return False, "name_error_labeled_as_attribute_restoration"
    if operation == "insert_import" or template == "import_restoration":
        if has_attribute_error:
            return False, "attribute_error_labeled_as_import_restoration"
    return True, None


def _repo_origin(row: dict[str, Any]) -> str:
    return str(row.get("repo_url_or_origin") or row.get("repo_origin") or row.get("repo_id") or "")


def _repo_commit(row: dict[str, Any]) -> str:
    return str(row.get("commit_hash") or row.get("repo_commit") or "")


def _convert_row(
    row: dict[str, Any],
    *,
    index: int,
    split: str,
    source_root: Path,
    test_paths: list[Path],
) -> dict[str, Any]:
    target = _target(row)
    test_rels = [_relative_path(path, source_root) for path in test_paths]
    converted = {
        "task_id": f"phase2av:{split}:{index:05d}",
        "benchmark_family": BENCHMARK_FAMILY,
        "claim_boundary": CLAIM_BOUNDARY,
        "split": split,
        "source_kind": row.get("source_kind"),
        "repo_origin": _repo_origin(row),
        "repo_commit": _repo_commit(row),
        "problem_statement": row.get("problem_statement")
        or "Select and execute a bounded descriptor-conditioned repair candidate using only runtime-visible public-repo evidence.",
        "current_visible_text": row.get("current_visible_text")
        or "Public repository structural repair task with bounded descriptor-conditioned repair candidates.",
        "runtime_visible_evidence": row.get("runtime_visible_evidence") or {},
        "artifact_paths": {"generated_tests": test_rels},
        "evaluation_commands": [
            f"python -m pytest -q {test_rel} --maxfail=1" for test_rel in test_rels
        ],
        "repair_candidates": _repair_candidates(row),
        "candidate_policy_commands": _candidate_policy_commands(row, test_rels),
        "expected_repair_action": row.get("expected_repair_action"),
        "expected_policy": {
            "patch_proposal": 1,
            "patch_operation": target.get("operation"),
            "patch_template": target.get("after_fragment_template_id"),
            "bounded_edit_scope": 1,
            "rollback_safety": 1,
            "test_selection": 1,
            "verification_state": 1,
            "progress_monitor": 1,
            "stop_condition": 1,
        },
        "learned_patch_descriptor_target": target,
        "runtime_visible_contract": {
            "no_candidate_slot_marker": True,
            "no_gold_hint": True,
            "no_sealed_feedback": True,
            "public_repo_origin": row.get("source_kind") == "public_repo",
            "learned_descriptor_runtime_delta_required": True,
            "no_parser_oracle_generated_tests": True,
            "no_recorded_patch_text_target": row.get("recorded_patch_artifact_as_generation_target")
            is False,
            "no_symbolic_generator_target": row.get("symbolic_generator_as_generation_target")
            is False,
            "no_freeform_patch_generation": row.get("freeform_patch_generation") is False,
        },
        "difficulty_axes": _difficulty_axes(row),
        "sealed_feedback_used": row.get("sealed_feedback_used", False),
        "source": {
            "source_task_id": row.get("trace_id") or row.get("task_id"),
            "source_benchmark_family": row.get("benchmark_family"),
            "source_trace_hash": row.get("trace_hash"),
        },
    }
    converted["task_spec_sha256"] = _sha256(
        {
            "source": converted["source"],
            "repo_origin": converted["repo_origin"],
            "repo_commit": converted["repo_commit"],
            "artifact_paths": converted["artifact_paths"],
            "expected_policy": converted["expected_policy"],
            "expected_repair_action": converted["expected_repair_action"],
            "learned_patch_descriptor_target": converted["learned_patch_descriptor_target"],
            "candidate_policy_commands": converted["candidate_policy_commands"],
        }
    )
    return converted


def build_phase2av_graded_descriptor_runtime_tasks(
    *,
    input_jsonl: str | Path,
    source_dataset_root: str | Path,
    output_jsonl: str | Path,
    split: str,
    min_rows: int = 64,
) -> dict[str, Any]:
    source_root = Path(source_dataset_root)
    output_root = Path(output_jsonl).parent
    rows = _read_jsonl(input_jsonl)
    converted: list[dict[str, Any]] = []
    reject_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        reject_counts[reason] = reject_counts.get(reason, 0) + 1

    for row in rows:
        target = _target(row)
        test_paths = _artifact_test_paths(row, source_root)
        candidates = _repair_candidates(row)
        row_rejects: list[str] = []
        if row.get("source_kind") != "public_repo":
            row_rejects.append("not_public_repo")
        if row.get("sealed_feedback_used") is not False and not (
            isinstance(row.get("normalization"), dict)
            and row["normalization"].get("sealed_feedback_absent") is True
        ):
            row_rejects.append("sealed_feedback_not_absent")
        if row.get("freeform_patch_generation") is not False:
            row_rejects.append("freeform_patch_generation_not_disabled")
        if row.get("recorded_patch_artifact_as_generation_target") is not False:
            row_rejects.append("recorded_patch_target_not_disabled")
        if row.get("symbolic_generator_as_generation_target") is not False:
            row_rejects.append("symbolic_generator_target_not_disabled")
        if not target.get("operation") or not target.get("after_fragment_template_id"):
            row_rejects.append("missing_descriptor_operation_or_template")
        if len(candidates) < 2:
            row_rejects.append("missing_nontrivial_repair_candidates")
        if str(row.get("expected_repair_action") or "") not in {
            candidate["repair_action"] for candidate in candidates
        }:
            row_rejects.append("expected_repair_action_missing_from_candidates")
        consistent, consistency_reason = _operation_exception_consistency(row)
        if not consistent:
            row_rejects.append(str(consistency_reason))
        if not test_paths or not all(path.exists() and path.is_file() for path in test_paths):
            row_rejects.append("missing_generated_tests")
        if row_rejects:
            for reason in row_rejects:
                reject(reason)
            continue
        for path in test_paths:
            destination = output_root / _relative_path(path, source_root)
            if path.resolve() != destination.resolve():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        converted.append(
            _convert_row(
                row,
                index=len(converted),
                split=split,
                source_root=source_root,
                test_paths=test_paths,
            )
        )

    pairs = sorted(
        {
            (
                str(row["expected_policy"].get("patch_operation") or ""),
                str(row["expected_policy"].get("patch_template") or ""),
            )
            for row in converted
        }
    )
    _write_jsonl(output_jsonl, converted)
    passed = len(converted) >= min_rows
    return {
        "artifact_family": "phase2av_graded_descriptor_runtime_task_builder",
        "passed": passed,
        "claim_boundary": (
            "phase2av_candidate_split_ready_for_data_health"
            if passed
            else "phase2av_candidate_split_gap_not_training_ready"
        ),
        "source_row_count": len(rows),
        "converted_row_count": len(converted),
        "operation_template_pairs": [list(pair) for pair in pairs],
        "operation_template_pair_count": len(pairs),
        "reject_counts": dict(sorted(reject_counts.items())),
        "input_jsonl": str(Path(input_jsonl)),
        "source_dataset_root": str(Path(source_dataset_root)),
        "output_jsonl": str(Path(output_jsonl)),
        "thresholds": {"min_rows": min_rows},
        "task_split_sha256": _sha256(converted),
        "blocked_actions": []
        if passed
        else [
            "do_not_train_phase2av",
            "do_not_claim_learned_descriptor_runtime_delta",
            "collect_or_materialize_non_parser_oracle_multi_template_runtime_tasks_first",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AV graded descriptor runtime task specs."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--source-dataset-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2av_graded_descriptor_runtime_tasks(
        input_jsonl=args.input_jsonl,
        source_dataset_root=args.source_dataset_root,
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
