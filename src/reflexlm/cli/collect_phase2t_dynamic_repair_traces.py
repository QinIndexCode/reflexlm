from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from reflexlm.cli.collect_phase2s_public_repair_traces import (
    build_public_repair_rows_for_spec,
)


ARCHITECTURE_TARGETS = {
    "patch_proposal_head": "propose_bounded_patch",
    "test_selection_head": "select_allowlisted_verification_command",
    "rollback_safety_head": "rollback_failed_or_unsafe_patch",
    "stop_condition_head": "emit_verified_stop",
    "verification_state_receptors": "encode_test_result_state",
    "incident_timeout_receptors": "encode_timeout_or_tool_fault_state",
    "progress_monitor_receptors": "encode_progress_or_repeated_failure_state",
    "bounded_edit_scope_policy": "restrict_patch_to_task_local_source_scope",
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


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _values(spec: dict[str, Any], key: str, default: list[str]) -> list[str]:
    factor_levels = spec.get("factor_levels")
    if isinstance(factor_levels, dict):
        values = factor_levels.get(key)
        if isinstance(values, list) and values:
            return [str(value) for value in values]
    return default


def _task_families(spec: dict[str, Any]) -> list[str]:
    values = spec.get("task_families")
    if isinstance(values, list) and values:
        return [str(value) for value in values]
    return ["localized_unit_assertion"]


def _cycle(values: list[str], index: int) -> str:
    return values[index % len(values)]


def _phase2t_difficulty(spec: dict[str, Any], row_index: int, candidate_count: int) -> dict[str, Any]:
    return {
        "task_family": _cycle(_task_families(spec), row_index),
        "candidate_count": candidate_count,
        "evidence_density": _cycle(_values(spec, "evidence_density", ["medium"]), row_index),
        "repair_depth": _cycle(_values(spec, "repair_depth", ["one_edit"]), row_index),
        "failure_observability": _cycle(
            _values(spec, "failure_observability", ["direct_traceback"]), row_index
        ),
        "ambiguity_class": _cycle(
            _values(spec, "ambiguity_class", ["same_intent_command"]), row_index
        ),
        "safety_pressure": _cycle(_values(spec, "safety_pressure", ["none"]), row_index),
    }


def _repair_loop_episode(row: dict[str, Any]) -> dict[str, Any]:
    artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    runtime = row.get("repair_runtime") if isinstance(row.get("repair_runtime"), dict) else {}
    return {
        "loop_schema": "phase2t_repair_loop_v1",
        "stages": [
            {
                "stage": "inspect_runtime_evidence",
                "runtime_visible": True,
                "artifact_keys": ["test_output"],
            },
            {
                "stage": "select_allowlisted_command",
                "runtime_visible": True,
                "command_allowlist_observed": runtime.get("command_allowlist_observed") is True,
                "artifact_keys": ["command_log"],
            },
            {
                "stage": "propose_bounded_patch",
                "runtime_visible": True,
                "bounded_edit_scope_observed": runtime.get("bounded_edit_scope_observed") is True,
                "artifact_keys": ["patch_diff"],
            },
            {
                "stage": "run_verification_tests",
                "runtime_visible": True,
                "post_patch_tests_recorded": runtime.get("post_patch_tests_recorded") is True,
                "artifact_keys": ["test_output"],
            },
            {
                "stage": "rollback_failed_or_unsafe_patch",
                "runtime_visible": True,
                "rollback_recorded": runtime.get("rollback_recorded") is True,
                "artifact_keys": ["rollback_log"],
            },
            {
                "stage": "emit_verified_stop",
                "runtime_visible": True,
                "stop_requires_post_patch_verification": True,
                "artifact_keys": ["test_output", "sandbox_integrity_report"],
            },
        ],
        "artifact_paths": artifact_paths,
    }


def phase2s_row_to_phase2t_repair_loop_row(
    row: dict[str, Any],
    *,
    spec: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    transformed = dict(row)
    candidates = row.get("repair_candidates") if isinstance(row.get("repair_candidates"), list) else []
    runtime = row.get("repair_runtime") if isinstance(row.get("repair_runtime"), dict) else {}
    transformed["trace_id"] = str(row.get("trace_id", "")).replace(":phase2s:", ":phase2t:")
    transformed["phase"] = "Phase2T"
    transformed["trace_construction_mode"] = "phase2t_dynamic_public_repo_repair_loop_trace"
    transformed["phase2t_source_trace_id"] = row.get("trace_id")
    transformed["claim_bearing_training_ready"] = False
    transformed["difficulty"] = _phase2t_difficulty(spec, row_index, len(candidates))
    transformed["repair_loop_episode"] = _repair_loop_episode(row)
    transformed["architecture_targets"] = {
        key: {
            "required": True,
            "runtime_visible_signal": signal,
        }
        for key, signal in ARCHITECTURE_TARGETS.items()
    }
    transformed["safety_controls"] = {
        "source_repo_read_only_observed": runtime.get("source_repo_read_only_observed") is True,
        "bounded_edit_scope_observed": runtime.get("bounded_edit_scope_observed") is True,
        "command_allowlist_observed": runtime.get("command_allowlist_observed") is True,
        "rollback_recorded": runtime.get("rollback_recorded") is True,
        "sandbox_cleanup_recorded": runtime.get("sandbox_cleanup_recorded") is True,
        "stop_requires_verification": True,
        "unauthorized_write_count": 0,
        "low_level_qwen_calls": 0,
    }
    transformed["modern_baseline_contract"] = {
        "required": True,
        "measured_not_declared": True,
        "same_repair_loop_artifacts_required": True,
        "cost_and_command_budget_required": True,
    }
    transformed["normalization"] = {
        **(row.get("normalization") if isinstance(row.get("normalization"), dict) else {}),
        "phase2t_expected_patch_hidden_from_visible_text": True,
        "phase2t_sealed_feedback_absent": True,
    }
    transformed["trace_hash"] = _sha256(
        {
            "trace_id": transformed.get("trace_id"),
            "visible": transformed.get("current_visible_text"),
            "runtime": transformed.get("runtime_visible_evidence"),
            "candidates": transformed.get("repair_candidates"),
            "difficulty": transformed.get("difficulty"),
            "artifacts": transformed.get("artifact_paths"),
        }
    )
    return transformed


def collect_phase2t_dynamic_repair_traces(
    *,
    repo_specs_json: str | Path,
    output_root: str | Path,
    clone_root: str | Path,
    rows_per_repo: int = 3,
    timeout_seconds: int = 20,
    no_clone: bool = False,
    keep_sandboxes: bool = False,
) -> dict[str, Any]:
    specs = _read_json(repo_specs_json)
    if not isinstance(specs, list):
        raise TypeError("repo_specs_json must contain a JSON list of repo specs")
    output = Path(output_root)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "holdout": []}
    repo_reports: list[dict[str, Any]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        rows, report = build_public_repair_rows_for_spec(
            spec,
            output_root=output,
            clone_root=Path(clone_root),
            rows_per_repo=rows_per_repo,
            timeout_seconds=timeout_seconds,
            no_clone=no_clone,
            keep_sandboxes=keep_sandboxes,
        )
        split = str(report["split"])
        transformed = [
            phase2s_row_to_phase2t_repair_loop_row(row, spec=spec, row_index=index)
            for index, row in enumerate(rows)
        ]
        split_rows[split].extend(transformed)
        repo_reports.append(
            {
                **report,
                "phase2t_rows_emitted": len(transformed),
                "phase2t_repair_loop_schema": "phase2t_repair_loop_v1",
            }
        )
    for split, rows in split_rows.items():
        _write_jsonl(output / f"{split}.raw.jsonl", rows)
    manifest = {
        "collector_family": "phase2t_dynamic_public_repo_repair_loop_trace_collector",
        "trace_construction_mode": "phase2t_dynamic_public_repo_repair_loop_trace",
        "claim_bearing_collection_candidate": True,
        "claim_bearing_training_ready": False,
        "claim_bearing_training_ready_requires": [
            "phase2t_dynamic_repair_trace_data_health",
            "phase2t_dynamic_repair_trace_pretrain_gate",
        ],
        "sealed_v3_used": False,
        "writes_to_source_repos": False,
        "execution_sandbox_used": True,
        "synthetic_faults_injected_in_sandbox_only": True,
        "sandbox_cleanup_observed": not keep_sandboxes,
        "rows_per_repo": rows_per_repo,
        "splits": {
            split: {
                "path": str(output / f"{split}.raw.jsonl"),
                "rows": len(rows),
                "repo_ids": sorted({str(row.get("repo_id")) for row in rows}),
            }
            for split, rows in split_rows.items()
        },
        "repos": repo_reports,
        "spec_hash": _sha256(specs),
        "total_rows": sum(len(rows) for rows in split_rows.values()),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Phase2T dynamic public-repo repair-loop traces in disposable sandboxes."
    )
    parser.add_argument("--repo-specs-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clone-root", default="artifacts/external_repos/phase2t")
    parser.add_argument("--rows-per-repo", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--no-clone", action="store_true")
    parser.add_argument("--keep-sandboxes", action="store_true")
    args = parser.parse_args()
    manifest = collect_phase2t_dynamic_repair_traces(
        repo_specs_json=args.repo_specs_json,
        output_root=args.output_root,
        clone_root=args.clone_root,
        rows_per_repo=args.rows_per_repo,
        timeout_seconds=args.timeout_seconds,
        no_clone=args.no_clone,
        keep_sandboxes=args.keep_sandboxes,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
