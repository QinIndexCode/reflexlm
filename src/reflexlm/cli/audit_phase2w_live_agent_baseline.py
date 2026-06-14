from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_CONFIG_KEYS = {
    "run_id",
    "model_or_provider",
    "tool_budget",
    "context_policy",
    "retry_policy",
    "edit_permissions",
    "stop_rule",
    "cost_or_command_budget",
    "task_source_manifest_sha256",
    "executor_commit",
    "sandbox_image_or_env",
    "trace_archive_uri",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _read_json(path: str | Path) -> dict[str, Any]:
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


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(_score(row.get(key)) for row in rows) / len(rows)


def audit_phase2w_live_agent_baseline(
    *,
    config_json: str | Path,
    results_jsonl: str | Path,
    min_rows: int = 64,
) -> dict[str, Any]:
    config = _read_json(config_json)
    rows = _read_jsonl(results_jsonl)
    required_row_keys = {
        "task_id",
        "task_success",
        "patch_correctness",
        "test_pass_rate",
        "stop_condition_correctness",
        "false_completion",
        "unauthorized_write_count",
        "commands_used",
        "elapsed_seconds",
        "transcript_sha256",
        "patch_sha256",
        "pre_test_log_sha256",
        "post_test_log_sha256",
    }
    row_shape_ok = bool(rows) and all(required_row_keys.issubset(row) for row in rows)
    provenance_keys = {
        "run_id",
        "task_source_manifest_sha256",
        "executor_commit",
        "sandbox_image_or_env",
        "trace_archive_uri",
    }
    provenance_complete = REQUIRED_CONFIG_KEYS.issubset(config) and all(
        str(config.get(key) or "").strip() for key in provenance_keys
    )
    config_hashes_valid = REQUIRED_CONFIG_KEYS.issubset(config) and _sha256(
        config.get("task_source_manifest_sha256")
    ) and isinstance(config.get("executor_commit"), str) and bool(
        COMMIT_RE.fullmatch(config["executor_commit"])
    )
    row_provenance_complete = row_shape_ok and all(
        str(row.get(key) or "").strip()
        for row in rows
        for key in (
            "transcript_sha256",
            "patch_sha256",
            "pre_test_log_sha256",
            "post_test_log_sha256",
        )
    )
    row_provenance_hashes_valid = row_shape_ok and all(
        _sha256(row.get(key))
        for row in rows
        for key in (
            "transcript_sha256",
            "patch_sha256",
            "pre_test_log_sha256",
            "post_test_log_sha256",
        )
    )
    checks = {
        "config_complete": REQUIRED_CONFIG_KEYS.issubset(config),
        "config_provenance_complete": provenance_complete,
        "config_hashes_valid": config_hashes_valid,
        "baseline_kind_live_tool_agent": config.get("baseline_kind") == "live_tool_agent",
        "rows_minimum_met": len(rows) >= min_rows,
        "row_schema_complete": row_shape_ok,
        "row_provenance_hashes_present": row_provenance_complete,
        "row_provenance_hashes_valid": row_provenance_hashes_valid,
        "budget_recorded_per_row": row_shape_ok and all("commands_used" in row for row in rows),
        "no_unauthorized_writes": row_shape_ok
        and all(_score(row.get("unauthorized_write_count")) == 0.0 for row in rows),
        "false_completion_recorded": row_shape_ok
        and all("false_completion" in row for row in rows),
        "nonzero_baseline_capability": _mean(rows, "task_success") > 0.0,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2w_live_agent_baseline_audit",
        "passed": passed,
        "baseline_kind": config.get("baseline_kind"),
        "model_or_provider": config.get("model_or_provider"),
        "tool_budget": config.get("tool_budget"),
        "context_policy": config.get("context_policy"),
        "retry_policy": config.get("retry_policy"),
        "edit_permissions": config.get("edit_permissions"),
        "stop_rule": config.get("stop_rule"),
        "cost_or_command_budget": config.get("cost_or_command_budget"),
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "task_success": _mean(rows, "task_success"),
            "patch_correctness": _mean(rows, "patch_correctness"),
            "test_pass_rate": _mean(rows, "test_pass_rate"),
            "stop_condition_correctness": _mean(rows, "stop_condition_correctness"),
            "false_completion_rate": _mean(rows, "false_completion"),
            "mean_commands_used": _mean(rows, "commands_used"),
            "mean_elapsed_seconds": _mean(rows, "elapsed_seconds"),
            "unauthorized_write_count": sum(
                _score(row.get("unauthorized_write_count")) for row in rows
            ),
        },
        "blocked_actions": []
        if passed
        else ["do_not_use_static_or_declared_baseline_as_phase2w_live_agent"],
        "inputs": {
            "config_json": str(Path(config_json)),
            "results_jsonl": str(Path(results_jsonl)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2W live-agent baseline.")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2w_live_agent_baseline(
        config_json=args.config_json,
        results_jsonl=args.results_jsonl,
        min_rows=args.min_rows,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
