from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2au_policy_required_head_dataset import (
    CLAIM_BOUNDARY,
    DATASET_FAMILY,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


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


def _has_forbidden_target(row: dict[str, Any]) -> bool:
    target = row.get("learned_patch_policy_target")
    if not isinstance(target, dict):
        return True
    return any(
        bool(target.get(key))
        for key in (
            "patch_text",
            "patch_diff",
            "unified_diff",
            "recorded_patch",
            "symbolic_generator_output",
        )
    )


def audit_phase2au_pretrain_gate(
    *,
    task_gate_json: str | Path,
    head_manifest_json: str | Path,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
) -> dict[str, Any]:
    task_gate = _read_json(task_gate_json)
    manifest = _read_json(head_manifest_json)
    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl)
    all_rows = [*train_rows, *val_rows]
    split_hashes = manifest.get("effective_split_hashes")
    checks = {
        "task_gate_passed": task_gate.get("passed") is True,
        "head_manifest_passed": manifest.get("passed") is True,
        "dataset_family_expected": manifest.get("dataset_family") == DATASET_FAMILY,
        "claim_boundary_correct": manifest.get("claim_boundary") == CLAIM_BOUNDARY,
        "train_and_val_rows_present": bool(train_rows and val_rows),
        "effective_split_hashes_present": isinstance(split_hashes, dict)
        and bool(split_hashes.get("phase2au_head_train"))
        and bool(split_hashes.get("phase2au_head_val")),
        "all_rows_have_phase2au_scope": bool(all_rows)
        and all(
            row.get("open_repair_control_label_scope")
            == "phase2au_policy_required_runtime_task_gate"
            for row in all_rows
        ),
        "no_recorded_or_symbolic_patch_target": bool(all_rows)
        and not any(_has_forbidden_target(row) for row in all_rows),
        "package_and_sealed_blocked_before_runtime_delta": manifest.get("package_allowed") is False
        and manifest.get("sealed_eval_allowed") is False
        and manifest.get("runtime_delta_supported") is False,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2au_policy_required_pretrain_gate",
        "passed": passed,
        "ready_for_phase2au_smoke_training": passed,
        "checks": checks,
        "metrics": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "task_gate_row_count": (
                task_gate.get("metrics", {}).get("row_count")
                if isinstance(task_gate.get("metrics"), dict)
                else None
            ),
            "effective_split_hashes": split_hashes if isinstance(split_hashes, dict) else {},
        },
        "claim_boundary": (
            "phase2au_head_pretrain_ready_not_runtime_delta_evidence"
            if passed
            else "phase2au_head_pretrain_blocked"
        ),
        "blocked_actions": []
        if passed
        else [
            "do_not_start_phase2au_training_until_pretrain_gate_passes",
            "do_not_package_or_run_sealed_before_phase2au_runtime_delta_postflight",
        ],
        "unsupported_claims": [
            "runtime_delta_before_execution_postflight",
            "sealed_cross_model_transfer",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "inputs": {
            "task_gate_json": str(Path(task_gate_json)),
            "head_manifest_json": str(Path(head_manifest_json)),
            "train_jsonl": str(Path(train_jsonl)),
            "val_jsonl": str(Path(val_jsonl)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AU head pretrain readiness.")
    parser.add_argument("--task-gate-json", required=True)
    parser.add_argument("--head-manifest-json", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2au_pretrain_gate(
        task_gate_json=args.task_gate_json,
        head_manifest_json=args.head_manifest_json,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
