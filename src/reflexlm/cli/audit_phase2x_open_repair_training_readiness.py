from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


OPEN_REPAIR_LABEL_FIELDS = (
    "patch_proposal_label",
    "test_selection_slot",
    "rollback_safety_label",
    "stop_condition_label",
    "bounded_edit_scope_label",
    "progress_monitor_label",
    "verification_state_label",
)

FORBIDDEN_MARKERS = (
    "sealed_v2",
    "sealed_v3",
    "candidate_0",
    "candidate_1",
    "candidate_2",
    "candidate_3",
    "gold",
    "hidden_hint",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _valid_label(value: Any) -> bool:
    try:
        return int(value) != -100
    except (TypeError, ValueError):
        return False


def _label_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    total = max(len(rows), 1)
    coverage: dict[str, dict[str, int | float]] = {}
    for field in OPEN_REPAIR_LABEL_FIELDS:
        present = sum(1 for row in rows if _valid_label(row.get(field, -100)))
        coverage[field] = {
            "present": present,
            "total": len(rows),
            "ratio": present / total,
        }
    return coverage


def _contains_forbidden_marker(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("sealed_feedback_used") is True:
            return True
        text = json.dumps(row, sort_keys=True, ensure_ascii=False).lower()
        if any(marker in text for marker in FORBIDDEN_MARKERS):
            return True
    return False


def _all_labels_covered(coverage: dict[str, dict[str, int | float]]) -> bool:
    return all(int(payload["present"]) > 0 for payload in coverage.values())


def audit_phase2x_open_repair_training_readiness(
    *,
    task_manifest_audit_json: str | Path,
    runtime_capability_audit_json: str | Path,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    head_dataset_manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    task_audit = _read_json(task_manifest_audit_json)
    runtime_audit = _read_json(runtime_capability_audit_json)
    head_manifest = _read_json(head_dataset_manifest_json) if head_dataset_manifest_json else {}
    train_rows = _read_jsonl(train_jsonl)
    val_rows = _read_jsonl(val_jsonl)
    train_coverage = _label_coverage(train_rows)
    val_coverage = _label_coverage(val_rows)
    checks = {
        "task_manifest_audit_passed": task_audit.get("passed") is True,
        "runtime_capability_audit_recorded": bool(runtime_audit),
        "train_rows_present": len(train_rows) > 0,
        "val_rows_present": len(val_rows) > 0,
        "train_open_repair_labels_covered": _all_labels_covered(train_coverage),
        "val_open_repair_labels_covered": _all_labels_covered(val_coverage),
        "full_repair_control_label_scope": (
            head_manifest.get("full_repair_control_training_ready") is True
            if head_dataset_manifest_json
            else False
        ),
        "no_forbidden_markers_in_train": not _contains_forbidden_marker(train_rows),
        "no_forbidden_markers_in_val": not _contains_forbidden_marker(val_rows),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2x_open_repair_training_readiness",
        "passed": passed,
        "checks": checks,
        "label_fields": list(OPEN_REPAIR_LABEL_FIELDS),
        "label_coverage": {
            "train": train_coverage,
            "val": val_coverage,
        },
        "effective_split_hashes": {
            "phase2x_open_repair_train": _rows_sha256(train_rows),
            "phase2x_open_repair_val": _rows_sha256(val_rows),
        },
        "blocked_actions": []
        if passed
        else [
            "do_not_start_phase2x_open_repair_training",
            "do_not_collect_phase2x_as_real_repair_result",
            "do_not_claim_open_ended_repair_capability",
        ],
        "post_training_package_requirements": {
            "runtime_capability_audit_passed": runtime_audit.get("passed") is True,
            "runtime_capability_is_not_training_prerequisite": True,
        },
        "inputs": {
            "task_manifest_audit_json": str(Path(task_manifest_audit_json)),
            "runtime_capability_audit_json": str(Path(runtime_capability_audit_json)),
            "head_dataset_manifest_json": str(Path(head_dataset_manifest_json))
            if head_dataset_manifest_json
            else None,
            "train_jsonl": str(Path(train_jsonl)),
            "val_jsonl": str(Path(val_jsonl)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2X open-repair training readiness.")
    parser.add_argument("--task-manifest-audit-json", required=True)
    parser.add_argument("--runtime-capability-audit-json", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--head-dataset-manifest-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2x_open_repair_training_readiness(
        task_manifest_audit_json=args.task_manifest_audit_json,
        runtime_capability_audit_json=args.runtime_capability_audit_json,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        head_dataset_manifest_json=args.head_dataset_manifest_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
