from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric(gate: dict[str, Any], split: str, name: str) -> float:
    split_metrics = gate.get("split_metrics") if isinstance(gate.get("split_metrics"), dict) else {}
    payload = split_metrics.get(split, {}).get(name, {})
    if not isinstance(payload, dict):
        return 0.0
    return float(payload.get("accuracy") or 0.0)


def _classify_gate(gate: dict[str, Any]) -> dict[str, Any]:
    val_source = _metric(gate, "val", "identity_text_ablated_source_overlap")
    holdout_source = _metric(gate, "holdout", "identity_text_ablated_source_overlap")
    val_identity = _metric(gate, "val", "runtime_identity_heuristic")
    holdout_identity = _metric(gate, "holdout", "runtime_identity_heuristic")
    source_max = max(val_source, holdout_source)
    identity_max = max(val_identity, holdout_identity)
    if source_max <= 0.0 and identity_max >= 0.95:
        issue = "identity_sidecar_ceiling_with_zero_nonidentity_control"
    elif source_max >= 0.90:
        issue = "source_overlap_ceiling_control_too_easy"
    elif identity_max >= 0.95:
        issue = "runtime_identity_heuristic_ceiling"
    else:
        issue = "mixed_gate_failure"
    return {
        "passed": bool(gate.get("passed")),
        "issue": issue,
        "val_identity_text_ablated_source_overlap": val_source,
        "holdout_identity_text_ablated_source_overlap": holdout_source,
        "val_runtime_identity_heuristic": val_identity,
        "holdout_runtime_identity_heuristic": holdout_identity,
        "blocked_actions": gate.get("blocked_actions", []),
    }


def build_phase2af_hardened_failure_audit(
    *,
    gate_jsons: list[str | Path],
    output_json: str | Path,
) -> dict[str, Any]:
    gate_summaries = {
        Path(path).stem: _classify_gate(_read_json(path)) for path in gate_jsons
    }
    passed_any = any(summary["passed"] for summary in gate_summaries.values())
    issue_counts: dict[str, int] = {}
    for summary in gate_summaries.values():
        issue = str(summary["issue"])
        issue_counts[issue] = issue_counts.get(issue, 0) + 1

    report = {
        "artifact_family": "phase2af_hardened_structural_sidecar_failure_audit",
        "passed_any_pretrain_gate": passed_any,
        "gate_count": len(gate_summaries),
        "gate_summaries": gate_summaries,
        "issue_counts": issue_counts,
        "training_allowed": passed_any,
        "claim_upgrade_allowed": False,
        "root_cause": (
            "Available non-sealed pools do not yet form a graded transfer/control benchmark. "
            "Phase2AE-style rows are solved by the deterministic runtime identity sidecar while "
            "nonidentity controls are zero; Phase2S/Phase2U-style rows are largely solved by "
            "source-overlap controls. Neither condition supports a hardened architecture claim."
        ),
        "required_next_dataset_properties": [
            "repo-origin-disjoint public traces only",
            "identity_text_ablated_source_overlap in [0.05, 0.75] on val and holdout",
            "runtime_identity_heuristic <= 0.90 on val and holdout",
            "full package must beat best non-full measured baseline after training",
            "no sealed-v3 feedback, no candidate slot marker, no gold hint",
        ],
        "blocked_actions": []
        if passed_any
        else [
            "do_not_train_phase2af_full",
            "do_not_package_phase2af",
            "do_not_claim_hardened_structural_sidecar_mechanism",
        ],
    }
    _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize why Phase2AF hardened structural-sidecar pretrain gates failed."
    )
    parser.add_argument("--gate-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2af_hardened_failure_audit(
        gate_jsons=args.gate_json,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
