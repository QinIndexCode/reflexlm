from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.run_phase2z_public_structural_repair_execution import _state_for_public_policy
from reflexlm.llm.candidate_features import (
    redact_structured_command_identity_text,
    source_overlap_command_slot_prediction,
)
from reflexlm.llm.head_dataset import build_phase2c_head_state_prompt_from_state
from reflexlm.llm.receptor_latent import runtime_command_identity_signal
from reflexlm.models.features import candidate_commands


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def _runtime_state_for_row(row: dict[str, Any]):
    expected = row.get("expected_repair_result") if isinstance(row.get("expected_repair_result"), dict) else {}
    test_rel = str(expected.get("test_target") or "phase2aa_runtime_baseline/test_case.py")
    return _state_for_public_policy(
        row=row,
        pre_test={
            "duration_seconds": 0.0,
            "exit_code": 1,
            "stdout": "",
            "stderr": "",
        },
        test_rel=test_rel,
    )


def _identity_text_ablated_state(state):
    return state.model_copy(
        deep=True,
        update={
            "goal": state.goal.model_copy(
                update={
                    "description": redact_structured_command_identity_text(
                        state.goal.description
                    ),
                    "command_allowlist": [
                        redact_structured_command_identity_text(command)
                        for command in candidate_commands(state)
                    ],
                }
            ),
            "terminal": state.terminal.model_copy(
                update={
                    "stdout_delta": redact_structured_command_identity_text(
                        state.terminal.stdout_delta
                    ),
                    "stderr_delta": redact_structured_command_identity_text(
                        state.terminal.stderr_delta
                    ),
                }
            ),
        },
    )


def _identity_prediction(state) -> int:
    signal = runtime_command_identity_signal(state)
    scores = [
        float(signal.get(f"command_identity_slot:{index}", 0.0))
        for index in range(len(candidate_commands(state)))
    ]
    if not scores:
        return 0
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def _source_overlap_prediction(state) -> int:
    return source_overlap_command_slot_prediction(
        build_phase2c_head_state_prompt_from_state(state),
        candidate_commands(state),
    )


def _summary_metrics(path: str | Path | None) -> dict[str, Any]:
    payload = _read_json(path)
    if not payload:
        return {}
    return {
        "rows": payload.get("rows"),
        "success_rate": payload.get("success_rate"),
        "patch_candidate_selection_accuracy": payload.get("patch_candidate_selection_accuracy"),
        "policy_loaded": payload.get("policy_loaded"),
    }


def build_phase2aa_candidate_selection_baseline_report(
    *,
    rows_jsonl: str | Path,
    output_json: str | Path,
    full_summary_json: str | Path | None = None,
    no_nsi_summary_json: str | Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(rows_jsonl)
    totals = {
        "slot0": 0,
        "source_overlap": 0,
        "source_overlap_identity_text_ablated": 0,
        "runtime_identity_heuristic": 0,
    }
    by_expected_slot: dict[str, dict[str, int]] = {}
    row_results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        expected_slot = int(row.get("expected_patch_candidate_slot", 0))
        state = _runtime_state_for_row(row)
        text_ablated_state = _identity_text_ablated_state(state)
        predictions = {
            "slot0": 0,
            "source_overlap": _source_overlap_prediction(state),
            "source_overlap_identity_text_ablated": _source_overlap_prediction(
                text_ablated_state
            ),
            "runtime_identity_heuristic": _identity_prediction(state),
        }
        slot_key = str(expected_slot)
        by_expected_slot.setdefault(
            slot_key,
            {
                "total": 0,
                "slot0": 0,
                "source_overlap": 0,
                "source_overlap_identity_text_ablated": 0,
                "runtime_identity_heuristic": 0,
            },
        )
        by_expected_slot[slot_key]["total"] += 1
        for name, prediction in predictions.items():
            correct = int(prediction == expected_slot)
            totals[name] += correct
            by_expected_slot[slot_key][name] += correct
        row_results.append(
            {
                "row_index": index,
                "trace_id": row.get("trace_id"),
                "expected_patch_candidate_slot": expected_slot,
                "predictions": predictions,
                "candidate_count": len(candidate_commands(state)),
            }
        )

    total = len(rows)
    baseline_metrics = {
        name: {"correct": correct, "total": total, "accuracy": _accuracy(correct, total)}
        for name, correct in totals.items()
    }
    full_metrics = _summary_metrics(full_summary_json)
    no_nsi_metrics = _summary_metrics(no_nsi_summary_json)
    full_accuracy = full_metrics.get("patch_candidate_selection_accuracy")
    identity_accuracy = baseline_metrics["runtime_identity_heuristic"]["accuracy"]
    source_accuracy = baseline_metrics["source_overlap"]["accuracy"]
    source_text_ablated_accuracy = baseline_metrics[
        "source_overlap_identity_text_ablated"
    ]["accuracy"]
    no_nsi_accuracy = no_nsi_metrics.get("patch_candidate_selection_accuracy")
    checks = {
        "source_overlap_below_full": (
            isinstance(full_accuracy, (int, float)) and float(source_accuracy) < float(full_accuracy)
        ),
        "source_overlap_identity_text_ablated_below_full": (
            isinstance(full_accuracy, (int, float))
            and float(source_text_ablated_accuracy) < float(full_accuracy)
        ),
        "no_nsi_below_full": (
            isinstance(full_accuracy, (int, float))
            and isinstance(no_nsi_accuracy, (int, float))
            and float(no_nsi_accuracy) < float(full_accuracy)
        ),
        "identity_heuristic_below_full": (
            isinstance(full_accuracy, (int, float)) and float(identity_accuracy) < float(full_accuracy)
        ),
    }
    interpretation = {
        "bounded_candidate_selection_supported": bool(
            checks["source_overlap_identity_text_ablated_below_full"]
            and checks["no_nsi_below_full"]
        ),
        "learned_head_necessity_supported": bool(checks["identity_heuristic_below_full"]),
        "architecture_claim_boundary": (
            "runtime_identity_signal_and_package_wiring_supported_but_learned_head_necessity_not_proven"
            if not checks["identity_heuristic_below_full"]
            else "learned_head_beats_measured_heuristic_baseline_on_this_split"
        ),
    }
    report = {
        "artifact_family": "phase2aa_candidate_selection_baseline_report",
        "rows_jsonl": str(Path(rows_jsonl)),
        "row_count": total,
        "baseline_metrics": baseline_metrics,
        "by_expected_slot": by_expected_slot,
        "full_summary": full_metrics,
        "no_nsi_summary": no_nsi_metrics,
        "checks": checks,
        "interpretation": interpretation,
        "row_results": row_results,
        "unsupported_claims": [
            "freeform_patch_generation",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "sealed_transfer",
            "epoch_making_architecture",
        ],
    }
    _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AA bounded patch candidate selection baseline report."
    )
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--full-summary-json")
    parser.add_argument("--no-nsi-summary-json")
    args = parser.parse_args()
    report = build_phase2aa_candidate_selection_baseline_report(
        rows_jsonl=args.rows_jsonl,
        output_json=args.output_json,
        full_summary_json=args.full_summary_json,
        no_nsi_summary_json=args.no_nsi_summary_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
