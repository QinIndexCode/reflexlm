from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2v_eval_summary(
    *,
    source_eval_summary_json: str | Path,
    val_jsonl: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
) -> dict[str, Any]:
    source = _read_json(source_eval_summary_json)
    data_health = _read_json(data_health_json)
    pretrain = _read_json(pretrain_gate_json)
    return {
        "audit_family": "phase2v_graded_transfer_eval_summary",
        "sealed_data_used_for_training_or_tuning": False,
        "prediction_source": "derived_from_source_eval_summary",
        "phase2v_row_level_predictions_recomputed": False,
        "source_eval_summary_json": str(Path(source_eval_summary_json)),
        "val_jsonl": str(Path(val_jsonl)),
        "data_health_json": str(Path(data_health_json)),
        "pretrain_gate_json": str(Path(pretrain_gate_json)),
        "source_adapter_name": source.get("adapter_name"),
        "source_training_config_hash": source.get("training_config_hash"),
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain.get("passed") is True,
        "missing_controls": list(source.get("missing_controls") or []),
        "metrics": source.get("metrics", {}),
        "claim_boundary": {
            "phase2v_eval_is_nonsealed": True,
            "reuses_phase2u_package_or_adapter": True,
            "derived_metrics_not_independent_row_level_eval": True,
            "sealed_v3_feedback_used": False,
            "purpose": "graded_transfer_nonzero_control_evaluation",
        },
        "blocked_claims": [
            "do_not_use_phase2v_derived_summary_as_independent_architecture_evidence",
            "require_row_level_phase2v_predictions_before_epoch_claim",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2V eval summary from non-sealed graded-transfer inputs.")
    parser.add_argument("--source-eval-summary-json", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2v_eval_summary(
        source_eval_summary_json=args.source_eval_summary_json,
        val_jsonl=args.val_jsonl,
        data_health_json=args.data_health_json,
        pretrain_gate_json=args.pretrain_gate_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
