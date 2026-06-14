import json
from pathlib import Path

from reflexlm.cli.build_phase2af_holdout_failure_audit import (
    build_phase2af_holdout_failure_audit,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2af_holdout_failure_audit_classifies_runtime_identity_gap(tmp_path: Path) -> None:
    postflight = _write(
        tmp_path / "postflight.json",
        {
            "passed": False,
            "checks": {
                "command_slot_accuracy_min": True,
                "model_beats_source_overlap": True,
                "model_beats_runtime_identity": False,
            },
            "metrics": {
                "split": "holdout",
                "command_slot_accuracy": 0.92,
                "command_slot_count": 100,
                "runtime_identity_heuristic_accuracy": 0.86,
            },
            "thresholds": {"min_model_minus_runtime_identity": 0.10},
        },
    )
    summary = _write(
        tmp_path / "summary.json",
        {
            "slot_intent_distribution": {"holdout": {"command_intents": {"other": 100}}},
            "eval_metrics": {
                "slot_confusion": {
                    "command_slot": {
                        "0": {"0": 44, "1": 6},
                        "1": {"1": 48, "0": 2},
                    }
                }
            },
        },
    )
    manifest = _write(
        tmp_path / "manifest.json",
        {
            "train_sampling": {
                "shortcut_bucket_balanced_train": True,
                "duplicates_are_training_sampling_only": True,
            }
        },
    )

    report = build_phase2af_holdout_failure_audit(
        postflight_json=postflight,
        eval_summary_json=summary,
        manifest_json=manifest,
        output_json=tmp_path / "audit.json",
    )

    assert report["failure_class"] == "runtime_identity_residual_shortcut_not_broken"
    assert report["runtime_identity_gap"]["required_correct_for_gate"] == 96
    assert report["runtime_identity_gap"]["actual_correct"] == 92
    assert report["runtime_identity_gap"]["additional_correct_rows_needed"] == 4
    assert report["slot_confusion"]["errors"] == 8
    assert report["slot_confusion"]["error_edges"][0]["count"] == 6
    assert report["diagnosis"]["single_command_intent_observed"] is True
    assert report["ready_for_package"] is False
