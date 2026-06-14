import json
from pathlib import Path

from reflexlm.cli.build_phase2av_holdout_failure_audit import (
    build_phase2av_holdout_failure_audit,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_phase2av_holdout_failure_audit_classifies_identity_transfer_gap(
    tmp_path: Path,
) -> None:
    postflight = _write(
        tmp_path / "holdout_postflight.json",
        {
            "passed": False,
            "metrics": {
                "command_slot_accuracy": 0.78,
                "source_overlap_accuracy": 0.43,
                "model_minus_source_overlap_accuracy": 0.35,
                "patch_operation_accuracy": 0.96,
                "patch_template_slot_accuracy": 0.86,
                "patch_target_file_slot_accuracy": 1.0,
            },
            "failure_modes": ["command_slot_accuracy_below_gate"],
        },
    )
    predictions = _write(
        tmp_path / "predictions.json",
        {
            "command_identity_logit_bias": 0.0,
            "prediction_records": [
                {
                    "example_id": "ex0",
                    "command_slot_label": 0,
                    "command_slot_prediction": 0,
                    "command_slot_correct": True,
                    "source_overlap_correct": True,
                    "command_identity_margin": 0.25,
                    "command_identity_confidence": 0.25,
                },
                {
                    "example_id": "ex1",
                    "command_slot_label": 1,
                    "command_slot_prediction": 0,
                    "command_slot_correct": False,
                    "source_overlap_correct": False,
                    "command_identity_margin": 0.25,
                    "command_identity_confidence": 0.25,
                },
            ],
        },
    )
    rows = _write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {
                "example_id": "ex0",
                "candidate_commands": ["a", "b"],
                "source_task_manifest": {"repo_origin": "repo-a"},
                "learned_patch_policy_target": {
                    "patch_operation": "replace_literal",
                    "patch_template": "literal_restoration",
                },
            },
            {
                "example_id": "ex1",
                "candidate_commands": ["a", "b"],
                "source_task_manifest": {"repo_origin": "repo-b"},
                "learned_patch_policy_target": {
                    "patch_operation": "replace_literal",
                    "patch_template": "literal_restoration",
                },
            },
        ],
    )
    data_health = _write(tmp_path / "data_health.json", {"passed": True})

    report = build_phase2av_holdout_failure_audit(
        holdout_postflight_json=postflight,
        prediction_json=predictions,
        head_rows_jsonl=rows,
        data_health_jsons=[data_health],
    )

    assert report["passed"] is False
    assert "command_slot_identity_transfer_below_holdout_gate" in report[
        "issue_classification"
    ]
    assert "descriptor_heads_pass_while_command_slot_fails" in report[
        "issue_classification"
    ]
    assert "command_identity_prior_disabled" in report["issue_classification"]
    assert "operation_specific_gap:replace_literal" in report["issue_classification"]
    assert "data_health_passed_nonsealed_failure_not_data_gate" in report[
        "issue_classification"
    ]
    assert "do_not_start_phase2av_full_training" in report["blocked_actions"]
    assert "sealed_cross_model_transfer" in report["unsupported_claims"]


def test_phase2av_holdout_failure_audit_keeps_sealed_boundary(tmp_path: Path) -> None:
    postflight = _write(
        tmp_path / "holdout_postflight.json",
        {
            "passed": False,
            "metrics": {
                "command_slot_accuracy": 0.4,
                "model_minus_source_overlap_accuracy": 0.0,
                "patch_operation_accuracy": 0.4,
                "patch_template_slot_accuracy": 0.4,
                "patch_target_file_slot_accuracy": 1.0,
            },
        },
    )
    predictions = _write(
        tmp_path / "predictions.json",
        {
            "command_identity_logit_bias": 4.0,
            "prediction_records": [
                {
                    "example_id": "ex0",
                    "command_slot_label": 1,
                    "command_slot_prediction": 0,
                    "command_slot_correct": False,
                    "source_overlap_correct": False,
                }
            ],
        },
    )
    rows = _write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {
                "example_id": "ex0",
                "candidate_commands": ["a", "b"],
                "source_task_manifest": {"repo_origin": "repo-a"},
                "learned_patch_policy_target": {
                    "patch_operation": "insert_import",
                    "patch_template": "import_restoration",
                },
            }
        ],
    )

    report = build_phase2av_holdout_failure_audit(
        holdout_postflight_json=postflight,
        prediction_json=predictions,
        head_rows_jsonl=rows,
    )

    assert report["preconditions"]["sealed_feedback_used"] is False
    assert "do_not_run_sealed_eval_for_phase2av" in report["blocked_actions"]
    assert report["recommended_next_actions"][-1] == (
        "keep_sealed_v3_excluded_from_data_design_training_and_failure_feedback"
    )
