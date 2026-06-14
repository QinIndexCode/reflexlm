import json
from pathlib import Path

import pytest

from reflexlm.cli.build_phase2af_prediction_failure_diagnostic import (
    build_phase2af_prediction_failure_diagnostic,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2af_prediction_failure_diagnostic_classifies_identity_ties(
    tmp_path: Path,
) -> None:
    summary = _write(
        tmp_path / "eval.json",
        {
            "prediction_records": [
                {
                    "row_index": 7,
                    "episode_id": "holdout:repo-a:case-7",
                    "source_trace": {"repo_id": "repo-a"},
                    "command_slot_correct": False,
                    "command_slot_label": 0,
                    "command_slot_prediction": 1,
                    "source_overlap_correct": True,
                    "command_identity_scores": {"slot:0": 6.0, "slot:1": 6.0, "slot:2": 0.0},
                    "command_identity_margin": 0.0,
                    "command_identity_confidence": 0.0,
                    "candidate_commands": ["gold", "competing", "other"],
                },
                {
                    "row_index": 8,
                    "source_trace": {"repo_id": "repo-a"},
                    "command_slot_correct": True,
                    "command_slot_label": 2,
                    "command_slot_prediction": 2,
                    "command_identity_scores": {"slot:2": 5.0, "slot:0": 0.0},
                    "candidate_commands": ["other", "other2", "gold"],
                },
            ]
        },
    )

    report = build_phase2af_prediction_failure_diagnostic(
        eval_summary_json=summary,
        output_json=tmp_path / "diagnostic.json",
    )

    assert report["passed"] is False
    assert report["failure_class"] == "identity_tie_candidate_indistinguishability"
    assert report["metrics"]["failed_command_slot_rows"] == 1
    assert report["metrics"]["identity_tie_failed_rows"] == 1
    assert report["failure_distribution"]["by_repo"] == {"repo-a": 1}
    assert report["failure_distribution"]["by_target_predicted_edge"] == {"0->1": 1}
    assert report["failed_rows"][0]["identity_tied_slots"] == [0, 1]
    assert report["failed_rows"][0]["gold_candidate"] == "gold"
    assert "do_not_package_phase2af" in report["blocked_actions"]


def test_phase2af_prediction_failure_diagnostic_handles_zero_score_ties(
    tmp_path: Path,
) -> None:
    summary = _write(
        tmp_path / "eval.json",
        {
            "prediction_records": [
                {
                    "row_index": 3,
                    "command_slot_correct": False,
                    "command_slot_label": 1,
                    "command_slot_prediction": 3,
                    "command_identity_scores": [0.0, 0.0, 0.0, 0.0],
                    "candidate_commands": ["a", "b", "c", "d"],
                }
            ]
        },
    )

    report = build_phase2af_prediction_failure_diagnostic(
        eval_summary_json=summary,
        output_json=tmp_path / "diagnostic.json",
    )

    assert report["metrics"]["zero_identity_score_tie_failed_rows"] == 1
    assert report["failed_rows"][0]["identity_tied_slots"] == [0, 1, 2, 3]


def test_phase2af_prediction_failure_diagnostic_passes_when_no_failed_rows(
    tmp_path: Path,
) -> None:
    summary = _write(
        tmp_path / "eval.json",
        {
            "prediction_records": [
                {
                    "row_index": 1,
                    "command_slot_correct": True,
                    "command_slot_label": 0,
                    "command_slot_prediction": 0,
                    "command_identity_scores": {"slot:0": 2.0, "slot:1": 0.0},
                }
            ]
        },
    )

    report = build_phase2af_prediction_failure_diagnostic(
        eval_summary_json=summary,
        output_json=tmp_path / "diagnostic.json",
    )

    assert report["passed"] is True
    assert report["failure_class"] == "no_failed_command_slot_rows"
    assert report["blocked_actions"] == []


def test_phase2af_prediction_failure_diagnostic_requires_prediction_records(
    tmp_path: Path,
) -> None:
    summary = _write(tmp_path / "eval.json", {"eval_metrics": {}})

    with pytest.raises(ValueError, match="prediction_records"):
        build_phase2af_prediction_failure_diagnostic(
            eval_summary_json=summary,
            output_json=tmp_path / "diagnostic.json",
        )
