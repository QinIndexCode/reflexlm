import json
from pathlib import Path

from reflexlm.cli.build_phase2w_epoch_preregistration import (
    build_phase2w_epoch_preregistration,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _readiness(*, bounded: bool = True, epoch: bool = False) -> dict:
    return {
        "bounded_mechanism_claim_ready": bounded,
        "epoch_making_architecture_claim_ready": epoch,
        "epoch_claim_blockers": [
            "independent_external_reproduction_passed",
            "open_ended_repair_benchmark_passed",
            "modern_live_agent_baseline_passed",
            "production_safety_benchmark_passed",
            "unanimous_readonly_reviewer_consensus",
        ],
    }


def test_phase2w_preregistration_targets_all_current_epoch_blockers(tmp_path: Path) -> None:
    report = build_phase2w_epoch_preregistration(
        phase2o_readiness_json=_write(tmp_path / "readiness.json", _readiness())
    )
    assert report["passed"] is True
    assert report["allowed_next_action"] == "build_phase2w_independent_open_repair_inputs"
    assert report["claim_boundary"]["before_phase2w_passes"] == "bounded_mechanism_evidence_only"
    assert "stop_if_any_task_specific_command_path_or_patch_template_is_hardcoded" in report["stop_rules"]
    assert report["study_design"]["modern_live_agent_baseline"]["required"] is True


def test_phase2w_preregistration_rejects_when_phase2o_bounded_not_ready(tmp_path: Path) -> None:
    report = build_phase2w_epoch_preregistration(
        phase2o_readiness_json=_write(
            tmp_path / "readiness.json", _readiness(bounded=False)
        )
    )
    assert report["passed"] is False
    assert "do_not_start_phase2w_training_until_preregistration_passes" in report["blocked_actions"]


def test_phase2w_preregistration_rejects_when_epoch_claim_already_marked_ready(
    tmp_path: Path,
) -> None:
    report = build_phase2w_epoch_preregistration(
        phase2o_readiness_json=_write(tmp_path / "readiness.json", _readiness(epoch=True))
    )
    assert report["passed"] is False
    assert report["checks"]["phase2o_epoch_not_ready"] is False
