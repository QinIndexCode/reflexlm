import json
from pathlib import Path

from reflexlm.cli.build_phase2at_pretrain_readiness_report import (
    build_phase2at_pretrain_readiness_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2at_pretrain_readiness_accepts_all_gates(tmp_path: Path) -> None:
    report = build_phase2at_pretrain_readiness_report(
        architecture_boundary_json=_write(
            tmp_path / "boundary.json",
            {"passed": True, "unsupported_claims": ["epoch_making_architecture"]},
        ),
        data_gate_json=_write(
            tmp_path / "data.json",
            {
                "passed": True,
                "schema_version": "phase2at.learned_bounded_patch_candidate.v1",
                "metrics": {},
                "unsupported_claims": ["production_autonomy"],
            },
        ),
        package_gate_json=_write(
            tmp_path / "package.json",
            {
                "passed": True,
                "metrics": {"patch_proposal_strategy": "learned_bounded_candidate"},
                "unsupported_claims": ["freeform_patch_generation"],
            },
        ),
    )

    assert report["passed"] is True
    assert report["ready_for_training"] is True
    assert (
        "phase2at_ready_for_learned_bounded_patch_candidate_training"
        in report["supported_claims"]
    )


def test_phase2at_pretrain_readiness_does_not_require_package_gate_before_training(
    tmp_path: Path,
) -> None:
    report = build_phase2at_pretrain_readiness_report(
        architecture_boundary_json=_write(
            tmp_path / "boundary.json",
            {"passed": True, "unsupported_claims": []},
        ),
        data_gate_json=_write(
            tmp_path / "data.json",
            {
                "passed": True,
                "schema_version": "phase2at.learned_bounded_patch_candidate.v1",
                "metrics": {},
                "unsupported_claims": [],
            },
        ),
    )

    assert report["passed"] is True
    assert report["ready_for_training"] is True
    assert report["checks"]["package_gate_supplied"] is False
    assert report["inputs"]["package_gate_json"] is None


def test_phase2at_pretrain_readiness_blocks_current_old_schema(tmp_path: Path) -> None:
    report = build_phase2at_pretrain_readiness_report(
        architecture_boundary_json=_write(
            tmp_path / "boundary.json",
            {"passed": True, "unsupported_claims": []},
        ),
        data_gate_json=_write(
            tmp_path / "data.json",
            {
                "passed": False,
                "schema_version": "phase2at.learned_bounded_patch_candidate.v1",
                "metrics": {"target_failure_reasons": {"missing": 1}},
                "unsupported_claims": ["learned_patch_generation"],
            },
        ),
        package_gate_json=_write(
            tmp_path / "package.json",
            {
                "passed": False,
                "metrics": {"patch_proposal_strategy": "none"},
                "unsupported_claims": ["learned_patch_generation"],
            },
        ),
    )

    assert report["passed"] is False
    assert report["ready_for_training"] is False
    assert "phase2at_data_health_failed" in report["blockers"]
    assert "do_not_start_phase2at_training" in report["blocked_actions"]
