import json
from pathlib import Path

from reflexlm.cli.audit_phase2aw_verified_candidate_pool import (
    audit_phase2aw_verified_candidate_pool,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "gate.json",
        {
            "checks": {
                "bounded_symbolic_execution_only": True,
                "execution_safety_met": True,
            }
        },
    )


def _failure(
    tmp_path: Path,
    *,
    full: float = 0.9,
    control: float = 0.35,
    split_counts: dict[str, int] | None = None,
) -> Path:
    return _write(
        tmp_path / "failure.json",
        {
            "checks": {"selection_is_not_primary_bottleneck": True},
            "metrics": {
                "full_success_rate": full,
                "control_success_rate": control,
                "full_minus_control_success_rate": full - control,
            },
            "failure_breakdown": {
                "source_artifact_split_counts": split_counts or {"holdout": 100}
            },
        },
    )


def test_phase2aw_verified_candidate_pool_accepts_split_clean_runtime_gate(
    tmp_path: Path,
) -> None:
    report = audit_phase2aw_verified_candidate_pool(
        execution_gate_json=_gate(tmp_path),
        failure_audit_json=_failure(tmp_path),
    )

    assert report["passed"] is True
    assert report["ready_for_phase2aw_package_or_successor_training"] is True
    assert "phase2aw_split_clean_verified_candidate_pool_ready" in report[
        "supported_claims"
    ]


def test_phase2aw_verified_candidate_pool_rejects_mixed_or_weak_pool(
    tmp_path: Path,
) -> None:
    report = audit_phase2aw_verified_candidate_pool(
        execution_gate_json=_gate(tmp_path),
        failure_audit_json=_failure(
            tmp_path,
            full=0.76,
            control=0.37,
            split_counts={"holdout": 84, "train": 48, "val": 24},
        ),
    )

    assert report["passed"] is False
    assert "full_success_rate_gate" in report["blocking_reasons"]
    assert "source_artifact_split_clean" in report["blocking_reasons"]
    assert "do_not_package_phase2av_or_phase2aw" in report["blocked_actions"]

