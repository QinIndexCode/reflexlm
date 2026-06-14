import json
from pathlib import Path

from reflexlm.cli.audit_phase2ar_symbolic_control_delta import (
    audit_phase2ar_symbolic_control_delta,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2ar_control_delta_accepts_nonzero_nonceiling_controls(tmp_path: Path) -> None:
    report = audit_phase2ar_symbolic_control_delta(
        full_summary_json=_write(
            tmp_path / "full.json",
            {"rows": 16, "successes": 16, "success_rate": 1.0},
        ),
        text_control_summary_json=_write(
            tmp_path / "text.json",
            {
                "rows": 16,
                "successes": 2,
                "success_rate": 0.125,
                "claim_boundary": "phase2ar_restricted_symbolic_control_not_claim_bearing",
            },
        ),
        attribute_control_summary_json=_write(
            tmp_path / "attr.json",
            {
                "rows": 16,
                "successes": 7,
                "success_rate": 0.4375,
                "claim_boundary": "phase2ar_restricted_symbolic_control_not_claim_bearing",
            },
        ),
    )

    assert report["passed"] is True
    assert report["metrics"]["best_control_success_rate"] == 0.4375
    assert report["checks"]["full_minus_best_control_met"] is True


def test_phase2ar_control_delta_rejects_all_zero_controls(tmp_path: Path) -> None:
    report = audit_phase2ar_symbolic_control_delta(
        full_summary_json=_write(
            tmp_path / "full.json",
            {"rows": 16, "successes": 16, "success_rate": 1.0},
        ),
        text_control_summary_json=_write(
            tmp_path / "text.json",
            {
                "rows": 16,
                "successes": 0,
                "success_rate": 0.0,
                "claim_boundary": "phase2ar_restricted_symbolic_control_not_claim_bearing",
            },
        ),
        attribute_control_summary_json=_write(
            tmp_path / "attr.json",
            {
                "rows": 16,
                "successes": 0,
                "success_rate": 0.0,
                "claim_boundary": "phase2ar_restricted_symbolic_control_not_claim_bearing",
            },
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["controls_nonzero"] is False
    assert report["checks"]["best_control_above_floor"] is False
