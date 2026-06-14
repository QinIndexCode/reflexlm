import json
from pathlib import Path

from reflexlm.cli.audit_phase2de_compact_evidence_rollup import REPORT_SPECS
from reflexlm.cli.audit_phase2df_compact_rollup_negative_controls import (
    audit_phase2df_compact_rollup_negative_controls,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _phase_row(spec: dict[str, str]) -> dict:
    return {
        "phase_id": spec["phase_id"],
        "category": spec["category"],
        "report_json": f"reports/{spec['filename']}",
        "readable": True,
        "passed": True,
        "artifact_family": f"{spec['phase_id']}_artifact",
        "bounded_claim_ok": True,
        "ready_flags": {
            f"ready_for_bounded_{spec['phase_id']}_claim": True,
            "ready_for_general_shell_autonomy_claim": False,
            "ready_for_general_runtime_invariance_claim": False,
            "ready_for_open_ended_native_perception_claim": False,
            "ready_for_production_autonomy_claim": False,
            "ready_for_epoch_making_architecture_claim": False,
        },
        "compact_metrics": {
            "runtime_count": 3,
            "control_count": 10,
        },
        "next_required_experiment": f"{spec['phase_id']}_next",
    }


def _phase2de_fixture(tmp_path: Path) -> Path:
    rows = [_phase_row(spec) for spec in REPORT_SPECS]
    return _write(
        tmp_path / "phase2de.json",
        {
            "artifact_family": "phase2de_compact_evidence_rollup",
            "passed": True,
            "ready_for_bounded_compact_evidence_rollup_claim": True,
            "ready_for_general_shell_autonomy_claim": False,
            "ready_for_general_runtime_invariance_claim": False,
            "ready_for_open_ended_native_perception_claim": False,
            "ready_for_production_autonomy_claim": False,
            "ready_for_epoch_making_architecture_claim": False,
            "checks": {
                "source_phase2dd_path_matches_expected_filename": True,
                "all_expected_phase_reports_present": True,
                "all_phase_reports_readable": True,
                "all_phase_reports_passed": True,
                "all_phase_claims_are_bounded": True,
                "positive_and_negative_evidence_present": True,
                "phase_order_matches_expected_chain": True,
            },
            "metrics": {
                "phase_count": len(rows),
                "positive_phase_count": 8,
                "negative_control_phase_count": 7,
                "passed_phase_count": len(rows),
                "bounded_phase_count": len(rows),
            },
            "phase_results": rows,
        },
    )


def test_phase2df_rejects_compact_rollup_negative_controls(tmp_path: Path) -> None:
    report = audit_phase2df_compact_rollup_negative_controls(
        phase2de_report_json=_phase2de_fixture(tmp_path),
        output_dir=tmp_path / "controls",
        output_report_json=tmp_path / "phase2df.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 10
    assert all(
        row["expected_failed_checks_observed"]
        for row in report["control_results"]
    )
