from __future__ import annotations

import json
from pathlib import Path

from reflexlm.cli.build_phase2aw_architecture_evidence_synthesis import (
    build_phase2aw_architecture_evidence_synthesis,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2aw_architecture_synthesis_keeps_claim_bounded(tmp_path: Path) -> None:
    sealed = _write(
        tmp_path / "sealed.json",
        {
            "passed": True,
            "claim_scope": "phase2aw_bounded_sealed_v3_package_transfer_positive",
            "ready_for_epoch_making_architecture_claim": False,
            "metrics": {
                "full_completion": 1.0,
                "no_nsi_completion": 0.265625,
                "native_head_only_completion": 0.0,
                "continuation_only_completion": 0.0,
                "full_minus_no_nsi": 0.734375,
                "full_minus_native_head_only": 1.0,
            },
        },
    )
    matrix = _write(
        tmp_path / "matrix.json",
        {
            "artifact_family": "phase2aw_package_loaded_mechanism_matrix",
            "claim_scope": "phase2aw_package_loaded_partial_mechanism_evidence",
            "checks": {"full_beats_native_head_only": False},
            "metrics": {
                "full_success_rate": 0.923,
                "native_head_only_success_rate": 0.923,
                "full_minus_native_head_only": 0.0,
                "full_minus_no_nsi": 0.609,
            },
        },
    )
    phase2k = _write(
        tmp_path / "phase2k.json",
        {
            "checks": {
                "full_nonsealed_beats_native_head_only": True,
                "sealed_gate_failed": True,
            },
            "metrics": {"nonsealed": {"full_minus_native_head_only": 0.5}},
        },
    )
    phase2l = _write(
        tmp_path / "phase2l.json",
        {
            "checks": {"package_postflight_passed": True, "sealed_gate_failed": True},
            "metrics": {"nonsealed_package": {"full_minus_native_head_only": 0.5}},
        },
    )

    report = build_phase2aw_architecture_evidence_synthesis(
        phase2aw_sealed_report_json=sealed,
        phase2aw_package_matrix_json=matrix,
        phase2k_freeze_json=phase2k,
        phase2l_freeze_json=phase2l,
    )

    assert report["passed"] is True
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert "full package necessity for open-repair execution" in report["unsupported_claims"]
    assert report["next_required_experiment"]["phase"] == (
        "phase2ax_package_loaded_counterfactual_repair"
    )
