from __future__ import annotations

import json
from pathlib import Path

from reflexlm.cli.audit_phase2aw_package_loaded_mechanism_matrix import (
    audit_phase2aw_package_loaded_mechanism_matrix,
)


def _summary(
    tmp_path: Path,
    name: str,
    rate: float,
    *,
    rows: int = 100,
    policy_loaded: bool = True,
    qwen_called_rows: int = 0,
) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "artifact_family": "phase2aw_package_loaded_descriptor_execution_runner",
                "rows": rows,
                "successes": round(rate * rows),
                "success_rate": rate,
                "patch_candidate_selection_accuracy": rate,
                "qwen_called_rows": qwen_called_rows,
                "policy_loaded": policy_loaded,
                "claim_boundary": "phase2aw_package_loaded_bounded_descriptor_execution_not_freeform_patch_generation",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_phase2aw_package_loaded_matrix_blocks_native_head_tie(tmp_path: Path) -> None:
    report = audit_phase2aw_package_loaded_mechanism_matrix(
        full_summary_json=_summary(tmp_path, "full", 0.923, qwen_called_rows=100),
        source_overlap_summary_json=_summary(tmp_path, "source", 0.410, policy_loaded=False),
        no_nsi_summary_json=_summary(tmp_path, "no_nsi", 0.314, qwen_called_rows=100),
        native_head_only_summary_json=_summary(tmp_path, "native", 0.923, qwen_called_rows=100),
        continuation_only_summary_json=_summary(tmp_path, "continuation", 0.0),
    )

    assert report["passed"] is False
    assert report["checks"]["full_beats_no_nsi"] is True
    assert report["checks"]["full_beats_native_head_only"] is False
    assert report["full_package_necessity_supported"] is False
    assert report["ready_for_claim_upgrade"] is False


def test_phase2aw_package_loaded_matrix_passes_when_full_beats_controls(tmp_path: Path) -> None:
    report = audit_phase2aw_package_loaded_mechanism_matrix(
        full_summary_json=_summary(tmp_path, "full", 0.93, qwen_called_rows=100),
        source_overlap_summary_json=_summary(tmp_path, "source", 0.40, policy_loaded=False),
        no_nsi_summary_json=_summary(tmp_path, "no_nsi", 0.30, qwen_called_rows=100),
        native_head_only_summary_json=_summary(tmp_path, "native", 0.70, qwen_called_rows=100),
        continuation_only_summary_json=_summary(tmp_path, "continuation", 0.10),
    )

    assert report["passed"] is True
    assert report["ready_for_claim_upgrade"] is True
    assert report["metrics"]["full_minus_native_head_only"] == 0.2300000000000001
