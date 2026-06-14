import json
from pathlib import Path

from reflexlm.cli.audit_phase2af_hardened_structural_sidecar import (
    audit_phase2af_hardened_structural_sidecar,
)


def _write_report(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _baseline(
    *,
    full: float = 0.92,
    no_nsi: float = 0.70,
    raw_source: float = 0.50,
    text_ablated_source: float = 0.35,
    identity_heuristic: float = 0.60,
) -> dict:
    return {
        "full_summary": {"patch_candidate_selection_accuracy": full},
        "no_nsi_summary": {"patch_candidate_selection_accuracy": no_nsi},
        "baseline_metrics": {
            "source_overlap": {"accuracy": raw_source},
            "source_overlap_identity_text_ablated": {"accuracy": text_ablated_source},
            "runtime_identity_heuristic": {"accuracy": identity_heuristic},
        },
    }


def test_phase2af_gate_accepts_graded_hardened_split(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path / "baseline.json", _baseline())

    report = audit_phase2af_hardened_structural_sidecar(
        baseline_report_json=report_path,
    )

    assert report["passed"] is True
    assert report["blocked_actions"] == []


def test_phase2af_gate_rejects_raw_source_overlap_ceiling(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path / "baseline.json", _baseline(raw_source=1.0))

    report = audit_phase2af_hardened_structural_sidecar(
        baseline_report_json=report_path,
    )

    assert report["passed"] is False
    assert report["checks"]["raw_source_overlap_not_ceiling"] is False
    assert "do_not_train_phase2af_full" in report["blocked_actions"]


def test_phase2af_gate_rejects_identity_heuristic_solved_split(tmp_path: Path) -> None:
    report_path = _write_report(
        tmp_path / "baseline.json",
        _baseline(identity_heuristic=1.0),
    )

    report = audit_phase2af_hardened_structural_sidecar(
        baseline_report_json=report_path,
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_identity_heuristic_not_sufficient_alone"] is False


def test_phase2af_gate_rejects_all_zero_ablated_source_overlap(tmp_path: Path) -> None:
    report_path = _write_report(
        tmp_path / "baseline.json",
        _baseline(text_ablated_source=0.0),
    )

    report = audit_phase2af_hardened_structural_sidecar(
        baseline_report_json=report_path,
    )

    assert report["passed"] is False
    assert report["checks"]["identity_text_ablated_source_overlap_nonzero_feasible"] is False
