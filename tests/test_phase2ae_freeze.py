import json
from pathlib import Path

from reflexlm.cli.freeze_phase2ae_structural_sidecar import (
    build_phase2ae_structural_sidecar_freeze_manifest,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2ae_freeze_records_control_leak_boundary(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_json(
        reports
        / "phase2ae_prov_holdout45_qwen3b_v2_slotbalanced_noretry.runtimefix.execution_summary.json",
        {"rows": 45, "correct_patch_candidate_selections": 45, "patch_candidate_selection_accuracy": 1.0},
    )
    _write_json(
        reports / "phase2ae_qwen3b_v2_slotbalanced_noretry_runtimefix_learning_gap_audit.json",
        {"passed": True},
    )
    _write_json(
        reports
        / "phase2ae_prov_holdout45_qwen3b_v2_slotbalanced_noretry.runtimefix.no_nsi_textablated.execution_summary.json",
        {"rows": 45, "correct_patch_candidate_selections": 32, "patch_candidate_selection_accuracy": 32 / 45},
    )
    _write_json(
        reports
        / "phase2ae_qwen3b_v2_slotbalanced_noretry_runtimefix_no_nsi_textablated_learning_gap_audit.json",
        {
            "passed": True,
            "checks": {"initial_policy_selection_supports_learned_head_claim": False},
        },
    )
    _write_json(
        reports / "phase2ae_qwen3b_v2_slotbalanced_runtimefix_full_vs_no_nsi_textablated_baseline_report.json",
        {
            "baseline_metrics": {
                "source_overlap": {"accuracy": 1.0},
                "source_overlap_identity_text_ablated": {"accuracy": 0.0},
                "runtime_identity_heuristic": {"accuracy": 1.0},
            },
            "by_expected_slot": {"3": {"total": 13, "runtime_identity_heuristic": 13}},
        },
    )
    _write_json(reports / "phase2ae_qwen3b_v2_slotbalanced_slot_support_audit.json", {"passed": True})
    _write_json(
        reports / "phase2ae_structural_sidecar_qwen3b_v2_slotbalanced.training_summary.json",
        {"history": [{"val_metrics": {"command_slot_accuracy": 1.0}}]},
    )
    package = tmp_path / "pkg"
    no_nsi = tmp_path / "pkg_no_nsi"
    _write_json(package / "native_nervous_package.json", {"policy_label": "full"})
    _write_json(no_nsi / "native_nervous_package.json", {"policy_label": "no_nsi"})

    manifest = build_phase2ae_structural_sidecar_freeze_manifest(
        report_dir=reports,
        package_dir=package,
        no_nsi_package_dir=no_nsi,
    )

    assert manifest["frozen"] is True
    assert manifest["checks"]["full_execution_45_of_45"] is True
    assert manifest["checks"]["training_summary_passed_val_gate"] is True
    assert manifest["checks"]["no_nsi_textablated_below_full"] is True
    assert manifest["checks"]["raw_source_overlap_contaminated_by_identity_sidecar"] is True
    assert manifest["checks"]["runtime_identity_heuristic_solves_split"] is True
    assert "freeform patch generation" in manifest["unsupported_claims"]
