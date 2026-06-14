import json
from pathlib import Path

from reflexlm.cli.audit_phase2_sidecar_package_readiness import (
    audit_phase2_sidecar_package_readiness,
)


def _write(path: Path, payload: dict | str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _boundary() -> dict:
    return {"passed": True, "strong_architecture_claim_ready": False}


def _phase2am() -> dict:
    return {
        "passed": True,
        "natural_repo_disjoint_sidecar_dependency_reproduced": True,
        "metrics": {
            "observed_min_full_minus_source_overlap": 0.46,
            "observed_min_full_minus_sidecar_erased": 0.47,
            "observed_min_full_minus_wrong_sidecar": 0.58,
        },
    }


def _phase2ap(*, pure: bool) -> dict:
    return {
        "passed": True,
        "stable_bounded_sidecar_control_supported": True,
        "strict_pure_sidecar_claim_ready": pure,
    }


def _phase2ae() -> dict:
    return {
        "passed": True,
        "checks": {
            "provenance_audit_passed": True,
            "structural_sidecar_holdout_solves": True,
            "stripped_identity_holdout_does_not_solve": True,
            "full_beats_policyless_budget": True,
            "erased_structural_counterfactual_fails": True,
            "wrong_structural_counterfactual_fails": True,
        },
    }


def test_sidecar_package_readiness_blocks_without_package_runtime_and_pure_control(
    tmp_path: Path,
) -> None:
    report = audit_phase2_sidecar_package_readiness(
        architecture_boundary_json=_write(tmp_path / "boundary.json", _boundary()),
        phase2am_reproduction_json=_write(tmp_path / "phase2am.json", _phase2am()),
        phase2ap_control_synthesis_json=_write(
            tmp_path / "phase2ap.json", _phase2ap(pure=False)
        ),
    )

    assert report["passed"] is False
    assert report["ready_for_package"] is False
    assert "strict_pure_sidecar_causality_not_ready" in report["blockers"]
    assert "package_manifest_missing" in report["blockers"]
    assert "package_runtime_execution_missing" in report["blockers"]


def test_sidecar_package_readiness_passes_only_with_runtime_artifacts_and_pure_control(
    tmp_path: Path,
) -> None:
    report = audit_phase2_sidecar_package_readiness(
        architecture_boundary_json=_write(tmp_path / "boundary.json", _boundary()),
        phase2am_reproduction_json=_write(tmp_path / "phase2am.json", _phase2am()),
        phase2ap_control_synthesis_json=_write(
            tmp_path / "phase2ap.json", _phase2ap(pure=True)
        ),
        package_manifest_json=_write(tmp_path / "native_nervous_package.json", {}),
        package_runtime_execution_jsonl=_write(
            tmp_path / "execution.jsonl", '{"success": true}\n'
        ),
    )

    assert report["passed"] is True
    assert report["ready_for_package"] is True
    assert report["blockers"] == []


def test_sidecar_package_readiness_can_use_phase2ae_package_runtime_sidecar_gate(
    tmp_path: Path,
) -> None:
    report = audit_phase2_sidecar_package_readiness(
        architecture_boundary_json=_write(tmp_path / "boundary.json", _boundary()),
        phase2am_reproduction_json=_write(tmp_path / "phase2am.json", _phase2am()),
        phase2ap_control_synthesis_json=_write(
            tmp_path / "phase2ap.json", _phase2ap(pure=False)
        ),
        phase2ae_structural_sidecar_comparison_json=_write(
            tmp_path / "phase2ae.json", _phase2ae()
        ),
        package_manifest_json=_write(tmp_path / "native_nervous_package.json", {}),
        package_runtime_execution_jsonl=_write(
            tmp_path / "execution.jsonl", '{"success": true}\n'
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["phase2ap_strict_pure_sidecar_ready"] is False
    assert report["checks"]["phase2ae_package_runtime_sidecar_supported"] is True
    assert report["checks"]["strict_sidecar_ready_from_ap_or_ae"] is True
    assert report["ready_for_sealed_eval"] is False
