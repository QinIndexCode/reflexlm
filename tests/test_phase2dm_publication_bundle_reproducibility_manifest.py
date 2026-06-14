import json
from pathlib import Path

from reflexlm.cli.audit_phase2de_compact_evidence_rollup import REPORT_SPECS
from reflexlm.cli.audit_phase2dg_compact_rollup_publication_table import (
    audit_phase2dg_compact_rollup_publication_table,
)
from reflexlm.cli.audit_phase2dh_publication_table_negative_controls import (
    audit_phase2dh_publication_table_negative_controls,
)
from reflexlm.cli.audit_phase2di_publication_table_latex_candidate import (
    audit_phase2di_publication_table_latex_candidate,
)
from reflexlm.cli.audit_phase2dj_latex_candidate_negative_controls import (
    audit_phase2dj_latex_candidate_negative_controls,
)
from reflexlm.cli.audit_phase2dk_latex_candidate_publication_bundle import (
    audit_phase2dk_latex_candidate_publication_bundle,
)
from reflexlm.cli.audit_phase2dl_publication_bundle_negative_controls import (
    audit_phase2dl_publication_bundle_negative_controls,
)
from reflexlm.cli.audit_phase2dm_publication_bundle_reproducibility_manifest import (
    REQUIRED_REPRODUCTION_STEPS,
    REQUIRED_SOURCE_REPORT_ROLES,
    audit_phase2dm_publication_bundle_reproducibility_manifest,
    validate_phase2dm_publication_bundle_reproducibility_manifest,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _phase_row(spec: dict[str, str]) -> dict:
    return {
        "phase_id": spec["phase_id"],
        "category": spec["category"],
        "artifact_family": f"{spec['phase_id']}_artifact",
        "passed": True,
        "bounded_claim_ok": True,
        "compact_metrics": {
            "runtime_count": 3,
            "control_count": 10,
            "negative_controls_failed": 9,
        },
        "next_required_experiment": f"{spec['phase_id']}_next",
    }


def _phase2df_fixture(tmp_path: Path) -> Path:
    phase2de = _write(
        tmp_path / "phase2de.json",
        {
            "passed": True,
            "metrics": {
                "phase_count": len(REPORT_SPECS),
                "positive_phase_count": 8,
                "negative_control_phase_count": 7,
            },
            "phase_results": [_phase_row(spec) for spec in REPORT_SPECS],
        },
    )
    return _write(
        tmp_path / "phase2df.json",
        {
            "passed": True,
            "evidence": {"phase2de_report_json": str(phase2de)},
        },
    )


def _phase2dl_fixture(tmp_path: Path) -> Path:
    phase2dg = audit_phase2dg_compact_rollup_publication_table(
        phase2df_report_json=_phase2df_fixture(tmp_path),
        output_report_json=tmp_path / "phase2dg.json",
        output_markdown=tmp_path / "phase2dg.md",
    )
    assert phase2dg["passed"] is True
    phase2dh = audit_phase2dh_publication_table_negative_controls(
        phase2dg_report_json=tmp_path / "phase2dg.json",
        output_dir=tmp_path / "dh_controls",
        output_report_json=tmp_path / "phase2dh.json",
    )
    assert phase2dh["passed"] is True
    phase2di = audit_phase2di_publication_table_latex_candidate(
        phase2dh_report_json=tmp_path / "phase2dh.json",
        output_report_json=tmp_path / "phase2di.json",
        output_latex=tmp_path / "phase2di.tex",
    )
    assert phase2di["passed"] is True
    phase2dj = audit_phase2dj_latex_candidate_negative_controls(
        phase2di_report_json=tmp_path / "phase2di.json",
        output_dir=tmp_path / "dj_controls",
        output_report_json=tmp_path / "phase2dj.json",
    )
    assert phase2dj["passed"] is True
    phase2dk = audit_phase2dk_latex_candidate_publication_bundle(
        phase2dj_report_json=tmp_path / "phase2dj.json",
        output_dir=tmp_path / "dk_bundle",
        output_report_json=tmp_path / "phase2dk.json",
    )
    assert phase2dk["passed"] is True
    phase2dl = audit_phase2dl_publication_bundle_negative_controls(
        phase2dk_report_json=tmp_path / "phase2dk.json",
        output_dir=tmp_path / "dl_controls",
        output_report_json=tmp_path / "phase2dl.json",
    )
    assert phase2dl["passed"] is True
    return tmp_path / "phase2dl.json"


def _phase2dm_report(tmp_path: Path) -> dict:
    report = audit_phase2dm_publication_bundle_reproducibility_manifest(
        phase2dl_report_json=_phase2dl_fixture(tmp_path),
        output_manifest_json=tmp_path / "phase2dm_manifest.json",
        output_report_json=tmp_path / "phase2dm.json",
    )
    assert report["passed"] is True
    return report


def test_phase2dm_accepts_reproducibility_manifest(tmp_path: Path) -> None:
    report = _phase2dm_report(tmp_path)
    validation = validate_phase2dm_publication_bundle_reproducibility_manifest(
        report
    )

    assert validation["passed"] is True
    assert report["metrics"]["source_report_count"] == len(REQUIRED_SOURCE_REPORT_ROLES)
    assert report["metrics"]["reproduction_step_count"] == len(
        REQUIRED_REPRODUCTION_STEPS
    )
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2dn_reproducibility_manifest_negative_controls"
    )


def test_phase2dm_validation_rejects_missing_source_report(
    tmp_path: Path,
) -> None:
    report = _phase2dm_report(tmp_path)
    manifest = json.loads(
        Path(report["evidence"]["reproducibility_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    source_entry = next(
        item for item in manifest["source_reports"] if item["role"] == "phase2di_report"
    )
    Path(source_entry["path"]).unlink()
    validation = validate_phase2dm_publication_bundle_reproducibility_manifest(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["source_reports_exist"] is False


def test_phase2dm_validation_rejects_tampered_bundle_artifact(
    tmp_path: Path,
) -> None:
    report = _phase2dm_report(tmp_path)
    manifest = json.loads(
        Path(report["evidence"]["reproducibility_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    artifact = next(
        item for item in manifest["bundle_artifacts"] if item["role"] == "bundle_readme"
    )
    Path(artifact["path"]).write_text("tampered\n", encoding="utf-8")
    validation = validate_phase2dm_publication_bundle_reproducibility_manifest(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["bundle_artifact_hashes_match"] is False


def test_phase2dm_validation_rejects_missing_reproduction_step(
    tmp_path: Path,
) -> None:
    report = _phase2dm_report(tmp_path)
    manifest_path = Path(report["evidence"]["reproducibility_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reproduction_steps"] = manifest["reproduction_steps"][:-1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_phase2dm_publication_bundle_reproducibility_manifest(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["reproduction_steps_complete"] is False


def test_phase2dm_validation_rejects_missing_boundary(tmp_path: Path) -> None:
    report = _phase2dm_report(tmp_path)
    manifest_path = Path(report["evidence"]["reproducibility_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["claim_boundary"] = "epoch-making architecture"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_phase2dm_publication_bundle_reproducibility_manifest(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["manifest_contains_bounded_boundary"] is False


def test_phase2dm_validation_rejects_top_level_epoch_claim(
    tmp_path: Path,
) -> None:
    report = _phase2dm_report(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2dm_publication_bundle_reproducibility_manifest(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
