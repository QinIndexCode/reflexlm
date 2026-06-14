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
    REQUIRED_ARTIFACT_ROLES,
    audit_phase2dk_latex_candidate_publication_bundle,
    validate_phase2dk_latex_candidate_publication_bundle,
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


def _phase2dj_fixture(tmp_path: Path) -> Path:
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
    return tmp_path / "phase2dj.json"


def test_phase2dk_accepts_latex_candidate_publication_bundle(
    tmp_path: Path,
) -> None:
    report = audit_phase2dk_latex_candidate_publication_bundle(
        phase2dj_report_json=_phase2dj_fixture(tmp_path),
        output_dir=tmp_path / "bundle",
        output_report_json=tmp_path / "phase2dk.json",
    )
    validation = validate_phase2dk_latex_candidate_publication_bundle(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["metrics"]["bundle_artifact_count"] == len(REQUIRED_ARTIFACT_ROLES)
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert report["next_required_experiment"] == "phase2dl_publication_bundle_negative_controls"


def test_phase2dk_validation_rejects_missing_bundle_artifact(
    tmp_path: Path,
) -> None:
    report = audit_phase2dk_latex_candidate_publication_bundle(
        phase2dj_report_json=_phase2dj_fixture(tmp_path),
        output_dir=tmp_path / "bundle",
        output_report_json=tmp_path / "phase2dk.json",
    )
    manifest = json.loads(
        Path(report["evidence"]["bundle_manifest"]).read_text(encoding="utf-8")
    )
    latex_entry = next(
        item for item in manifest["artifacts"] if item["role"] == "latex_candidate"
    )
    Path(latex_entry["path"]).unlink()
    validation = validate_phase2dk_latex_candidate_publication_bundle(report)

    assert validation["passed"] is False
    assert validation["checks"]["all_manifest_artifacts_exist"] is False


def test_phase2dk_validation_rejects_tampered_bundle_hash(
    tmp_path: Path,
) -> None:
    report = audit_phase2dk_latex_candidate_publication_bundle(
        phase2dj_report_json=_phase2dj_fixture(tmp_path),
        output_dir=tmp_path / "bundle",
        output_report_json=tmp_path / "phase2dk.json",
    )
    manifest = json.loads(
        Path(report["evidence"]["bundle_manifest"]).read_text(encoding="utf-8")
    )
    readme_entry = next(
        item for item in manifest["artifacts"] if item["role"] == "bundle_readme"
    )
    Path(readme_entry["path"]).write_text("tampered\n", encoding="utf-8")
    validation = validate_phase2dk_latex_candidate_publication_bundle(report)

    assert validation["passed"] is False
    assert validation["checks"]["all_manifest_hashes_match"] is False


def test_phase2dk_validation_rejects_missing_readme_boundary(
    tmp_path: Path,
) -> None:
    report = audit_phase2dk_latex_candidate_publication_bundle(
        phase2dj_report_json=_phase2dj_fixture(tmp_path),
        output_dir=tmp_path / "bundle",
        output_report_json=tmp_path / "phase2dk.json",
    )
    manifest = json.loads(
        Path(report["evidence"]["bundle_manifest"]).read_text(encoding="utf-8")
    )
    readme_entry = next(
        item for item in manifest["artifacts"] if item["role"] == "bundle_readme"
    )
    Path(readme_entry["path"]).write_text("Phase2DK bundle\n", encoding="utf-8")
    validation = validate_phase2dk_latex_candidate_publication_bundle(report)

    assert validation["passed"] is False
    assert validation["checks"]["readme_contains_bounded_boundary"] is False


def test_phase2dk_validation_rejects_top_level_epoch_claim(
    tmp_path: Path,
) -> None:
    report = audit_phase2dk_latex_candidate_publication_bundle(
        phase2dj_report_json=_phase2dj_fixture(tmp_path),
        output_dir=tmp_path / "bundle",
        output_report_json=tmp_path / "phase2dk.json",
    )
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2dk_latex_candidate_publication_bundle(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
