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
    audit_phase2dm_publication_bundle_reproducibility_manifest,
)
from reflexlm.cli.audit_phase2dn_reproducibility_manifest_negative_controls import (
    audit_phase2dn_reproducibility_manifest_negative_controls,
)
from reflexlm.cli.audit_phase2do_reproducibility_manifest_cross_directory_replay import (
    audit_phase2do_reproducibility_manifest_cross_directory_replay,
)
from reflexlm.cli.audit_phase2dp_cross_directory_replay_negative_controls import (
    audit_phase2dp_cross_directory_replay_negative_controls,
)
from reflexlm.cli.audit_phase2dq_replay_bundle_portability_summary import (
    audit_phase2dq_replay_bundle_portability_summary,
)
from reflexlm.cli.audit_phase2dr_portability_summary_negative_controls import (
    audit_phase2dr_portability_summary_negative_controls,
)
from reflexlm.cli.audit_phase2ds_portability_summary_cross_directory_replay import (
    audit_phase2ds_portability_summary_cross_directory_replay,
    validate_phase2ds_portability_summary_cross_directory_replay,
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


def _phase2dr_fixture(tmp_path: Path) -> Path:
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
    phase2dm = audit_phase2dm_publication_bundle_reproducibility_manifest(
        phase2dl_report_json=tmp_path / "phase2dl.json",
        output_manifest_json=tmp_path / "phase2dm_manifest.json",
        output_report_json=tmp_path / "phase2dm.json",
    )
    assert phase2dm["passed"] is True
    phase2dn = audit_phase2dn_reproducibility_manifest_negative_controls(
        phase2dm_report_json=tmp_path / "phase2dm.json",
        output_dir=tmp_path / "dn_controls",
        output_report_json=tmp_path / "phase2dn.json",
    )
    assert phase2dn["passed"] is True
    phase2do = audit_phase2do_reproducibility_manifest_cross_directory_replay(
        phase2dn_report_json=tmp_path / "phase2dn.json",
        output_dir=tmp_path / "do_replay",
        output_report_json=tmp_path / "phase2do.json",
    )
    assert phase2do["passed"] is True
    phase2dp = audit_phase2dp_cross_directory_replay_negative_controls(
        phase2do_report_json=tmp_path / "phase2do.json",
        output_dir=tmp_path / "dp_controls",
        output_report_json=tmp_path / "phase2dp.json",
    )
    assert phase2dp["passed"] is True
    phase2dq = audit_phase2dq_replay_bundle_portability_summary(
        phase2dp_report_json=tmp_path / "phase2dp.json",
        output_report_json=tmp_path / "phase2dq.json",
        output_markdown=tmp_path / "phase2dq.md",
    )
    assert phase2dq["passed"] is True
    phase2dr = audit_phase2dr_portability_summary_negative_controls(
        phase2dq_report_json=tmp_path / "phase2dq.json",
        output_dir=tmp_path / "dr_controls",
        output_report_json=tmp_path / "phase2dr.json",
    )
    assert phase2dr["passed"] is True
    return tmp_path / "phase2dr.json"


def _phase2ds_report(tmp_path: Path) -> dict:
    report = audit_phase2ds_portability_summary_cross_directory_replay(
        phase2dr_report_json=_phase2dr_fixture(tmp_path),
        output_dir=tmp_path / "ds_replay",
        output_report_json=tmp_path / "phase2ds.json",
    )
    assert report["passed"] is True
    return report


def test_phase2ds_accepts_portability_summary_cross_directory_replay(
    tmp_path: Path,
) -> None:
    report = _phase2ds_report(tmp_path)
    validation = validate_phase2ds_portability_summary_cross_directory_replay(report)

    assert validation["passed"] is True
    assert report["metrics"]["replayed_summary_row_count"] >= 5
    assert report["metrics"]["phase2dr_negative_controls_failed"] == report[
        "metrics"
    ]["phase2dr_negative_control_count"]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2dt_portability_summary_replay_negative_controls"
    )


def test_phase2ds_validation_rejects_tampered_replayed_markdown(
    tmp_path: Path,
) -> None:
    report = _phase2ds_report(tmp_path)
    Path(report["evidence"]["replayed_markdown"]).write_text(
        "tampered\n",
        encoding="utf-8",
    )
    validation = validate_phase2ds_portability_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["replayed_markdown_hash_matches_source"] is False


def test_phase2ds_validation_rejects_missing_replayed_report(
    tmp_path: Path,
) -> None:
    report = _phase2ds_report(tmp_path)
    Path(report["evidence"]["replayed_phase2dq_report"]).unlink()
    validation = validate_phase2ds_portability_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["replayed_phase2dq_report_readable"] is False


def test_phase2ds_validation_rejects_non_distinct_replay_dir(
    tmp_path: Path,
) -> None:
    report = _phase2ds_report(tmp_path)
    report["replay_summary"]["replay_markdown_parent"] = report["replay_summary"][
        "source_markdown_parent"
    ]
    validation = validate_phase2ds_portability_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["replay_directory_is_distinct"] is False


def test_phase2ds_validation_rejects_top_level_epoch_claim(
    tmp_path: Path,
) -> None:
    report = _phase2ds_report(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2ds_portability_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
