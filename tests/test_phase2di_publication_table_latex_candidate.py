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
    LATEX_COLUMNS,
    audit_phase2di_publication_table_latex_candidate,
    validate_phase2di_publication_table_latex_candidate,
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


def _phase2dh_fixture(tmp_path: Path) -> Path:
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
    return tmp_path / "phase2dh.json"


def test_phase2di_accepts_publication_table_latex_candidate(
    tmp_path: Path,
) -> None:
    report = audit_phase2di_publication_table_latex_candidate(
        phase2dh_report_json=_phase2dh_fixture(tmp_path),
        output_report_json=tmp_path / "phase2di.json",
        output_latex=tmp_path / "phase2di.tex",
    )
    validation = validate_phase2di_publication_table_latex_candidate(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["table_summary"]["columns"] == list(LATEX_COLUMNS)
    assert report["metrics"]["row_count"] == len(REPORT_SPECS)
    assert Path(report["evidence"]["latex_candidate_path"]).exists()
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert report["next_required_experiment"] == "phase2dj_latex_candidate_negative_controls"


def test_phase2di_validation_rejects_missing_latex_candidate(
    tmp_path: Path,
) -> None:
    report = audit_phase2di_publication_table_latex_candidate(
        phase2dh_report_json=_phase2dh_fixture(tmp_path),
        output_report_json=tmp_path / "phase2di.json",
        output_latex=tmp_path / "phase2di.tex",
    )
    Path(report["evidence"]["latex_candidate_path"]).unlink()
    validation = validate_phase2di_publication_table_latex_candidate(report)

    assert validation["passed"] is False
    assert validation["checks"]["latex_candidate_readable"] is False


def test_phase2di_validation_rejects_latex_missing_phase_id(
    tmp_path: Path,
) -> None:
    report = audit_phase2di_publication_table_latex_candidate(
        phase2dh_report_json=_phase2dh_fixture(tmp_path),
        output_report_json=tmp_path / "phase2di.json",
        output_latex=tmp_path / "phase2di.tex",
    )
    latex_path = Path(report["evidence"]["latex_candidate_path"])
    first_phase_id = report["table_summary"]["rows"][0]["phase_id"]
    latex_path.write_text(
        latex_path.read_text(encoding="utf-8").replace(first_phase_id, "phase2xx"),
        encoding="utf-8",
    )
    validation = validate_phase2di_publication_table_latex_candidate(report)

    assert validation["passed"] is False
    assert validation["checks"]["latex_contains_all_phase_ids"] is False


def test_phase2di_rejects_output_under_main_paper_tables_dir(
    tmp_path: Path,
) -> None:
    report = audit_phase2di_publication_table_latex_candidate(
        phase2dh_report_json=_phase2dh_fixture(tmp_path),
        output_report_json=tmp_path / "phase2di.json",
        output_latex=tmp_path / "docs" / "paper_b" / "tables" / "bad.tex",
    )
    validation = validate_phase2di_publication_table_latex_candidate(report)

    assert report["passed"] is False
    assert report["checks"]["latex_candidate_not_in_main_tables_dir"] is False
    assert validation["passed"] is False
    assert validation["checks"]["latex_candidate_not_in_main_tables_dir"] is False


def test_phase2di_validation_rejects_top_level_epoch_claim(
    tmp_path: Path,
) -> None:
    report = audit_phase2di_publication_table_latex_candidate(
        phase2dh_report_json=_phase2dh_fixture(tmp_path),
        output_report_json=tmp_path / "phase2di.json",
        output_latex=tmp_path / "phase2di.tex",
    )
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2di_publication_table_latex_candidate(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
