import json
from pathlib import Path

from reflexlm.cli.audit_phase2de_compact_evidence_rollup import REPORT_SPECS
from reflexlm.cli.audit_phase2dg_compact_rollup_publication_table import (
    TABLE_COLUMNS,
    audit_phase2dg_compact_rollup_publication_table,
    validate_phase2dg_compact_rollup_publication_table,
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


def test_phase2dg_accepts_publication_table(tmp_path: Path) -> None:
    report = audit_phase2dg_compact_rollup_publication_table(
        phase2df_report_json=_phase2df_fixture(tmp_path),
        output_report_json=tmp_path / "phase2dg.json",
        output_markdown=tmp_path / "phase2dg.md",
    )
    validation = validate_phase2dg_compact_rollup_publication_table(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["table"]["columns"] == list(TABLE_COLUMNS)
    assert report["metrics"]["row_count"] == len(REPORT_SPECS)
    assert Path(report["evidence"]["publication_table_markdown"]).exists()
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_phase2dg_validation_rejects_missing_markdown(tmp_path: Path) -> None:
    report = audit_phase2dg_compact_rollup_publication_table(
        phase2df_report_json=_phase2df_fixture(tmp_path),
        output_report_json=tmp_path / "phase2dg.json",
        output_markdown=tmp_path / "phase2dg.md",
    )
    Path(report["evidence"]["publication_table_markdown"]).unlink()
    validation = validate_phase2dg_compact_rollup_publication_table(report)

    assert validation["passed"] is False
    assert validation["checks"]["markdown_table_readable"] is False


def test_phase2dg_validation_rejects_unbounded_row(tmp_path: Path) -> None:
    report = audit_phase2dg_compact_rollup_publication_table(
        phase2df_report_json=_phase2df_fixture(tmp_path),
        output_report_json=tmp_path / "phase2dg.json",
        output_markdown=tmp_path / "phase2dg.md",
    )
    report["table"]["rows"][0]["bounded_claim_ok"] = False
    validation = validate_phase2dg_compact_rollup_publication_table(report)

    assert validation["passed"] is False
    assert validation["checks"]["all_rows_passed_and_bounded"] is False


def test_phase2dg_validation_rejects_top_level_epoch_claim(tmp_path: Path) -> None:
    report = audit_phase2dg_compact_rollup_publication_table(
        phase2df_report_json=_phase2df_fixture(tmp_path),
        output_report_json=tmp_path / "phase2dg.json",
        output_markdown=tmp_path / "phase2dg.md",
    )
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2dg_compact_rollup_publication_table(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
