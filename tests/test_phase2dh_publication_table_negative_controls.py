import json
from pathlib import Path

from reflexlm.cli.audit_phase2de_compact_evidence_rollup import REPORT_SPECS
from reflexlm.cli.audit_phase2dg_compact_rollup_publication_table import (
    audit_phase2dg_compact_rollup_publication_table,
)
from reflexlm.cli.audit_phase2dh_publication_table_negative_controls import (
    audit_phase2dh_publication_table_negative_controls,
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


def _phase2dg_fixture(tmp_path: Path) -> Path:
    report = audit_phase2dg_compact_rollup_publication_table(
        phase2df_report_json=_phase2df_fixture(tmp_path),
        output_report_json=tmp_path / "phase2dg.json",
        output_markdown=tmp_path / "phase2dg.md",
    )
    assert report["passed"] is True
    return tmp_path / "phase2dg.json"


def test_phase2dh_rejects_publication_table_negative_controls(
    tmp_path: Path,
) -> None:
    report = audit_phase2dh_publication_table_negative_controls(
        phase2dg_report_json=_phase2dg_fixture(tmp_path),
        output_dir=tmp_path / "controls",
        output_report_json=tmp_path / "phase2dh.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 13
    assert all(
        row["expected_failed_checks_observed"]
        for row in report["control_results"]
    )
