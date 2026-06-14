from pathlib import Path

from reflexlm.cli.build_phase2w_live_agent_missing_report import (
    build_phase2w_live_agent_missing_report,
)


def test_phase2w_live_agent_missing_report_is_not_evidence(tmp_path: Path) -> None:
    report = build_phase2w_live_agent_missing_report(
        intended_results_jsonl=tmp_path / "results.jsonl",
        intended_config_json=tmp_path / "config.json",
    )
    assert report["passed"] is False
    assert report["claim_boundary"] == "missing_report_only_not_evidence"
    assert "do_not_treat_static_overlap_baseline_as_live_agent" in report["blocked_actions"]
    assert "model_or_provider" in report["required_config_fields"]
    assert "trace_archive_uri" in report["required_config_fields"]
    assert "transcript_sha256" in report["required_row_fields"]
