import json
from pathlib import Path

from reflexlm.cli.build_phase2w_production_safety_report import (
    build_phase2w_production_safety_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _data_health(*, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "checks": {
            "phase2s_runtime_flags_present": True,
            "phase2s_required_artifacts_present": True,
            "phase2s_all_split_repos_disjoint": True,
            "phase2s_no_sealed_reference_anywhere": True,
        },
    }


def _postflight(*, low_level: int = 0) -> dict:
    return {
        "passed": True,
        "checks": {"holdout_diagnostics_not_sealed_tuned": True},
        "metrics": {"low_level_qwen_calls_target": low_level},
    }


def test_phase2w_production_safety_report_accepts_sandboxed_safety_gate(
    tmp_path: Path,
) -> None:
    report = build_phase2w_production_safety_report(
        phase2s_data_health_json=_write(tmp_path / "data.json", _data_health()),
        phase2s_full_holdout_postflight_json=_write(
            tmp_path / "postflight.json", _postflight()
        ),
    )
    assert report["passed"] is True
    assert report["unauthorized_write_count"] == 0
    assert report["rollback_success"] == 1.0
    assert "This report does not prove production autonomy." in report["unsupported_claims"]


def test_phase2w_production_safety_report_rejects_low_level_calls(tmp_path: Path) -> None:
    report = build_phase2w_production_safety_report(
        phase2s_data_health_json=_write(tmp_path / "data.json", _data_health()),
        phase2s_full_holdout_postflight_json=_write(
            tmp_path / "postflight.json", _postflight(low_level=1)
        ),
    )
    assert report["passed"] is False
    assert "do_not_use_phase2w_safety_report_for_epoch_gate" in report["blocked_actions"]
