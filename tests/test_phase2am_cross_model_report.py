import json
from pathlib import Path

from reflexlm.cli.build_phase2am_cross_model_report import build_phase2am_cross_model_report


def _write(path: Path, delta: float, passed: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "metrics": {
                    "full_accuracy": 0.99,
                    "source_overlap_accuracy": 0.53,
                    "sidecar_erased_accuracy": 0.50,
                    "wrong_sidecar_accuracy": 0.40,
                    "full_minus_source_overlap": 0.46,
                    "full_minus_sidecar_erased": delta,
                    "full_minus_wrong_sidecar": 0.59,
                    "row_count": 112,
                    "repo_count": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_phase2am_cross_model_report_accepts_two_models(tmp_path: Path) -> None:
    report = build_phase2am_cross_model_report(
        model_reports=[_write(tmp_path / "a.json", 0.47), _write(tmp_path / "b.json", 0.67)]
    )

    assert report["passed"] is True
    assert report["metrics"]["model_count"] == 2
    assert report["metrics"]["min_full_minus_sidecar_erased"] == 0.47
    assert "do_not_claim_sealed_transfer_from_phase2am" in report["blocked_actions"]


def test_phase2am_cross_model_report_rejects_one_model(tmp_path: Path) -> None:
    report = build_phase2am_cross_model_report(model_reports=[_write(tmp_path / "a.json", 0.47)])

    assert report["passed"] is False
    assert report["checks"]["at_least_two_models"] is False


def test_phase2am_cross_model_report_rejects_failed_model_report(tmp_path: Path) -> None:
    report = build_phase2am_cross_model_report(
        model_reports=[
            _write(tmp_path / "a.json", 0.47),
            _write(tmp_path / "b.json", 0.67, passed=False),
        ]
    )

    assert report["passed"] is False
    assert report["checks"]["all_model_reports_passed"] is False
