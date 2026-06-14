import json
from pathlib import Path

from reflexlm.cli.build_phase2am_reproduction_report import (
    build_phase2am_reproduction_report,
)


def _write(path: Path, *, model: str, delta_erased: float, seed: int | None = None) -> Path:
    seed_part = f"_seed{seed}" if seed is not None else ""
    report_path = path / f"phase2am_natural_v12_{model}{seed_part}_sidecar_postflight.json"
    report_path.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "full_accuracy": 0.99,
                    "source_overlap_accuracy": 0.53,
                    "sidecar_erased_accuracy": 0.45,
                    "wrong_sidecar_accuracy": 0.35,
                    "full_minus_source_overlap": 0.46,
                    "full_minus_sidecar_erased": delta_erased,
                    "full_minus_wrong_sidecar": 0.59,
                    "row_count": 112,
                    "repo_count": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    return report_path


def test_phase2am_reproduction_report_accepts_multi_model_and_seed(tmp_path: Path) -> None:
    report = build_phase2am_reproduction_report(
        postflight_reports=[
            _write(tmp_path, model="qwen3b", delta_erased=0.67),
            _write(tmp_path, model="qwen7b", delta_erased=0.47),
            _write(tmp_path, model="qwen3b", delta_erased=0.50, seed=20260531),
        ]
    )

    assert report["passed"] is True
    assert report["metrics"]["run_count"] == 3
    assert report["metrics"]["model_count"] == 2
    assert report["metrics"]["seed_replication_count"] == 1
    assert report["metrics"]["observed_min_full_minus_sidecar_erased"] == 0.47
    assert report["metrics"]["passed_min_full_minus_sidecar_erased"] == 0.47


def test_phase2am_reproduction_report_rejects_missing_seed_replication(tmp_path: Path) -> None:
    report = build_phase2am_reproduction_report(
        postflight_reports=[
            _write(tmp_path, model="qwen3b", delta_erased=0.67),
            _write(tmp_path, model="qwen7b", delta_erased=0.47),
            _write(tmp_path, model="qwen15b", delta_erased=0.50),
        ]
    )

    assert report["passed"] is False
    assert report["checks"]["has_seed_replication"] is False


def test_phase2am_reproduction_report_rejects_weak_sidecar_delta(tmp_path: Path) -> None:
    report = build_phase2am_reproduction_report(
        postflight_reports=[
            _write(tmp_path, model="qwen3b", delta_erased=0.67),
            _write(tmp_path, model="qwen7b", delta_erased=0.24),
            _write(tmp_path, model="qwen3b", delta_erased=0.50, seed=20260531),
        ]
    )

    assert report["passed"] is False
    assert report["checks"]["observed_min_full_minus_erased_ge_0_25"] is False
