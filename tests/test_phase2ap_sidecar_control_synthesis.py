import json
from pathlib import Path

from reflexlm.cli.build_phase2ap_sidecar_control_synthesis import (
    build_phase2ap_sidecar_control_synthesis,
)


def _an(path: Path, *, pure: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "pure_sidecar_dependency_supported": pure,
                "metrics": {
                    "neutral_original_accuracy": 1.0,
                    "neutral_original_minus_erased": 0.7,
                    "neutral_original_minus_wrong": 1.0,
                    "erased_residual_above_source": not pure,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _ao(path: Path, *, strict: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "passed": strict,
                "sidecar_order_robustness_supported": True,
                "strict_erased_residual_control_passed": strict,
                "metrics": {
                    "original_permuted_accuracy": 0.99,
                    "original_minus_erased_permuted": 0.6,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_phase2ap_synthesis_accepts_bounded_with_strict_failure(tmp_path: Path) -> None:
    report = build_phase2ap_sidecar_control_synthesis(
        phase2an_reports=[_an(tmp_path / "qwen3b_seed1_an.json")],
        phase2ao_reports=[_ao(tmp_path / "qwen3b_seed1_ao.json", strict=False)],
    )

    assert report["passed"] is True
    assert report["stable_bounded_sidecar_control_supported"] is True
    assert report["strict_pure_sidecar_claim_ready"] is False
    assert report["checks"]["all_strict_erased_residual_controls_passed"] is False


def test_phase2ap_synthesis_rejects_missing_order_support(tmp_path: Path) -> None:
    ao = tmp_path / "qwen3b_seed1_ao.json"
    ao.write_text(
        json.dumps(
            {
                "passed": False,
                "sidecar_order_robustness_supported": False,
                "strict_erased_residual_control_passed": False,
                "metrics": {"original_permuted_accuracy": 0.4},
            }
        ),
        encoding="utf-8",
    )
    report = build_phase2ap_sidecar_control_synthesis(
        phase2an_reports=[_an(tmp_path / "qwen3b_seed1_an.json")],
        phase2ao_reports=[ao],
    )

    assert report["passed"] is False
    assert report["stable_bounded_sidecar_control_supported"] is False
