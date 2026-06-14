import json
from pathlib import Path

from reflexlm.cli.audit_phase2ao_order_permutation_postflight import (
    audit_phase2ao_order_permutation_postflight,
)


def _manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "changed_gold_slot_rows": 10,
                "non_identity_permutation_rows": 12,
            }
        ),
        encoding="utf-8",
    )
    return path


def _residual(path: Path, explained: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "passed": False if explained else True,
                "checks": {"model_exceeds_max_nonleaky_baseline": not explained},
            }
        ),
        encoding="utf-8",
    )
    return path


def _eval(path: Path, accuracy: float, source: float = 0.35) -> Path:
    path.write_text(
        json.dumps(
            {
                "device": "cuda:0",
                "use_pairwise_command_reranker": False,
                "eval_metrics": {"command_slot_accuracy": accuracy},
                "source_overlap_command_slot_baseline": {"split": {"accuracy": source}},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_phase2ao_order_postflight_accepts_order_robust_sidecar(tmp_path: Path) -> None:
    report = audit_phase2ao_order_permutation_postflight(
        original_permutation_manifest_json=_manifest(tmp_path / "original_manifest.json"),
        erased_permutation_manifest_json=_manifest(tmp_path / "erased_manifest.json"),
        residual_baseline_json=_residual(tmp_path / "residual.json"),
        original_permuted_eval_json=_eval(tmp_path / "original.json", 0.99),
        erased_permuted_eval_json=_eval(tmp_path / "erased.json", 0.27),
    )

    assert report["passed"] is True
    assert report["sidecar_order_robustness_supported"] is True
    assert report["strict_erased_residual_control_passed"] is True
    assert report["erased_residual_explained_by_position_prior"] is True


def test_phase2ao_order_postflight_rejects_weak_sidecar_delta(tmp_path: Path) -> None:
    report = audit_phase2ao_order_permutation_postflight(
        original_permutation_manifest_json=_manifest(tmp_path / "original_manifest.json"),
        erased_permutation_manifest_json=_manifest(tmp_path / "erased_manifest.json"),
        residual_baseline_json=_residual(tmp_path / "residual.json"),
        original_permuted_eval_json=_eval(tmp_path / "original.json", 0.80),
        erased_permuted_eval_json=_eval(tmp_path / "erased.json", 0.60),
    )

    assert report["passed"] is False
    assert report["sidecar_order_robustness_supported"] is False
    assert report["checks"]["original_permuted_accuracy_min"] is False


def test_phase2ao_order_postflight_separates_order_support_from_erased_risk(
    tmp_path: Path,
) -> None:
    report = audit_phase2ao_order_permutation_postflight(
        original_permutation_manifest_json=_manifest(tmp_path / "original_manifest.json"),
        erased_permutation_manifest_json=_manifest(tmp_path / "erased_manifest.json"),
        residual_baseline_json=_residual(tmp_path / "residual.json"),
        original_permuted_eval_json=_eval(tmp_path / "original.json", 0.99, source=0.35),
        erased_permuted_eval_json=_eval(tmp_path / "erased.json", 0.37, source=0.35),
    )

    assert report["passed"] is False
    assert report["sidecar_order_robustness_supported"] is True
    assert report["strict_erased_residual_control_passed"] is False
