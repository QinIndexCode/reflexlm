import json
from pathlib import Path

from reflexlm.cli.audit_phase2an_candidate_artifact_postflight import (
    audit_phase2an_candidate_artifact_postflight,
)


def _manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {"candidate_artifacts_removed": True},
            }
        ),
        encoding="utf-8",
    )
    return path


def _eval(path: Path, *, accuracy: float, source: float = 0.25) -> Path:
    path.write_text(
        json.dumps(
            {
                "device": "cuda:0",
                "use_pairwise_command_reranker": False,
                "eval_metrics": {"command_slot_accuracy": accuracy},
                "source_overlap_command_slot_baseline": {
                    "split": {"accuracy": source},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_phase2an_postflight_accepts_sidecar_control_with_residual(tmp_path: Path) -> None:
    report = audit_phase2an_candidate_artifact_postflight(
        controls_manifest_json=_manifest(tmp_path / "manifest.json"),
        neutral_original_eval_json=_eval(tmp_path / "original.json", accuracy=1.0),
        neutral_erased_eval_json=_eval(tmp_path / "erased.json", accuracy=0.43),
        neutral_wrong_eval_json=_eval(tmp_path / "wrong.json", accuracy=0.0),
    )

    assert report["passed"] is True
    assert report["sidecar_control_effect_supported"] is True
    assert report["pure_sidecar_dependency_supported"] is False
    assert report["metrics"]["erased_residual_above_source"] is True


def test_phase2an_postflight_rejects_weak_wrong_sidecar_degradation(tmp_path: Path) -> None:
    report = audit_phase2an_candidate_artifact_postflight(
        controls_manifest_json=_manifest(tmp_path / "manifest.json"),
        neutral_original_eval_json=_eval(tmp_path / "original.json", accuracy=0.90),
        neutral_erased_eval_json=_eval(tmp_path / "erased.json", accuracy=0.50),
        neutral_wrong_eval_json=_eval(tmp_path / "wrong.json", accuracy=0.60),
    )

    assert report["passed"] is False
    assert report["checks"]["neutral_wrong_degrades_strongly"] is False
