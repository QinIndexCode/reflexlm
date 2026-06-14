import json
from pathlib import Path

from reflexlm.cli.audit_phase2am_natural_sidecar_postflight import (
    build_phase2am_natural_sidecar_postflight,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _eval(split: str, accuracy: float, source: float = 0.53, count: int = 112) -> dict:
    return {
        "device": "cuda:0",
        "use_pairwise_command_reranker": False,
        "eval_metrics": {"command_slot_accuracy": accuracy, "command_slot_count": count},
        "source_overlap_command_slot_baseline": {
            split: {"accuracy": source, "total": count, "correct": int(source * count)}
        },
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "controls_manifest_json": _write(
            tmp_path / "manifest.json",
            {
                "passed": True,
                "source_data_unchanged": True,
                "sealed_v3_used": False,
                "row_count_selected": 112,
                "repo_count": 7,
                "checks": {
                    "source_overlap_nonzero": True,
                    "source_overlap_not_ceiling": True,
                },
            },
        ),
        "original_eval_json": _write(tmp_path / "original.json", _eval("phase2am_holdout", 0.99)),
        "erased_eval_json": _write(
            tmp_path / "erased.json", _eval("phase2am_holdout_sidecar_erased", 0.32)
        ),
        "wrong_eval_json": _write(
            tmp_path / "wrong.json", _eval("phase2am_holdout_wrong_sidecar", 0.28)
        ),
    }


def test_phase2am_postflight_accepts_natural_sidecar_dependency(tmp_path: Path) -> None:
    report = build_phase2am_natural_sidecar_postflight(**_paths(tmp_path))

    assert report["passed"] is True
    assert report["claim_bearing_mechanism_evidence"] is True
    assert report["ready_for_package"] is False
    assert report["metrics"]["full_minus_wrong_sidecar"] == 0.71


def test_phase2am_postflight_rejects_erased_control_that_still_solves(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths["erased_eval_json"] = _write(
        tmp_path / "erased.json", _eval("phase2am_holdout_sidecar_erased", 0.90)
    )

    report = build_phase2am_natural_sidecar_postflight(**paths)

    assert report["passed"] is False
    assert report["checks"]["erased_control_degrades"] is False


def test_phase2am_postflight_rejects_source_overlap_ceiling(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths["controls_manifest_json"] = _write(
        tmp_path / "manifest.json",
        {
            "passed": True,
            "source_data_unchanged": True,
            "sealed_v3_used": False,
            "row_count_selected": 112,
            "repo_count": 7,
            "checks": {"source_overlap_nonzero": True, "source_overlap_not_ceiling": False},
        },
    )

    report = build_phase2am_natural_sidecar_postflight(**paths)

    assert report["passed"] is False
    assert report["checks"]["source_overlap_nonzero_not_ceiling"] is False
