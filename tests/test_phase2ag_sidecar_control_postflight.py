import json
from pathlib import Path

from reflexlm.cli.audit_phase2ag_sidecar_control_postflight import (
    build_phase2ag_sidecar_control_postflight,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _eval(split: str, accuracy: float, source: float = 0.67) -> dict:
    return {
        "eval_examples": 58,
        "eval_metrics": {
            "command_slot_accuracy": accuracy,
            "command_slot_count": 58,
        },
        "source_overlap_command_slot_baseline": {
            split: {"accuracy": source, "total": 58, "correct": int(source * 58)}
        },
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "smoke_postflight_json": _write(
            tmp_path / "smoke.json",
            {"passed": True, "ready_for_package": False, "ready_for_sealed_eval": False},
        ),
        "controls_manifest_json": _write(tmp_path / "controls.json", {"passed": True}),
        "holdout_eval_json": _write(tmp_path / "full.json", _eval("holdout", 1.0)),
        "erased_eval_json": _write(
            tmp_path / "erased.json", _eval("holdout_sidecar_erased", 0.55)
        ),
        "wrong_eval_json": _write(
            tmp_path / "wrong.json", _eval("holdout_wrong_sidecar", 0.0)
        ),
    }


def test_phase2ag_sidecar_control_postflight_accepts_dependency_smoke(
    tmp_path: Path,
) -> None:
    report = build_phase2ag_sidecar_control_postflight(**_paths(tmp_path))

    assert report["passed"] is True
    assert report["sidecar_dependency_smoke_evidence"] is True
    assert report["claim_bearing_mechanism_evidence"] is False
    assert report["ready_for_package"] is False
    assert report["metrics"]["full_minus_sidecar_erased"] == 0.44999999999999996
    assert "do_not_claim_epoch_making_architecture" in report["blocked_actions"]


def test_phase2ag_sidecar_control_postflight_rejects_no_erased_drop(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["erased_eval_json"] = _write(
        tmp_path / "erased.json", _eval("holdout_sidecar_erased", 0.90)
    )

    report = build_phase2ag_sidecar_control_postflight(**paths)

    assert report["passed"] is False
    assert report["checks"]["erased_control_degrades"] is False


def test_phase2ag_sidecar_control_postflight_rejects_source_drift(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths["wrong_eval_json"] = _write(
        tmp_path / "wrong.json", _eval("holdout_wrong_sidecar", 0.0, source=0.50)
    )

    report = build_phase2ag_sidecar_control_postflight(**paths)

    assert report["passed"] is False
    assert report["checks"]["source_overlap_stable_across_controls"] is False
