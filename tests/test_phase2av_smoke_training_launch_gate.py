import json
from pathlib import Path

from reflexlm.cli.audit_phase2av_smoke_training_launch_gate import (
    audit_phase2av_smoke_training_launch_gate,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pretrain(tmp_path: Path, *, passed: bool = True) -> Path:
    return _write(
        tmp_path / "pretrain.json",
        {
            "passed": passed,
            "ready_for_phase2av_smoke_training": passed,
            "ready_for_phase2av_full_training": False,
        },
    )


def _manifest(tmp_path: Path, *, package_allowed: bool = False) -> Path:
    return _write(
        tmp_path / "manifest.json",
        {
            "dataset_family": "phase2av_graded_descriptor_runtime_head_dataset",
            "passed": True,
            "train_rows": 14,
            "val_rows": 8,
            "smoke_training_allowed": True,
            "full_training_allowed": False,
            "package_allowed": package_allowed,
            "sealed_eval_allowed": False,
            "effective_split_hashes": {
                "phase2av_head_train": "a" * 64,
                "phase2av_head_val": "b" * 64,
            },
        },
    )


def test_phase2av_launch_gate_accepts_smoke_only_path(tmp_path: Path) -> None:
    report = audit_phase2av_smoke_training_launch_gate(
        pretrain_gate_json=_pretrain(tmp_path),
        head_manifest_json=_manifest(tmp_path),
        min_train_rows=14,
        min_val_rows=8,
    )

    assert report["passed"] is True
    assert report["ready_to_start_phase2av_smoke_training"] is True
    assert report["ready_for_phase2av_full_training"] is False


def test_phase2av_launch_gate_rejects_package_ready_manifest(tmp_path: Path) -> None:
    report = audit_phase2av_smoke_training_launch_gate(
        pretrain_gate_json=_pretrain(tmp_path),
        head_manifest_json=_manifest(tmp_path, package_allowed=True),
    )

    assert report["passed"] is False
    assert report["checks"]["head_manifest_blocks_full_package_sealed"] is False
    assert "do_not_start_phase2av_smoke_training" in report["blocked_actions"]
