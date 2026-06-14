from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

from reflexlm.cli.audit_phase2ft_replay_bundle_summary_replay_bundle_negative_controls import (
    audit_phase2ft_replay_bundle_summary_replay_bundle_negative_controls,
)
from reflexlm.cli.audit_phase2fu_replay_bundle_summary_reproducibility_manifest import (
    REQUIRED_REPRODUCTION_STEPS,
    REQUIRED_SOURCE_REPORT_ROLES,
    audit_phase2fu_replay_bundle_summary_reproducibility_manifest,
    validate_phase2fu_replay_bundle_summary_reproducibility_manifest,
)
from test_phase2fs_replay_bundle_summary_replay_bundle import _phase2fs_report


def _phase2ft_fixture(tmp_path: Path) -> Path:
    _phase2fs_report(tmp_path)
    phase2ft = audit_phase2ft_replay_bundle_summary_replay_bundle_negative_controls(
        phase2fs_report_json=tmp_path / "phase2fs.json",
        output_dir=tmp_path / "ft_controls",
        output_report_json=tmp_path / "phase2ft.json",
    )
    assert phase2ft["passed"] is True
    return tmp_path / "phase2ft.json"


def _phase2fu_report(tmp_path: Path) -> dict:
    report = audit_phase2fu_replay_bundle_summary_reproducibility_manifest(
        phase2ft_report_json=_phase2ft_fixture(tmp_path),
        output_manifest_json=tmp_path / "phase2fu_manifest.json",
        output_report_json=tmp_path / "phase2fu.json",
    )
    assert report["passed"] is True
    return report


def test_phase2fu_accepts_and_rejects_reproducibility_manifest(tmp_path: Path) -> None:
    report = _phase2fu_report(tmp_path)
    validation = validate_phase2fu_replay_bundle_summary_reproducibility_manifest(report)
    manifest = json.loads(
        Path(report["evidence"]["reproducibility_manifest"]).read_text(
            encoding="utf-8"
        )
    )

    assert validation["passed"] is True
    assert set(REQUIRED_SOURCE_REPORT_ROLES).issubset(
        {item["role"] for item in manifest["source_reports"]}
    )
    assert set(REQUIRED_REPRODUCTION_STEPS).issubset(
        {item["step_id"] for item in manifest["reproduction_steps"]}
    )
    assert report["metrics"]["bundle_artifact_count"] == report["metrics"][
        "bundle_artifact_hash_match_count"
    ]
    assert report["metrics"]["phase2ft_negative_control_count"] == report["metrics"][
        "phase2ft_negative_controls_failed"
    ]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2fv_replay_bundle_summary_manifest_negative_controls"
    )

    tampered = _isolated_manifest_report(report, tmp_path, "tampered")
    manifest = json.loads(
        Path(tampered["evidence"]["reproducibility_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    first_artifact = manifest["bundle_artifacts"][0]
    Path(first_artifact["path"]).write_text("tampered\n", encoding="utf-8")
    validation = validate_phase2fu_replay_bundle_summary_reproducibility_manifest(
        tampered
    )
    assert validation["passed"] is False
    assert validation["checks"]["bundle_artifact_hashes_match"] is False

    missing_source = _isolated_manifest_report(report, tmp_path, "missing_source")
    manifest_path = Path(missing_source["evidence"]["reproducibility_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_reports"] = [
        item
        for item in manifest["source_reports"]
        if item["role"] != REQUIRED_SOURCE_REPORT_ROLES[0]
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_phase2fu_replay_bundle_summary_reproducibility_manifest(
        missing_source
    )
    assert validation["passed"] is False
    assert validation["checks"]["source_report_roles_complete"] is False

    missing_step = _isolated_manifest_report(report, tmp_path, "missing_step")
    manifest_path = Path(missing_step["evidence"]["reproducibility_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reproduction_steps"] = [
        item
        for item in manifest["reproduction_steps"]
        if item["step_id"] != REQUIRED_REPRODUCTION_STEPS[0]
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_phase2fu_replay_bundle_summary_reproducibility_manifest(
        missing_step
    )
    assert validation["passed"] is False
    assert validation["checks"]["reproduction_steps_complete"] is False

    count_mismatch = _isolated_manifest_report(report, tmp_path, "count_mismatch")
    count_mismatch["source_summary"]["phase2fs_copied_control_artifact_count"] = 0
    validation = validate_phase2fu_replay_bundle_summary_reproducibility_manifest(
        count_mismatch
    )
    assert validation["passed"] is False
    assert validation["checks"]["control_artifact_count_preserved"] is False

    overclaim = _isolated_manifest_report(report, tmp_path, "overclaim")
    overclaim["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2fu_replay_bundle_summary_reproducibility_manifest(
        overclaim
    )
    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False


def _rewrite_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [_rewrite_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, path_map) for key, item in value.items()}
    return value


def _isolated_manifest_report(report: dict, tmp_path: Path, name: str) -> dict:
    isolated = deepcopy(report)
    manifest_path = Path(report["evidence"]["reproducibility_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_paths = [
        Path(str(item["path"]))
        for item in manifest.get("bundle_artifacts", [])
        if isinstance(item, dict) and item.get("path")
    ]
    phase2fs = json.loads(
        Path(report["evidence"]["phase2fs_report_json"]).read_text(encoding="utf-8")
    )
    bundle_root = Path(phase2fs["evidence"]["bundle_dir"])
    target_bundle_root = tmp_path / f"{name}_bundle"
    if target_bundle_root.exists():
        shutil.rmtree(target_bundle_root)
    if bundle_root.exists():
        shutil.copytree(bundle_root, target_bundle_root)
    path_map = {str(bundle_root): str(target_bundle_root)}
    for source_path in bundle_paths:
        try:
            relative = source_path.relative_to(bundle_root)
        except ValueError:
            continue
        path_map[str(source_path)] = str(target_bundle_root / relative)
    target_manifest = tmp_path / f"{name}_manifest.json"
    rewritten_manifest = _rewrite_paths(manifest, path_map)
    target_manifest.write_text(json.dumps(rewritten_manifest), encoding="utf-8")
    isolated["evidence"]["reproducibility_manifest"] = str(target_manifest)
    return isolated
