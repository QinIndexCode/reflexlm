from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

from reflexlm.cli.audit_phase2fr_replay_bundle_summary_replay_negative_controls import (
    audit_phase2fr_replay_bundle_summary_replay_negative_controls,
)
from reflexlm.cli.audit_phase2fs_replay_bundle_summary_replay_bundle import (
    REQUIRED_ARTIFACT_ROLES,
    audit_phase2fs_replay_bundle_summary_replay_bundle,
    validate_phase2fs_replay_bundle_summary_replay_bundle,
)
from test_phase2fq_replay_bundle_summary_cross_directory_replay import _phase2fq_report


def _phase2fr_fixture(tmp_path: Path) -> Path:
    _phase2fq_report(tmp_path)
    phase2fr = audit_phase2fr_replay_bundle_summary_replay_negative_controls(
        phase2fq_report_json=tmp_path / "phase2fq.json",
        output_dir=tmp_path / "fr_controls",
        output_report_json=tmp_path / "phase2fr.json",
    )
    assert phase2fr["passed"] is True
    return tmp_path / "phase2fr.json"


def _phase2fs_report(tmp_path: Path) -> dict:
    report = audit_phase2fs_replay_bundle_summary_replay_bundle(
        phase2fr_report_json=_phase2fr_fixture(tmp_path),
        output_dir=tmp_path / "fs_bundle",
        output_report_json=tmp_path / "phase2fs.json",
    )
    assert report["passed"] is True
    return report


def test_phase2fs_accepts_and_rejects_replay_bundle_summary_replay_bundle(
    tmp_path: Path,
) -> None:
    report = _phase2fs_report(tmp_path)
    validation = validate_phase2fs_replay_bundle_summary_replay_bundle(report)
    manifest = json.loads(
        Path(report["evidence"]["bundle_manifest"]).read_text(encoding="utf-8")
    )

    assert validation["passed"] is True
    assert set(REQUIRED_ARTIFACT_ROLES).issubset(
        {item["role"] for item in manifest["artifacts"]}
    )
    assert report["metrics"]["copied_control_artifact_count"] == report["metrics"][
        "phase2fq_replayed_control_count"
    ] * 2
    assert report["metrics"]["phase2fr_negative_controls_failed"] == report[
        "metrics"
    ]["phase2fr_negative_control_count"]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2ft_replay_bundle_summary_replay_bundle_negative_controls"
    )

    tampered = _isolated_bundle_report(report, tmp_path, "tampered")
    tampered_manifest = json.loads(
        Path(tampered["evidence"]["bundle_manifest"]).read_text(encoding="utf-8")
    )
    markdown = next(
        item
        for item in tampered_manifest["artifacts"]
        if item["role"] == "replayed_bundle_summary_markdown"
    )
    Path(markdown["path"]).write_text("tampered\n", encoding="utf-8")
    validation = validate_phase2fs_replay_bundle_summary_replay_bundle(tampered)
    assert validation["passed"] is False
    assert validation["checks"]["all_manifest_hashes_match"] is False

    missing_role = _isolated_bundle_report(report, tmp_path, "missing_role")
    manifest_path = Path(missing_role["evidence"]["bundle_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        item
        for item in manifest["artifacts"]
        if item["role"] != "replayed_phase2fo_validation"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_phase2fs_replay_bundle_summary_replay_bundle(missing_role)
    assert validation["passed"] is False
    assert validation["checks"]["manifest_roles_complete"] is False

    incomplete = _isolated_bundle_report(report, tmp_path, "incomplete")
    incomplete["source_summary"]["phase2fr_negative_controls_failed"] = 0
    validation = validate_phase2fs_replay_bundle_summary_replay_bundle(incomplete)
    assert validation["passed"] is False
    assert validation["checks"]["source_negative_controls_complete"] is False

    overclaim = _isolated_bundle_report(report, tmp_path, "overclaim")
    overclaim["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2fs_replay_bundle_summary_replay_bundle(overclaim)
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


def _isolated_bundle_report(report: dict, tmp_path: Path, name: str) -> dict:
    isolated = deepcopy(report)
    source_bundle_dir = Path(report["evidence"]["bundle_dir"])
    target_bundle_dir = tmp_path / f"{name}_bundle"
    if target_bundle_dir.exists():
        shutil.rmtree(target_bundle_dir)
    shutil.copytree(source_bundle_dir, target_bundle_dir)
    path_map = {str(source_bundle_dir): str(target_bundle_dir)}
    for source_path in source_bundle_dir.rglob("*"):
        if source_path.is_file():
            path_map[str(source_path)] = str(
                target_bundle_dir / source_path.relative_to(source_bundle_dir)
            )
    isolated = _rewrite_paths(isolated, path_map)
    manifest_path = Path(isolated["evidence"]["bundle_manifest"])
    manifest = _rewrite_paths(json.loads(manifest_path.read_text(encoding="utf-8")), path_map)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return isolated
