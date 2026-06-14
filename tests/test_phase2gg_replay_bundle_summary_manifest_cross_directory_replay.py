from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

from reflexlm.cli.audit_phase2gf_replay_bundle_summary_manifest_negative_controls import (
    audit_phase2gf_replay_bundle_summary_manifest_negative_controls,
)
from reflexlm.cli.audit_phase2gg_replay_bundle_summary_manifest_cross_directory_replay import (
    audit_phase2gg_replay_bundle_summary_manifest_cross_directory_replay,
    validate_phase2gg_replay_bundle_summary_manifest_cross_directory_replay,
)
from test_phase2ge_replay_bundle_summary_reproducibility_manifest import (
    _phase2ge_report,
)


def _phase2gf_fixture(tmp_path: Path) -> Path:
    _phase2ge_report(tmp_path)
    phase2gf = audit_phase2gf_replay_bundle_summary_manifest_negative_controls(
        phase2ge_report_json=tmp_path / "phase2ge.json",
        output_dir=tmp_path / "gf_controls",
        output_report_json=tmp_path / "phase2gf.json",
    )
    assert phase2gf["passed"] is True
    return tmp_path / "phase2gf.json"


def _phase2gg_report(tmp_path: Path) -> dict:
    report = audit_phase2gg_replay_bundle_summary_manifest_cross_directory_replay(
        phase2gf_report_json=_phase2gf_fixture(tmp_path),
        output_dir=tmp_path / "gg_replay",
        output_report_json=tmp_path / "phase2gg.json",
    )
    assert report["passed"] is True
    return report


def test_phase2gg_accepts_manifest_cross_directory_replay(tmp_path: Path) -> None:
    report = _phase2gg_report(tmp_path)
    validation = validate_phase2gg_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )

    assert validation["passed"] is True
    assert report["metrics"]["replayed_source_report_count"] == 4
    assert report["metrics"]["replayed_bundle_artifact_count"] == report["metrics"][
        "replayed_bundle_artifact_hash_match_count"
    ]
    assert report["metrics"]["replayed_control_artifact_count"] == report[
        "source_summary"
    ]["phase2ge_control_artifact_count"]
    assert report["metrics"]["phase2gf_negative_control_count"] == report["metrics"][
        "phase2gf_negative_controls_failed"
    ]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2gh_replay_bundle_summary_manifest_cross_directory_negative_controls"
    )


def test_phase2gg_validation_rejects_negative_matrix(tmp_path: Path) -> None:
    base_report = _phase2gg_report(tmp_path)

    report = _isolated_replay_report(base_report, tmp_path, "tampered_artifact")
    artifact = next(Path(report["evidence"]["replay_dir"]).glob("bundle_artifacts/*"))
    artifact.write_text("tampered\n", encoding="utf-8")
    validation = validate_phase2gg_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )
    assert validation["passed"] is False
    assert validation["checks"]["replayed_phase2ge_validation_passed"] is False

    report = _isolated_replay_report(base_report, tmp_path, "missing_report")
    Path(report["evidence"]["replayed_phase2ge_report"]).unlink()
    validation = validate_phase2gg_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )
    assert validation["passed"] is False
    assert validation["checks"]["replayed_phase2ge_report_readable"] is False

    report = _isolated_replay_report(base_report, tmp_path, "count_mismatch")
    report["replay_summary"]["control_artifact_count"] = 0
    validation = validate_phase2gg_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )
    assert validation["passed"] is False
    assert validation["checks"]["replayed_control_artifact_count_preserved"] is False

    report = _isolated_replay_report(base_report, tmp_path, "same_dir")
    report["replay_summary"]["replay_manifest_parent"] = report["replay_summary"][
        "source_manifest_parent"
    ]
    validation = validate_phase2gg_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )
    assert validation["passed"] is False
    assert validation["checks"]["replay_directory_is_distinct"] is False

    report = _isolated_replay_report(base_report, tmp_path, "epoch_claim")
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2gg_replay_bundle_summary_manifest_cross_directory_replay(
        report
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


def _isolated_replay_report(report: dict, tmp_path: Path, name: str) -> dict:
    isolated = deepcopy(report)
    source_replay_dir = Path(report["evidence"]["replay_dir"])
    target_replay_dir = tmp_path / f"{name}_replay"
    if target_replay_dir.exists():
        shutil.rmtree(target_replay_dir)
    shutil.copytree(source_replay_dir, target_replay_dir)
    path_map = {str(source_replay_dir): str(target_replay_dir)}
    for source_path in source_replay_dir.rglob("*"):
        if source_path.is_file():
            path_map[str(source_path)] = str(
                target_replay_dir / source_path.relative_to(source_replay_dir)
            )
    isolated = _rewrite_paths(isolated, path_map)
    replayed_report = Path(isolated["evidence"]["replayed_phase2ge_report"])
    replayed_report_json = _rewrite_paths(
        json.loads(replayed_report.read_text(encoding="utf-8")),
        path_map,
    )
    replayed_report.write_text(json.dumps(replayed_report_json), encoding="utf-8")
    replayed_manifest = Path(
        replayed_report_json["evidence"]["reproducibility_manifest"]
    )
    replayed_manifest_json = _rewrite_paths(
        json.loads(replayed_manifest.read_text(encoding="utf-8")),
        path_map,
    )
    replayed_manifest.write_text(json.dumps(replayed_manifest_json), encoding="utf-8")
    return isolated
