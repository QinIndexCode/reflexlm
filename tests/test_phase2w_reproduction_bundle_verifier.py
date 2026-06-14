import json
from pathlib import Path

from reflexlm.cli.verify_phase2w_reproduction_bundle import (
    verify_phase2w_reproduction_bundle,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2w_reproduction_bundle_verifier_accepts_matching_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"passed": true}', encoding="utf-8")
    import hashlib

    bundle = _write(
        tmp_path / "bundle.json",
        {
            "artifacts": [
                {
                    "path": str(artifact),
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
            "verify_commands": [],
        },
    )
    report = verify_phase2w_reproduction_bundle(bundle_json=bundle)
    assert report["passed"] is True
    assert report["runner_independent"] is False
    assert report["artifact_results"][0]["matches"] is True


def test_phase2w_reproduction_bundle_verifier_rejects_stale_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("changed", encoding="utf-8")
    bundle = _write(
        tmp_path / "bundle.json",
        {"artifacts": [{"path": str(artifact), "sha256": "0" * 64}]},
    )
    report = verify_phase2w_reproduction_bundle(bundle_json=bundle)
    assert report["passed"] is False
    assert report["artifact_results"][0]["matches"] is False
