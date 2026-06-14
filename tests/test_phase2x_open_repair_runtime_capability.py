import json
from pathlib import Path

from reflexlm.cli.audit_phase2x_open_repair_runtime_capability import (
    audit_phase2x_open_repair_runtime_capability,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest() -> dict:
    return {
        "package_family": "phase2d_native_nervous_package",
        "native_head_path": "native_heads",
        "json_text_target": False,
        "cortex_invocation_policy": "ESCALATE_TO_DEBUG_CORTEX only",
        "open_repair_capabilities": {
            "patch_proposal_head": True,
            "test_selection_head": True,
            "rollback_safety_head": True,
            "stop_condition_head": True,
            "bounded_edit_scope_policy": True,
            "progress_monitor_receptors": True,
            "verification_state_receptors": True,
        },
    }


def _write_head_config(tmp_path: Path, *, enabled: bool = True) -> None:
    head_dir = tmp_path / "native_heads"
    head_dir.mkdir(parents=True, exist_ok=True)
    (head_dir / "head_config.json").write_text(
        json.dumps(
            {
                "backbone_hidden_dim": 8,
                "nsi_latent_dim": 4,
                "open_repair_heads_enabled": enabled,
            }
        ),
        encoding="utf-8",
    )


def test_phase2x_runtime_capability_accepts_open_repair_manifest(tmp_path: Path) -> None:
    _write_head_config(tmp_path)
    report = audit_phase2x_open_repair_runtime_capability(
        package_manifest_json=_write(tmp_path / "package.json", _manifest())
    )
    assert report["passed"] is True
    assert report["missing_capabilities"] == []


def test_phase2x_runtime_capability_rejects_command_selection_only_package(tmp_path: Path) -> None:
    _write_head_config(tmp_path)
    manifest = _manifest()
    del manifest["open_repair_capabilities"]["patch_proposal_head"]
    report = audit_phase2x_open_repair_runtime_capability(
        package_manifest_json=_write(tmp_path / "package.json", manifest)
    )
    assert report["passed"] is False
    assert "patch_proposal_head" in report["missing_capabilities"]
    assert "do_not_claim_open_ended_repair_until_runtime_capabilities_exist" in report["blocked_actions"]


def test_phase2x_runtime_capability_rejects_manifest_only_open_repair_claim(tmp_path: Path) -> None:
    _write_head_config(tmp_path, enabled=False)
    report = audit_phase2x_open_repair_runtime_capability(
        package_manifest_json=_write(tmp_path / "package.json", _manifest())
    )

    assert report["passed"] is False
    assert report["checks"]["open_repair_heads_enabled_in_head_config"] is False
