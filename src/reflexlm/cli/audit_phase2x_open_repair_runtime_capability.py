from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.native_cortex import OPEN_REPAIR_CAPABILITY_NAMES

REQUIRED_OPEN_REPAIR_CAPABILITIES = set(OPEN_REPAIR_CAPABILITY_NAMES)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _capabilities(manifest: dict[str, Any]) -> dict[str, Any]:
    raw = manifest.get("open_repair_capabilities")
    if isinstance(raw, dict):
        return raw
    targets = manifest.get("architecture_targets")
    if isinstance(targets, dict):
        return {
            key: value.get("required") is True if isinstance(value, dict) else bool(value)
            for key, value in targets.items()
        }
    return {}


def _resolve_native_head_path(manifest_path: Path, manifest: dict[str, Any]) -> Path | None:
    raw = manifest.get("native_head_path")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        manifest_relative = manifest_path.parent / path
        cwd_relative = Path.cwd() / path
        path = cwd_relative if cwd_relative.exists() else manifest_relative
    return path


def _read_head_config(native_head_path: Path | None) -> dict[str, Any]:
    if native_head_path is None:
        return {}
    config_path = native_head_path / "head_config.json"
    if not config_path.exists():
        return {}
    return _read_json(config_path)


def audit_phase2x_open_repair_runtime_capability(
    *,
    package_manifest_json: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(package_manifest_json)
    manifest = _read_json(manifest_path)
    native_head_path = _resolve_native_head_path(manifest_path, manifest)
    head_config = _read_head_config(native_head_path)
    caps = _capabilities(manifest)
    missing = sorted(
        key for key in REQUIRED_OPEN_REPAIR_CAPABILITIES if caps.get(key) is not True
    )
    head_config_enabled = head_config.get("open_repair_heads_enabled") is True
    checks = {
        "package_manifest_present": bool(manifest),
        "native_nervous_package_family": manifest.get("package_family")
        == "phase2d_native_nervous_package",
        "native_head_path_declared": native_head_path is not None,
        "head_config_present": bool(head_config),
        "open_repair_heads_enabled_in_head_config": head_config_enabled,
        "open_repair_capabilities_declared": bool(caps),
        "all_required_open_repair_capabilities_present": not missing,
        "manifest_capabilities_match_head_config": (
            not missing and head_config_enabled
        ),
        "explicit_non_json_motor_output": manifest.get("json_text_target") is False,
        "low_level_qwen_calls_disabled_or_bounded": manifest.get("cortex_invocation_policy")
        == "ESCALATE_TO_DEBUG_CORTEX only",
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2x_open_repair_runtime_capability_audit",
        "passed": passed,
        "checks": checks,
        "missing_capabilities": missing,
        "declared_capabilities": caps,
        "head_config_open_repair_heads_enabled": head_config_enabled,
        "blocked_actions": []
        if passed
        else [
            "do_not_run_phase2x_open_repair_as_full_package_result",
            "do_not_claim_open_ended_repair_until_runtime_capabilities_exist",
        ],
        "inputs": {
            "package_manifest_json": str(manifest_path),
            "native_head_path": str(native_head_path) if native_head_path else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2X open-repair runtime capabilities.")
    parser.add_argument("--package-manifest-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2x_open_repair_runtime_capability(
        package_manifest_json=args.package_manifest_json
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
