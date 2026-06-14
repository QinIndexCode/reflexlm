from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_VERIFY_COMMANDS = [
    "python -m pytest tests/test_phase2o_epoch_claim_readiness.py tests/test_phase2w_epoch_preregistration.py tests/test_phase2w_epoch_gate.py -q",
    "python -m pytest tests/test_phase2v_graded_transfer_controls.py tests/test_phase2s_open_repair_smoke.py tests/test_phase2x_open_repair_task_manifest.py tests/test_phase2x_open_repair_task_manifest_builder.py tests/test_phase2x_open_repair_runtime_capability.py -q",
    "python -m pytest tests/test_phase2w_reproduction_bundle.py tests/test_phase2w_reproduction_bundle_verifier.py tests/test_phase2w_production_safety_report.py tests/test_phase2w_live_agent_baseline.py tests/test_phase2w_live_agent_missing_report.py tests/test_phase2w_open_ended_repair.py tests/test_phase2w_open_repair_missing_report.py tests/test_phase2w_reviewer_consensus.py tests/test_phase2w_reviewer_consensus_from_reviews.py -q",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2w_reproduction_bundle(
    *,
    artifact_paths: list[str | Path],
    verify_commands: list[str] | None = None,
) -> dict[str, Any]:
    artifacts = []
    missing = []
    for raw_path in artifact_paths:
        path = Path(raw_path)
        if not path.exists():
            missing.append(str(path))
            continue
        artifacts.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    bundle_ready = not missing and bool(artifacts)
    return {
        "artifact_family": "phase2w_reproduction_bundle_manifest",
        "passed": False,
        "bundle_ready_for_independent_runner": bundle_ready,
        "runner_independent": False,
        "one_command_reproduction": bundle_ready,
        "hash_locked_splits": bundle_ready,
        "no_local_patch_required": False,
        "reason_not_independent": (
            "This manifest was generated in the development workspace. It locks inputs for an outside or clean replay runner but is not itself independent reproduction."
        ),
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "verify_commands": verify_commands or DEFAULT_VERIFY_COMMANDS,
        "next_required_action": "run_this_bundle_in_clean_or_external_workspace",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2W reproduction bundle manifest.")
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--verify-command", action="append")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2w_reproduction_bundle(
        artifact_paths=args.artifact,
        verify_commands=args.verify_command,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["bundle_ready_for_independent_runner"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
