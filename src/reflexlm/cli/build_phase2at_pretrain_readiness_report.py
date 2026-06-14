from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2at_pretrain_readiness_report(
    *,
    architecture_boundary_json: str | Path,
    data_gate_json: str | Path,
    package_gate_json: str | Path | None = None,
) -> dict[str, Any]:
    boundary = _read_json(architecture_boundary_json)
    data_gate = _read_json(data_gate_json)
    package_gate = _read_json(package_gate_json) if package_gate_json else {}
    checks = {
        "phase2as_boundary_passed": boundary.get("passed") is True,
        "data_gate_passed": data_gate.get("passed") is True,
        "data_gate_schema_is_phase2at": data_gate.get("schema_version")
        == "phase2at.learned_bounded_patch_candidate.v1",
        "package_gate_supplied": package_gate_json is not None,
        "package_gate_passed_if_supplied": (
            package_gate.get("passed") is True if package_gate_json else None
        ),
        "package_gate_strategy_is_learned_if_supplied": (
            package_gate.get("metrics", {}).get("patch_proposal_strategy")
            == "learned_bounded_candidate"
            if package_gate_json
            else None
        ),
    }
    ready = (
        checks["phase2as_boundary_passed"]
        and checks["data_gate_passed"]
        and checks["data_gate_schema_is_phase2at"]
    )
    blockers = []
    if not checks["data_gate_passed"]:
        blockers.append("phase2at_data_health_failed")
    if not checks["phase2as_boundary_passed"]:
        blockers.append("phase2as_boundary_not_passed")
    if not checks["data_gate_schema_is_phase2at"]:
        blockers.append("phase2at_data_schema_missing_or_wrong")
    unsupported_claims = sorted(
        set(boundary.get("unsupported_claims") or [])
        | set(data_gate.get("unsupported_claims") or [])
        | set(package_gate.get("unsupported_claims") or [])
        | {
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        }
    )
    return {
        "artifact_family": "phase2at_pretrain_readiness_report",
        "passed": ready,
        "ready_for_training": ready,
        "claim_boundary": (
            "Phase2AT training may start only when the Phase2AS boundary, "
            "learned target data health, and Phase2AT schema gates pass. "
            "Package schema gates are pre-package conditions after training, "
            "not prerequisites for starting descriptor-head smoke training. "
            "A failed readiness report is a stop condition, not a prompt to "
            "relabel older symbolic or recorded-patch evidence."
        ),
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "supported_claims": [
            "phase2at_ready_for_learned_bounded_patch_candidate_training"
        ]
        if ready
        else [],
        "unsupported_claims": unsupported_claims,
        "blocked_actions": []
        if ready
        else [
            "do_not_start_phase2at_training",
            "do_not_package_phase2at",
            "do_not_claim_learned_patch_generation",
        ],
        "inputs": {
            "architecture_boundary_json": str(Path(architecture_boundary_json)),
            "data_gate_json": str(Path(data_gate_json)),
            "package_gate_json": str(Path(package_gate_json)) if package_gate_json else None,
        },
        "metrics": {
            "data_gate": data_gate.get("metrics"),
            "package_gate": package_gate.get("metrics"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AT pretrain readiness go/no-go report."
    )
    parser.add_argument("--architecture-boundary-json", required=True)
    parser.add_argument("--data-gate-json", required=True)
    parser.add_argument("--package-gate-json")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2at_pretrain_readiness_report(
        architecture_boundary_json=args.architecture_boundary_json,
        data_gate_json=args.data_gate_json,
        package_gate_json=args.package_gate_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
