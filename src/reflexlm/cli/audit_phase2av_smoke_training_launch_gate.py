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


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def audit_phase2av_smoke_training_launch_gate(
    *,
    pretrain_gate_json: str | Path,
    head_manifest_json: str | Path,
    min_train_rows: int = 1,
    min_val_rows: int = 1,
) -> dict[str, Any]:
    pretrain = _read_json(pretrain_gate_json)
    manifest = _read_json(head_manifest_json)
    split_hashes = manifest.get("effective_split_hashes")
    if not isinstance(split_hashes, dict):
        split_hashes = {}
    checks = {
        "pretrain_gate_passed": pretrain.get("passed") is True
        and pretrain.get("ready_for_phase2av_smoke_training") is True,
        "pretrain_gate_not_full_ready": pretrain.get("ready_for_phase2av_full_training")
        is False,
        "head_manifest_family_expected": manifest.get("dataset_family")
        == "phase2av_graded_descriptor_runtime_head_dataset",
        "head_manifest_passed": manifest.get("passed") is True,
        "head_manifest_smoke_allowed": manifest.get("smoke_training_allowed") is True,
        "head_manifest_blocks_full_package_sealed": (
            manifest.get("full_training_allowed") is False
            and manifest.get("package_allowed") is False
            and manifest.get("sealed_eval_allowed") is False
        ),
        "head_split_counts_sufficient": (
            isinstance(manifest.get("train_rows"), int)
            and manifest.get("train_rows", 0) >= min_train_rows
            and isinstance(manifest.get("val_rows"), int)
            and manifest.get("val_rows", 0) >= min_val_rows
        ),
        "head_split_hashes_present": _is_sha256(split_hashes.get("phase2av_head_train"))
        and _is_sha256(split_hashes.get("phase2av_head_val")),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2av_smoke_training_launch_gate",
        "passed": passed,
        "ready_to_start_phase2av_smoke_training": passed,
        "ready_for_phase2av_full_training": False,
        "claim_boundary": (
            "This gate only authorizes small-scale non-sealed Phase2AV smoke "
            "training from hash-bound native-head rows. It does not authorize "
            "full training, packaging, sealed evaluation, freeform patch "
            "generation, production autonomy, or epoch-making claims."
        ),
        "checks": checks,
        "metrics": {
            "train_rows": manifest.get("train_rows"),
            "val_rows": manifest.get("val_rows"),
            "effective_split_hashes": split_hashes,
        },
        "blocked_actions": []
        if passed
        else [
            "do_not_start_phase2av_smoke_training",
            "do_not_package_phase2av",
            "do_not_run_sealed_eval_for_phase2av",
        ],
        "unsupported_claims": [
            "learned_descriptor_runtime_delta_before_postflight",
            "phase2av_full_training_ready",
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "inputs": {
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "head_manifest_json": str(Path(head_manifest_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AV smoke training launch readiness."
    )
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--head-manifest-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-train-rows", type=int, default=1)
    parser.add_argument("--min-val-rows", type=int, default=1)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2av_smoke_training_launch_gate(
        pretrain_gate_json=args.pretrain_gate_json,
        head_manifest_json=args.head_manifest_json,
        min_train_rows=args.min_train_rows,
        min_val_rows=args.min_val_rows,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
