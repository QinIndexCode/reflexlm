from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2d_gates import _metric


DEFAULT_REPORT_DIR = Path("artifacts/reports/phase2l_counterfactual_continuation")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/phase2l_localfull512_freeze")


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path, *, required: bool = True) -> dict[str, Any]:
    resolved = Path(path)
    exists = resolved.exists()
    if required and not exists:
        raise FileNotFoundError(f"Missing required Phase2L freeze artifact: {resolved}")
    return {
        "path": str(resolved),
        "exists": exists,
        "sha256": _sha256(resolved) if exists and resolved.is_file() else None,
    }


def _history0(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history") or []
    return history[0] if history and isinstance(history[0], dict) else {}


def _val_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = _history0(summary).get("val_metrics") or {}
    return metrics if isinstance(metrics, dict) else {}


def build_phase2l_localfull512_freeze_manifest(
    *,
    preregistration_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    training_summary_json: str | Path,
    full_eval_json: str | Path,
    native_head_only_eval_json: str | Path,
    wrong_cache_eval_json: str | Path,
    postflight_json: str | Path,
    cache_erased_cpu_failure_json: str | Path | None = None,
    full_training_failure_json: str | Path | None = None,
    full_gpuenv_training_failure_json: str | Path | None = None,
    current_hardware: str = "RTX 4070 Laptop 8GB WDDM",
) -> dict[str, Any]:
    prereg = _load(preregistration_json)
    data_health = _load(data_health_json)
    pretrain_gate = _load(pretrain_gate_json)
    summary = _load(training_summary_json)
    full_eval = _load(full_eval_json)
    native_eval = _load(native_head_only_eval_json)
    wrong_eval = _load(wrong_cache_eval_json)
    postflight = _load(postflight_json)

    artifacts = {
        "preregistration": _artifact(preregistration_json),
        "data_health": _artifact(data_health_json),
        "pretrain_gate": _artifact(pretrain_gate_json),
        "training_summary": _artifact(training_summary_json),
        "full_eval": _artifact(full_eval_json),
        "native_head_only_eval": _artifact(native_head_only_eval_json),
        "wrong_cache_eval": _artifact(wrong_cache_eval_json),
        "postflight": _artifact(postflight_json),
    }
    if cache_erased_cpu_failure_json:
        artifacts["cache_erased_cpu_eval_control_failure"] = _artifact(
            cache_erased_cpu_failure_json,
            required=False,
        )
    if full_training_failure_json:
        artifacts["full1024_first_throughput_failure"] = _artifact(
            full_training_failure_json,
            required=False,
        )
    if full_gpuenv_training_failure_json:
        artifacts["full1024_gpuenv_throughput_failure"] = _artifact(
            full_gpuenv_training_failure_json,
            required=False,
        )

    checks = {
        "preregistered_before_local_freeze": bool(prereg),
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain_gate.get("passed") is True,
        "localfull512_postflight_passed": postflight.get("passed") is True,
        "localfull512_not_package_ready": postflight.get("ready_for_package") is False,
        "localfull512_not_sealed_ready": postflight.get("ready_for_sealed_eval") is False,
        "larger_gpu_or_repeat_seed_next": postflight.get("allowed_next_action")
        == "run_phase2l_full1024_on_larger_gpu_or_repeat_local_full512_seed",
        "full_beats_native_head_only": postflight.get("checks", {}).get(
            "full_beats_native_head_only_by_required_delta"
        )
        is True,
        "full_beats_wrong_cache": postflight.get("checks", {}).get(
            "full_beats_wrong_cache_by_required_delta"
        )
        is True,
        "full_beats_cache_erased": postflight.get("checks", {}).get(
            "full_beats_cache_erased_by_required_delta"
        )
        is True,
        "full_low_level_qwen_calls_zero": postflight.get("checks", {}).get(
            "full_low_level_qwen_calls_zero"
        )
        is True,
        "sealed_v3_not_used_for_postflight": postflight.get("checks", {}).get(
            "sealed_v3_not_used_for_postflight"
        )
        is True,
    }
    val_metrics = _val_metrics(summary)
    return {
        "manifest_family": "phase2l_localfull512_freeze",
        "frozen": True,
        "passed": all(checks.values()),
        "active_evidence_boundary": "phase2l_local_nonsealed_counterfactual_continuation_positive",
        "sealed_v3_used_for_training_sampling_tuning_or_failure_feedback": False,
        "adapter": {
            "name": summary.get("adapter_name"),
            "path": summary.get("adapter_output_dir"),
            "config_hash": summary.get("config_hash"),
            "effective_split_hashes": summary.get("effective_split_hashes"),
            "train_examples": summary.get("train_examples"),
            "val_examples": summary.get("val_examples"),
            "train_elapsed_seconds": _history0(summary).get("train_elapsed_seconds"),
            "train_steps_per_second": _history0(summary).get("train_steps_per_second"),
            "val_command_slot_accuracy": val_metrics.get("command_slot_accuracy"),
        },
        "metrics": {
            "data_health_source_overlap_val_accuracy": data_health.get("rollups", {})
            .get("source_overlap", {})
            .get("val", {})
            .get("accuracy"),
            "full_completion": _metric(full_eval, "task_completion_rate"),
            "native_head_only_completion": _metric(native_eval, "task_completion_rate"),
            "wrong_cache_completion": _metric(wrong_eval, "task_completion_rate"),
            "cache_erased_completion": postflight.get("metrics", {}).get(
                "cache_erased_completion"
            ),
            "full_minus_native_head_only": postflight.get("metrics", {}).get(
                "full_minus_native_head_only"
            ),
            "full_minus_wrong_cache": postflight.get("metrics", {}).get(
                "full_minus_wrong_cache"
            ),
            "full_minus_cache_erased": postflight.get("metrics", {}).get(
                "full_minus_cache_erased"
            ),
            "full_trace_audit": postflight.get("metrics", {}).get("full_trace_audit"),
        },
        "checks": checks,
        "supported_claims": [
            "local non-sealed counterfactual-continuation full policy beats native-head-only",
            "local non-sealed counterfactual-continuation full policy beats wrong-cache",
            "local non-sealed counterfactual-continuation full policy beats cache-erased",
            "full local control path records zero low-level Qwen calls",
        ],
        "unsupported_claims": [
            "Phase2L full1024 completed on current hardware",
            "Phase2L is package-ready",
            "Phase2L is sealed-v3-ready",
            "Phase2L proves sealed continuation-cache necessity",
            "open-ended debugging or production autonomy",
        ],
        "blocked_actions": [
            "do_not_package_from_localfull512",
            "do_not_run_sealed_v3_from_localfull512",
            "do_not_use_sealed_v3_failures_for_phase2l_data_or_tuning",
        ],
        "hardware_boundary": {
            "current_hardware": current_hardware,
            "full1024_status": "local throughput failure recorded; not a data or sealed-transfer result",
            "allowed_next_steps": [
                "run Phase2L full1024 on a larger GPU",
                "repeat local-full512 with additional seeds as non-package robustness evidence",
            ],
        },
        "artifacts": artifacts,
    }


def markdown_summary(manifest: dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    lines = [
        "# Phase2L Local-Full512 Freeze",
        "",
        f"- Frozen: `{manifest['frozen']}`",
        f"- Passed: `{manifest['passed']}`",
        f"- Evidence boundary: `{manifest['active_evidence_boundary']}`",
        f"- Full completion: `{metrics['full_completion']}`",
        f"- Native-head-only completion: `{metrics['native_head_only_completion']}`",
        f"- Wrong-cache completion: `{metrics['wrong_cache_completion']}`",
        f"- Cache-erased completion: `{metrics['cache_erased_completion']}`",
        f"- Full minus native-head-only: `{metrics['full_minus_native_head_only']}`",
        f"- Full minus wrong-cache: `{metrics['full_minus_wrong_cache']}`",
        f"- Full minus cache-erased: `{metrics['full_minus_cache_erased']}`",
        "",
        "## Supported Claims",
    ]
    lines.extend(f"- {claim}" for claim in manifest["supported_claims"])
    lines += ["", "## Unsupported Claims"]
    lines.extend(f"- {claim}" for claim in manifest["unsupported_claims"])
    lines += ["", "## Blocked Actions"]
    lines.extend(f"- {action}" for action in manifest["blocked_actions"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Phase2L local-full512 evidence.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--current-hardware", default="RTX 4070 Laptop 8GB WDDM")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_phase2l_localfull512_freeze_manifest(
        preregistration_json=report_dir / "phase2l_local_full512_preregistration.json",
        data_health_json=report_dir / "phase2l_full_data_health.json",
        pretrain_gate_json=report_dir / "phase2l_full_pretrain_gate.json",
        training_summary_json=report_dir / "phase2l_localfull512.training_summary.json",
        full_eval_json=report_dir / "phase2l_localfull512_full_eval.json",
        native_head_only_eval_json=report_dir / "phase2l_localfull512_native_head_only_eval.json",
        wrong_cache_eval_json=report_dir / "phase2l_localfull512_wrong_cache_eval.json",
        postflight_json=report_dir / "phase2l_localfull512_postflight.json",
        cache_erased_cpu_failure_json=report_dir
        / "phase2l_localfull512_cache_erased_cpu_eval_control_failure.json",
        full_training_failure_json=report_dir / "phase2l_full_training_throughput_failure.json",
        full_gpuenv_training_failure_json=report_dir
        / "phase2l_full_gpuenv_training_throughput_failure.json",
        current_hardware=args.current_hardware,
    )
    (output_dir / "phase2l_localfull512_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "phase2l_localfull512_freeze_manifest.md").write_text(
        markdown_summary(manifest),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
