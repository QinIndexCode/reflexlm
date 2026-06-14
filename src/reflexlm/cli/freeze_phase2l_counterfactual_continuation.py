from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2d_gates import _metric, _trace_audit


DEFAULT_NONSEALED_REPORT_DIR = Path("artifacts/reports/phase2l_counterfactual_continuation")
DEFAULT_SEALED_REPORT_DIR = Path("artifacts/reports/phase2l_external_trace_v3_semantic_required")
DEFAULT_PACKAGE_ROOT = Path("artifacts/packages/phase2l_counterfactual_continuation_nervous")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/phase2l_counterfactual_continuation_freeze")
DEFAULT_ADAPTER_NAME = (
    "phase2l_counterfactual_continuation_r16_alpha32_lr1e-4_len256_"
    "full1024_val72_checkpointed"
)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "sha256": _sha256(resolved) if resolved.exists() and resolved.is_file() else None,
    }


def _package(path: str | Path) -> dict[str, Any]:
    package = Path(path)
    return {
        "path": str(package),
        "exists": package.exists(),
        "manifest": _artifact(package / "native_nervous_package.json"),
    }


def _completion_counts(payload: dict[str, Any]) -> str | None:
    metric = payload.get("metrics", {}).get("aggregate", {}).get("task_completion_rate")
    positives = metric.get("positives") if isinstance(metric, dict) else None
    episodes = payload.get("episode_count")
    if isinstance(positives, int) and isinstance(episodes, int):
        return f"{positives}/{episodes}"
    completion = _metric(payload, "task_completion_rate")
    if isinstance(completion, float) and isinstance(episodes, int):
        return f"{round(completion * episodes)}/{episodes}"
    return None


def _eval_metrics(path: str | Path) -> dict[str, Any]:
    payload = _load(path)
    return {
        "task_completion_rate": _metric(payload, "task_completion_rate"),
        "completion_counts": _completion_counts(payload),
        "command_decision_accuracy": _metric(payload, "command_decision_accuracy"),
        "read_file_decision_accuracy": _metric(payload, "read_file_decision_accuracy"),
        "oracle_step_accuracy": _metric(payload, "oracle_step_accuracy"),
        "model_calls": _metric(payload, "model_calls"),
        "token_equivalent_cost": _metric(payload, "token_equivalent_cost"),
        "state_hallucination_rate": _metric(payload, "state_hallucination_rate"),
        "trace_audit": _trace_audit(payload),
    }


def build_phase2l_freeze_manifest(
    *,
    nonsealed_report_dir: str | Path = DEFAULT_NONSEALED_REPORT_DIR,
    sealed_report_dir: str | Path = DEFAULT_SEALED_REPORT_DIR,
    package_root: str | Path = DEFAULT_PACKAGE_ROOT,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
) -> dict[str, Any]:
    nonsealed = Path(nonsealed_report_dir)
    sealed = Path(sealed_report_dir)
    packages = Path(package_root)
    required_artifacts = {
        "data_health": nonsealed / "phase2l_full_data_health.json",
        "pretrain_gate": nonsealed / "phase2l_full_pretrain_gate.json",
        "full_training_summary": nonsealed / "phase2l_full1024_checkpointed.training_summary.json",
        "full_postflight": nonsealed / "phase2l_full1024_checkpointed_postflight.json",
        "package_postflight": nonsealed / "phase2l_full1024_checkpointed_package_postflight.json",
        "package_full_eval": nonsealed / "phase2l_full1024_checkpointed_package_full_eval.json",
        "package_native_head_only_eval": nonsealed
        / "phase2l_full1024_checkpointed_package_native_head_only_eval.json",
        "package_wrong_cache_eval": nonsealed
        / "phase2l_full1024_checkpointed_package_wrong_cache_eval.json",
        "sealed_gate": sealed / "phase2l_sealed_v3_gate.json",
        "sealed_baseline_table_md": sealed / "phase2l_sealed_v3_baseline_table.md",
    }
    eval_paths = {
        "full": sealed / "phase2l_full_sealed_v3_eval.json",
        "no_nsi": sealed / "phase2l_no_nsi_sealed_v3_eval.json",
        "native_head_only": sealed / "phase2l_native_head_only_sealed_v3_eval.json",
        "continuation_only": sealed / "phase2l_continuation_only_sealed_v3_eval.json",
        "prompt_only": sealed / "phase2l_prompt_only_sealed_v3_eval.json",
        "react": sealed / "phase2l_react_sealed_v3_eval.json",
    }
    required_artifacts.update({f"{name}_sealed_eval": path for name, path in eval_paths.items()})
    missing = [name for name, path in required_artifacts.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing Phase2L freeze artifacts: " + ", ".join(missing))

    data_health = _load(required_artifacts["data_health"])
    full_postflight = _load(required_artifacts["full_postflight"])
    package_postflight = _load(required_artifacts["package_postflight"])
    sealed_gate = _load(required_artifacts["sealed_gate"])
    evals = {name: _eval_metrics(path) for name, path in eval_paths.items()}
    nonsealed_metrics = package_postflight.get("metrics", {})
    sealed_deltas = sealed_gate.get("metrics", {}).get("deltas", {})

    return {
        "freeze_family": "phase2l_counterfactual_continuation_freeze",
        "frozen": True,
        "sealed_v3_used_for_training_sampling_tuning_or_failure_feedback": False,
        "active_evidence_boundary": (
            "phase2l_nonsealed_counterfactual_continuation_positive_"
            "sealed_v3_transfer_failure"
        ),
        "supported_claims": [
            "non-sealed counterfactual-continuation benchmark distinguishes full package from native-head-only",
            "non-sealed counterfactual-continuation benchmark distinguishes full package from wrong-cache",
            "non-sealed packaged policy reproduces the direct runtime continuation-control evidence",
        ],
        "unsupported_claims": [
            "Phase2L proves continuation memory necessity on sealed v3",
            "Phase2L full package beats native-head-only on sealed v3",
            "Phase2L full package beats continuation-only on sealed v3",
            "Phase2L achieves zero low-level Qwen calls on sealed v3",
            "open-ended debugging / production autonomy",
        ],
        "claim_boundary": sealed_gate.get(
            "claim_boundary",
            "bounded_claim_only_do_not_upgrade_continuation_memory_necessity",
        ),
        "checks": {
            "data_health_passed": data_health.get("passed") is True,
            "full_postflight_passed": full_postflight.get("passed") is True,
            "package_postflight_passed": package_postflight.get("passed") is True,
            "package_postflight_ready_for_package": package_postflight.get("ready_for_package")
            is True,
            "sealed_gate_failed": sealed_gate.get("passed") is False,
            "sealed_full_completion_gate_failed": sealed_gate.get("checks", {}).get(
                "full_completion_gate_passed"
            )
            is False,
            "sealed_full_low_level_qwen_calls_not_zero": sealed_gate.get("checks", {}).get(
                "full_low_level_qwen_calls_zero"
            )
            is False,
            "sealed_v3_inputs_only": sealed_gate.get("checks", {}).get(
                "sealed_v3_inputs_only"
            )
            is True,
            "allowlist_hallucination_zero": sealed_gate.get("checks", {}).get(
                "allowlist_hallucination_zero"
            )
            is True,
        },
        "metrics": {
            "nonsealed_package": {
                "full_completion": nonsealed_metrics.get("full_completion"),
                "native_head_only_completion": nonsealed_metrics.get(
                    "native_head_only_completion"
                ),
                "wrong_cache_completion": nonsealed_metrics.get("wrong_cache_completion"),
                "cache_erased_completion": nonsealed_metrics.get("cache_erased_completion"),
                "full_minus_native_head_only": nonsealed_metrics.get(
                    "full_minus_native_head_only"
                ),
                "full_minus_wrong_cache": nonsealed_metrics.get("full_minus_wrong_cache"),
                "full_minus_cache_erased": nonsealed_metrics.get("full_minus_cache_erased"),
                "source_overlap_val_baseline": data_health.get("rollups", {})
                .get("source_overlap", {})
                .get("val", {})
                .get("accuracy"),
            },
            "sealed_v3": evals,
            "sealed_v3_deltas": sealed_deltas,
        },
        "failure_analysis": {
            "primary_failure": (
                "The Phase2L full package records 0/64 sealed-v3 task completion, so the "
                "predeclared full >= 0.85 sealed gate fails."
            ),
            "secondary_failure": (
                "The Phase2L full package records model_calls=1.0 on sealed v3, so the "
                "zero low-level Qwen-call requirement is not met."
            ),
            "nonsealed_to_sealed_gap": (
                "The registered non-sealed counterfactual benchmark isolates continuation "
                "memory, but the learned mechanism does not transfer to sealed v3."
            ),
            "interpretation": (
                "Use Phase2L as controlled non-sealed continuation-memory evidence only. "
                "Do not claim sealed continuation necessity or tune from sealed failures."
            ),
            "forbidden_followup": [
                "do not inspect sealed failures to design the next training data",
                "do not tune Phase2L from sealed v3 outcomes",
                "do not upgrade the paper claim to sealed continuation memory necessity",
            ],
        },
        "artifacts": {name: _artifact(path) for name, path in required_artifacts.items()},
        "packages": {
            "full": _package(packages / adapter_name),
            "no_nsi": _package(packages / f"{adapter_name}_no_nsi_latent"),
            "native_head_only": _package(packages / f"{adapter_name}_native_head_only"),
            "continuation_only": _package(packages / f"{adapter_name}_continuation_only"),
        },
        "next_step": {
            "allowed": True,
            "phase": "external_generalization_or_new_nonsealed_transfer_study",
            "constraints": [
                "pre-register before training",
                "use non-sealed inputs only",
                "do not use sealed-v3 failures for sampling or tuning",
                "keep paper claim bounded unless a future preregistered final evaluation passes",
            ],
        },
    }


def render_phase2l_freeze_markdown(manifest: dict[str, Any]) -> str:
    nonsealed = manifest["metrics"]["nonsealed_package"]
    sealed = manifest["metrics"]["sealed_v3_deltas"]
    lines = [
        "# Phase2L Freeze",
        "",
        f"- Frozen: `{str(manifest['frozen']).lower()}`",
        f"- Active boundary: `{manifest['active_evidence_boundary']}`",
        f"- Claim boundary: `{manifest['claim_boundary']}`",
        "",
        "## Non-Sealed Package Evidence",
        "",
        f"- Full completion: `{nonsealed['full_completion']}`",
        f"- Native-head-only completion: `{nonsealed['native_head_only_completion']}`",
        f"- Wrong-cache completion: `{nonsealed['wrong_cache_completion']}`",
        f"- Full minus native-head-only: `{nonsealed['full_minus_native_head_only']}`",
        f"- Full minus wrong-cache: `{nonsealed['full_minus_wrong_cache']}`",
        "",
        "## Sealed V3 Boundary",
        "",
        f"- Full minus no-NSI: `{sealed.get('full_minus_no_nsi')}`",
        f"- Full minus native-head-only: `{sealed.get('full_minus_native_head_only')}`",
        f"- Full minus continuation-only: `{sealed.get('full_minus_continuation_only')}`",
        "- Result: bounded claim only; do not use sealed failures for tuning.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Phase2L evidence boundary.")
    parser.add_argument("--nonsealed-report-dir", default=str(DEFAULT_NONSEALED_REPORT_DIR))
    parser.add_argument("--sealed-report-dir", default=str(DEFAULT_SEALED_REPORT_DIR))
    parser.add_argument("--package-root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--adapter-name", default=DEFAULT_ADAPTER_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_phase2l_freeze_manifest(
        nonsealed_report_dir=args.nonsealed_report_dir,
        sealed_report_dir=args.sealed_report_dir,
        package_root=args.package_root,
        adapter_name=args.adapter_name,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase2l_freeze_manifest.json"
    md_path = output_dir / "phase2l_freeze_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_phase2l_freeze_markdown(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
