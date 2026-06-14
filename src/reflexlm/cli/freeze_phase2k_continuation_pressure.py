from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2d_gates import _metric, _trace_audit


DEFAULT_NONSEALED_REPORT_DIR = Path("artifacts/reports/phase2k_continuation_pressure")
DEFAULT_SEALED_REPORT_DIR = Path("artifacts/reports/phase2k_external_trace_v3_semantic_required")
DEFAULT_PACKAGE_ROOT = Path("artifacts/packages/phase2k_continuation_pressure_nervous")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/phase2k_continuation_pressure_freeze")
DEFAULT_ADAPTER_NAME = "phase2k_continuation_pressure_r16_alpha32_lr1e-4_len256_full1024_val512"


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


def build_phase2k_freeze_manifest(
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
        "data_health": nonsealed / "phase2k_continuation_pressure_data_health.json",
        "head_dataset_manifest": nonsealed / "phase2k_head_dataset_manifest.json",
        "smoke_training_summary": nonsealed
        / "phase2k_continuation_pressure_r16_alpha32_lr1e-4_len256_smoke128_val144.training_summary.json",
        "smoke_postflight": nonsealed / "phase2k_smoke_postflight.json",
        "full_training_summary": nonsealed / f"{adapter_name}.training_summary.json",
        "full_postflight": nonsealed / "phase2k_full_postflight.json",
        "sealed_gate": sealed / "phase2k_sealed_v3_gate.json",
        "sealed_baseline_table_md": sealed / "phase2k_sealed_v3_baseline_table.md",
    }
    eval_paths = {
        "full": sealed / "phase2k_full_sealed_v3_eval.json",
        "no_nsi": sealed / "phase2k_no_nsi_sealed_v3_eval.json",
        "native_head_only": sealed / "phase2k_native_head_only_sealed_v3_eval.json",
        "continuation_only": sealed / "phase2k_continuation_only_sealed_v3_eval.json",
        "prompt_only": sealed / "phase2k_prompt_only_sealed_v3_eval.json",
        "react": sealed / "phase2k_react_sealed_v3_eval.json",
    }
    required_artifacts.update({f"{name}_sealed_eval": path for name, path in eval_paths.items()})
    missing = [name for name, path in required_artifacts.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing Phase2K freeze artifacts: " + ", ".join(missing))

    data_health = _load(required_artifacts["data_health"])
    smoke_postflight = _load(required_artifacts["smoke_postflight"])
    full_postflight = _load(required_artifacts["full_postflight"])
    sealed_gate = _load(required_artifacts["sealed_gate"])
    evals = {name: _eval_metrics(path) for name, path in eval_paths.items()}
    sealed_deltas = sealed_gate.get("metrics", {}).get("deltas", {})
    nonsealed_delta = full_postflight.get("metrics", {}).get("full_minus_native_head_only")

    return {
        "freeze_family": "phase2k_continuation_pressure_freeze",
        "frozen": True,
        "sealed_v3_used_for_training_or_tuning": False,
        "active_evidence_boundary": "phase2k_nonsealed_positive_sealed_v3_strict_failure",
        "supported_claims": [
            "non-sealed continuation-pressure benchmark distinguishes full package from native-head-only",
            "Phase2K sealed v3 full package improves over no-NSI, but only as bounded delta evidence",
        ],
        "unsupported_claims": [
            "full package beats native-head-only on sealed v3",
            "continuation cache is necessary on sealed v3",
            "Phase2K proves full-package necessity",
            "Phase2K achieves zero low-level Qwen calls on sealed v3",
            "open-ended debugging / production autonomy",
        ],
        "claim_boundary": (
            "Freeze Phase2K as non-sealed continuation-pressure positive evidence and sealed-v3 "
            "strict-gate failure evidence. Do not use sealed-v3 failures to construct, sample, tune, "
            "or debug later training data."
        ),
        "checks": {
            "data_health_passed": data_health.get("passed") is True,
            "smoke_postflight_passed": smoke_postflight.get("passed") is True,
            "smoke_not_package_ready": smoke_postflight.get("ready_for_package") is False,
            "full_postflight_passed": full_postflight.get("passed") is True,
            "full_nonsealed_beats_native_head_only": isinstance(nonsealed_delta, float)
            and nonsealed_delta >= 0.10,
            "sealed_gate_failed": sealed_gate.get("passed") is False,
            "sealed_full_beats_no_nsi": sealed_gate.get("checks", {}).get(
                "full_beats_no_nsi_by_required_delta"
            )
            is True,
            "sealed_full_does_not_beat_native_head_only": sealed_gate.get("checks", {}).get(
                "full_beats_native_head_only_by_required_delta"
            )
            is False,
            "sealed_full_low_level_qwen_calls_not_zero": sealed_gate.get("checks", {}).get(
                "full_low_level_qwen_calls_zero"
            )
            is False,
            "allowlist_hallucination_zero": sealed_gate.get("checks", {}).get(
                "allowlist_hallucination_zero"
            )
            is True,
        },
        "metrics": {
            "nonsealed": {
                "full_completion": full_postflight.get("metrics", {}).get("full_completion"),
                "native_head_only_completion": full_postflight.get("metrics", {}).get(
                    "native_head_only_completion"
                ),
                "full_minus_native_head_only": nonsealed_delta,
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
                "The sealed v3 full package ties native-head-only at 21/64, so the "
                "predeclared full_minus_native_head_only >= 0.10 mechanism gate fails."
            ),
            "secondary_failure": (
                "The sealed v3 full package records model_calls=1.0, so the zero low-level "
                "Qwen-call requirement is not met."
            ),
            "nonsealed_to_sealed_gap": (
                "The registered non-sealed benchmark distinguishes continuation memory from "
                "native-head-only, but that distinction does not transfer to sealed v3."
            ),
            "interpretation": (
                "Use Phase2K as evidence that the pressure benchmark can be made diagnostic "
                "under controlled non-sealed conditions, not as evidence that continuation "
                "cache or the full package is necessary on sealed v3."
            ),
            "forbidden_followup": [
                "do not inspect sealed failures to design Phase2L cases",
                "do not resample or tune Phase2K from sealed v3 outcomes",
                "do not upgrade the paper claim to full-package necessity",
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
            "phase": "phase2l_counterfactual_continuation",
            "allowed": True,
            "constraints": [
                "pre-register before training",
                "use non-sealed synthetic profiles only",
                "use counterfactual pairs with identical current visible state and different prior state",
                "measure wrong-cache and cache-erased controls before package",
                "sealed v3 remains final evaluation-only and cannot guide data construction",
            ],
        },
    }


def build_phase2l_preregistration() -> dict[str, Any]:
    return {
        "preregistration_family": "phase2l_counterfactual_continuation",
        "status": "preregistered_not_generated_not_trained",
        "objective": (
            "Test whether continuation memory provides a necessary runtime-visible advantage "
            "when current-state rows are counterfactually identical but prior context changes "
            "the correct command."
        ),
        "data_constraints": {
            "sealed_inputs_allowed": False,
            "gold_or_hidden_labels_visible": False,
            "hardcoded_test_names_allowed": False,
            "profiles": [
                "phase2l_counterfactual_continuation_train",
                "phase2l_counterfactual_continuation_val",
            ],
            "case_design": [
                "paired episodes share identical command-state visible text",
                "paired episodes have different prior source/read/stderr evidence",
                "correct command differs only because prior continuation state differs",
                "candidate commands are lexically symmetric and same-intent",
                "wrong-cache and cache-erased controls are explicitly evaluated",
            ],
        },
        "graded_axes": {
            "evidence_density": ["low", "medium", "high"],
            "candidate_count": [2, 3, 4],
            "continuation_depth": ["one_step", "two_step", "stale_state_refresh"],
            "counterfactual_class": [
                "same_visible_state_different_prior_file",
                "same_visible_state_different_prior_traceback",
                "same_visible_state_stale_cache_refresh",
            ],
        },
        "predeclared_gates": {
            "data_health_passes": True,
            "source_overlap_baseline_max": 0.50,
            "native_head_only_baseline_measured": True,
            "wrong_cache_baseline_measured": True,
            "cache_erased_baseline_measured": True,
            "smoke_full_completion_min": 0.85,
            "full_minus_native_head_only_min": 0.15,
            "full_minus_wrong_cache_min": 0.25,
            "full_minus_cache_erased_min": 0.25,
            "full_low_level_qwen_calls": 0,
        },
        "stop_conditions": [
            "data health fails",
            "source overlap or native-head-only solves the validation set",
            "smoke full completion < 0.85",
            "full_minus_native_head_only < 0.15",
            "full_minus_wrong_cache < 0.25",
            "full_minus_cache_erased < 0.25",
            "full low-level Qwen calls > 0",
        ],
        "claim_boundary": (
            "Passing Phase2L would support continuation-memory necessity only under the "
            "registered counterfactual non-sealed benchmark until package and final sealed "
            "evaluation also pass. Failing Phase2L preserves the current bounded claim."
        ),
    }


def markdown_summary(manifest: dict[str, Any]) -> str:
    nonsealed = manifest["metrics"]["nonsealed"]
    sealed = manifest["metrics"]["sealed_v3_deltas"]
    lines = [
        "# Phase2K Continuation-Pressure Freeze",
        "",
        f"- Frozen: `{manifest['frozen']}`",
        f"- Evidence boundary: `{manifest['active_evidence_boundary']}`",
        f"- Non-sealed full completion: `{nonsealed['full_completion']}`",
        f"- Non-sealed native-head-only completion: `{nonsealed['native_head_only_completion']}`",
        f"- Non-sealed full minus native-head-only: `{nonsealed['full_minus_native_head_only']}`",
        f"- Sealed full minus no-NSI: `{sealed.get('full_minus_no_nsi')}`",
        f"- Sealed full minus native-head-only: `{sealed.get('full_minus_native_head_only')}`",
        f"- Sealed full minus continuation-only: `{sealed.get('full_minus_continuation_only')}`",
        "",
        "## Supported Claims",
    ]
    lines.extend(f"- {claim}" for claim in manifest["supported_claims"])
    lines += ["", "## Unsupported Claims"]
    lines.extend(f"- {claim}" for claim in manifest["unsupported_claims"])
    lines += ["", "## Failure Analysis"]
    lines.extend(f"- {value}" for value in manifest["failure_analysis"].values() if isinstance(value, str))
    lines += ["", "## Checks"]
    lines.extend(f"- `{key}`: `{value}`" for key, value in manifest["checks"].items())
    return "\n".join(lines) + "\n"


def preregistration_markdown(prereg: dict[str, Any]) -> str:
    lines = [
        "# Phase2L Counterfactual Continuation Preregistration",
        "",
        f"- Status: `{prereg['status']}`",
        f"- Objective: {prereg['objective']}",
        "",
        "## Case Design",
    ]
    lines.extend(f"- {item}" for item in prereg["data_constraints"]["case_design"])
    lines += ["", "## Gates"]
    lines.extend(f"- `{key}`: `{value}`" for key, value in prereg["predeclared_gates"].items())
    lines += ["", "## Stop Conditions"]
    lines.extend(f"- {item}" for item in prereg["stop_conditions"])
    lines += ["", f"Claim boundary: {prereg['claim_boundary']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Phase2K and preregister Phase2L.")
    parser.add_argument("--nonsealed-report-dir", default=str(DEFAULT_NONSEALED_REPORT_DIR))
    parser.add_argument("--sealed-report-dir", default=str(DEFAULT_SEALED_REPORT_DIR))
    parser.add_argument("--package-root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--adapter-name", default=DEFAULT_ADAPTER_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_phase2k_freeze_manifest(
        nonsealed_report_dir=args.nonsealed_report_dir,
        sealed_report_dir=args.sealed_report_dir,
        package_root=args.package_root,
        adapter_name=args.adapter_name,
    )
    prereg = build_phase2l_preregistration()

    manifest_json = output / "phase2k_continuation_pressure_freeze_manifest.json"
    manifest_md = output / "phase2k_continuation_pressure_freeze_manifest.md"
    prereg_json = output / "phase2l_counterfactual_continuation_preregistration.json"
    prereg_md = output / "phase2l_counterfactual_continuation_preregistration.md"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_md.write_text(markdown_summary(manifest), encoding="utf-8")
    prereg_json.write_text(json.dumps(prereg, indent=2, ensure_ascii=False), encoding="utf-8")
    prereg_md.write_text(preregistration_markdown(prereg), encoding="utf-8")
    print(
        json.dumps(
            {
                "freeze_manifest": str(manifest_json),
                "freeze_markdown": str(manifest_md),
                "phase2l_preregistration": str(prereg_json),
                "phase2l_markdown": str(prereg_md),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
