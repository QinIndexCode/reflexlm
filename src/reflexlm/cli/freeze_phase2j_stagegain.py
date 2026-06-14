from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2d_gates import _metric, _trace_audit


DEFAULT_REPORT_DIR = Path("artifacts/reports/phase2i_external_trace_v3_semantic_required")
DEFAULT_ACTIONGATE_REPORT_DIR = Path(
    "artifacts/reports/phase2j_source_overlap_hard_actiongate_actionbalanced"
)
DEFAULT_PACKAGE_ROOT = Path("artifacts/packages/p2j_actionbal_stagegain_nervous")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/phase2j_stagegain_freeze")


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
    manifest = package / "native_nervous_package.json"
    return {
        "path": str(package),
        "exists": package.exists(),
        "manifest": _artifact(manifest),
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


def build_phase2j_stagegain_freeze_manifest(
    *,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    actiongate_report_dir: str | Path = DEFAULT_ACTIONGATE_REPORT_DIR,
    package_root: str | Path = DEFAULT_PACKAGE_ROOT,
) -> dict[str, Any]:
    report = Path(report_dir)
    actiongate = Path(actiongate_report_dir)
    packages = Path(package_root)
    eval_paths = {
        "full": report / "p2j_stagegain.full.external_trace_v3_eval.json",
        "no_nsi_latent": report / "p2j_stagegain.no_nsi_latent.external_trace_v3_eval.json",
        "native_head_only": report / "p2j_stagegain.native_head_only.external_trace_v3_eval.json",
        "continuation_only": report / "p2j_stagegain.continuation_only.external_trace_v3_eval.json",
        "prompt_only": report / "p2j_stagegain.prompt_only.external_trace_v3_eval.json",
        "react": report / "p2j_stagegain.react.external_trace_v3_eval.json",
    }
    required_artifacts = {
        "full_data_health": actiongate / "p2j_actionbal_stagegain_full_data_health.json",
        "full_pretrain_gate": actiongate / "p2j_actionbal_stagegain_full_pretrain_gate.json",
        "full_training_summary": actiongate / "p2j_actionbal_stagegain_full.training_summary.json",
        "full_postflight": actiongate / "p2j_actionbal_stagegain_full_postflight.json",
        "exact_baseline_table_json": report / "p2j_stagegain_external_v3_exact_baseline_table.json",
        "exact_baseline_table_md": report / "p2j_stagegain_external_v3_exact_baseline_table.md",
        "external_trace_gate": report / "p2j_stagegain_external_v3_gate.json",
        "strict_delta_review": report / "p2j_stagegain_external_v3_strict_delta_review.json",
        "reproducibility_report": report / "p2j_stagegain_reproducibility_report.md",
        **{f"{name}_eval": path for name, path in eval_paths.items()},
    }
    missing = [name for name, path in required_artifacts.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing Phase2J freeze artifacts: " + ", ".join(missing))

    evals = {name: _eval_metrics(path) for name, path in eval_paths.items()}
    full = evals["full"]["task_completion_rate"]
    no_nsi = evals["no_nsi_latent"]["task_completion_rate"]
    native = evals["native_head_only"]["task_completion_rate"]
    continuation = evals["continuation_only"]["task_completion_rate"]
    strict_review = _load(required_artifacts["strict_delta_review"])
    gate = _load(required_artifacts["external_trace_gate"])
    postflight = _load(required_artifacts["full_postflight"])

    return {
        "freeze_family": "phase2j_stagegain_freeze",
        "frozen": True,
        "sealed_v3_used_for_training_or_tuning": False,
        "active_evidence_boundary": "phase2j_actiongate_stagegain_external_v3",
        "supported_claims": [
            "native-head/debug-stage semantic-required mechanism supported",
            "NSI latent contributes versus no-NSI on sealed v3",
        ],
        "unsupported_claims": [
            "full package beats native-head-only",
            "continuation cache is necessary on sealed v3",
            "open-ended debugging / production autonomy",
        ],
        "claim_boundary": (
            "Use Phase2J stagegain as bounded semantic-required native-head/debug-stage evidence; "
            "do not reinterpret Phase2I or sealed-v3 failures as training signals."
        ),
        "checks": {
            "all_required_artifacts_present": not missing,
            "nonsealed_full_postflight_passed": postflight.get("passed") is True,
            "repo_external_trace_gate_passed": gate.get("passed") is True,
            "strict_full_package_delta_not_proven": (
                strict_review.get("strict_full_package_gate_passed") is False
            ),
            "full_beats_no_nsi_by_15pp": isinstance(full, float)
            and isinstance(no_nsi, float)
            and full - no_nsi >= 0.15,
            "full_beats_continuation_only_by_15pp": isinstance(full, float)
            and isinstance(continuation, float)
            and full - continuation >= 0.15,
            "full_does_not_beat_native_head_only_by_10pp": isinstance(full, float)
            and isinstance(native, float)
            and full - native < 0.10,
            "allowlist_hallucination_zero": evals["full"]["state_hallucination_rate"] == 0.0,
            "low_level_qwen_calls_zero": (
                evals["full"]["trace_audit"]["low_level_qwen_calls"] == 0
            ),
        },
        "metrics": {
            "sealed_v3": evals,
            "deltas": {
                "full_minus_no_nsi_latent": full - no_nsi,
                "full_minus_continuation_only": full - continuation,
                "full_minus_native_head_only": full - native,
            },
            "nonsealed_postflight": postflight.get("metrics", {}),
        },
        "artifacts": {name: _artifact(path) for name, path in required_artifacts.items()},
        "packages": {
            "full": _package(packages / "p2j_actionbal_stagegain_full1024_val288"),
            "no_nsi_latent": _package(
                packages / "p2j_actionbal_stagegain_full1024_val288_no_nsi_latent"
            ),
            "native_head_only": _package(
                packages / "p2j_actionbal_stagegain_full1024_val288_native_head_only"
            ),
            "continuation_only": _package(
                packages / "p2j_actionbal_stagegain_full1024_val288_continuation_only"
            ),
        },
        "next_step": {
            "phase": "phase2k_continuation_pressure",
            "allowed": True,
            "constraints": [
                "non-sealed profiles only before package",
                "measure source-overlap and native-head-only baselines",
                "require full_minus_native_head_only >= 0.10 before package",
                "do not use sealed v3 to construct, sample, tune, or debug Phase2K",
            ],
        },
    }


def markdown_summary(manifest: dict[str, Any]) -> str:
    deltas = manifest["metrics"]["deltas"]
    lines = [
        "# Phase2J Stagegain Freeze",
        "",
        f"- Frozen: `{manifest['frozen']}`",
        f"- Evidence boundary: `{manifest['active_evidence_boundary']}`",
        f"- Full minus no-NSI latent: `{deltas['full_minus_no_nsi_latent']:.6f}`",
        f"- Full minus continuation-only: `{deltas['full_minus_continuation_only']:.6f}`",
        f"- Full minus native-head-only: `{deltas['full_minus_native_head_only']:.6f}`",
        "",
        "## Supported Claims",
    ]
    lines.extend(f"- {claim}" for claim in manifest["supported_claims"])
    lines += ["", "## Unsupported Claims"]
    lines.extend(f"- {claim}" for claim in manifest["unsupported_claims"])
    lines += ["", "## Checks"]
    lines.extend(f"- `{key}`: `{value}`" for key, value in manifest["checks"].items())
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze current Phase2J stagegain evidence.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--actiongate-report-dir", default=str(DEFAULT_ACTIONGATE_REPORT_DIR))
    parser.add_argument("--package-root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_phase2j_stagegain_freeze_manifest(
        report_dir=args.report_dir,
        actiongate_report_dir=args.actiongate_report_dir,
        package_root=args.package_root,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "phase2j_stagegain_freeze_manifest.json"
    md_path = output / "phase2j_stagegain_freeze_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_summary(manifest), encoding="utf-8")
    print(json.dumps({"manifest": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
