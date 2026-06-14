from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path("artifacts/reports/phase2ae_structural_sidecar_budget_pressure_patch_candidates")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/phase2ae_structural_sidecar_freeze")
DEFAULT_PACKAGE_DIR = Path(
    "artifacts/packages/phase2ae_structural_sidecar_nervous_qwen3b_v2_slotbalanced/"
    "phase2ae_structural_sidecar_qwen3b_v2_slotbalanced_e5"
)
DEFAULT_NO_NSI_PACKAGE_DIR = Path(
    "artifacts/packages/phase2ae_structural_sidecar_nervous_qwen3b_v2_slotbalanced/"
    "phase2ae_structural_sidecar_qwen3b_v2_slotbalanced_e5_no_nsi"
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
    candidate = Path(path)
    return {
        "path": str(candidate),
        "exists": candidate.exists(),
        "sha256": _sha256(candidate) if candidate.exists() and candidate.is_file() else None,
    }


def _package(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    return {
        "path": str(root),
        "exists": root.exists(),
        "manifest": _artifact(root / "native_nervous_package.json"),
    }


def build_phase2ae_structural_sidecar_freeze_manifest(
    *,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    package_dir: str | Path = DEFAULT_PACKAGE_DIR,
    no_nsi_package_dir: str | Path = DEFAULT_NO_NSI_PACKAGE_DIR,
) -> dict[str, Any]:
    reports = Path(report_dir)
    artifact_paths = {
        "full_execution_summary": reports
        / "phase2ae_prov_holdout45_qwen3b_v2_slotbalanced_noretry.runtimefix.execution_summary.json",
        "full_learning_gap_audit": reports
        / "phase2ae_qwen3b_v2_slotbalanced_noretry_runtimefix_learning_gap_audit.json",
        "no_nsi_textablated_execution_summary": reports
        / "phase2ae_prov_holdout45_qwen3b_v2_slotbalanced_noretry.runtimefix.no_nsi_textablated.execution_summary.json",
        "no_nsi_textablated_learning_gap_audit": reports
        / "phase2ae_qwen3b_v2_slotbalanced_noretry_runtimefix_no_nsi_textablated_learning_gap_audit.json",
        "full_vs_no_nsi_textablated_baseline_report": reports
        / "phase2ae_qwen3b_v2_slotbalanced_runtimefix_full_vs_no_nsi_textablated_baseline_report.json",
        "slot_support_audit": reports / "phase2ae_qwen3b_v2_slotbalanced_slot_support_audit.json",
        "training_summary": reports / "phase2ae_structural_sidecar_qwen3b_v2_slotbalanced.training_summary.json",
    }
    missing = [name for name, path in artifact_paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing Phase2AE freeze artifacts: " + ", ".join(missing))

    full = _load(artifact_paths["full_execution_summary"])
    no_nsi = _load(artifact_paths["no_nsi_textablated_execution_summary"])
    baseline = _load(artifact_paths["full_vs_no_nsi_textablated_baseline_report"])
    full_learning = _load(artifact_paths["full_learning_gap_audit"])
    no_nsi_learning = _load(artifact_paths["no_nsi_textablated_learning_gap_audit"])
    slot_support = _load(artifact_paths["slot_support_audit"])
    training = _load(artifact_paths["training_summary"])

    full_accuracy = float(full.get("patch_candidate_selection_accuracy") or 0.0)
    no_nsi_accuracy = float(no_nsi.get("patch_candidate_selection_accuracy") or 0.0)
    source_ablated = float(
        baseline.get("baseline_metrics", {})
        .get("source_overlap_identity_text_ablated", {})
        .get("accuracy")
        or 0.0
    )
    raw_source = float(
        baseline.get("baseline_metrics", {}).get("source_overlap", {}).get("accuracy") or 0.0
    )
    identity_heuristic = float(
        baseline.get("baseline_metrics", {})
        .get("runtime_identity_heuristic", {})
        .get("accuracy")
        or 0.0
    )
    by_slot = baseline.get("by_expected_slot", {})
    slot3 = by_slot.get("3", {}) if isinstance(by_slot, dict) else {}
    slot3_total = int(slot3.get("total") or 0)
    slot3_identity = int(slot3.get("runtime_identity_heuristic") or 0)
    history = training.get("history") if isinstance(training.get("history"), list) else []
    final_history = history[-1] if history and isinstance(history[-1], dict) else {}
    final_val = (
        final_history.get("val_metrics")
        if isinstance(final_history.get("val_metrics"), dict)
        else {}
    )

    return {
        "freeze_family": "phase2ae_structural_sidecar_freeze",
        "frozen": True,
        "active_evidence_boundary": "bounded_patch_candidate_selection_not_freeform_patch_generation",
        "sealed_feedback_used": False,
        "supported_claims": [
            "bounded public-repo patch-candidate selection executes end-to-end on clean holdout clones",
            "text-ablated no-NSI/candidate-identity control fails the structural slot3 pressure subset",
            "structured runtime sidecar wiring contributes to the Phase2AE candidate-selection result",
        ],
        "unsupported_claims": [
            "freeform patch generation",
            "production autonomy",
            "open-ended debugging generalization",
            "sealed transfer",
            "epoch-making architecture",
            "learned native head beats a raw runtime identity heuristic sidecar",
        ],
        "claim_boundary": (
            "Phase2AE supports bounded candidate selection under structural sidecar pressure. "
            "It does not prove freeform repair or learned-head superiority over an explicit runtime identity heuristic."
        ),
        "checks": {
            "all_required_artifacts_present": not missing,
            "training_summary_passed_val_gate": final_val.get("command_slot_accuracy") == 1.0,
            "slot_support_audit_passed": slot_support.get("passed") is True,
            "full_execution_45_of_45": full.get("rows") == 45
            and full.get("correct_patch_candidate_selections") == 45,
            "full_learning_gap_passed": full_learning.get("passed") is True,
            "no_nsi_textablated_below_full": no_nsi_accuracy < full_accuracy,
            "no_nsi_textablated_below_085": no_nsi_accuracy < 0.85,
            "identity_text_ablated_source_overlap_below_full": source_ablated < full_accuracy,
            "raw_source_overlap_contaminated_by_identity_sidecar": raw_source == full_accuracy,
            "runtime_identity_heuristic_solves_split": identity_heuristic == full_accuracy,
            "slot3_requires_identity_sidecar_under_text_ablated_control": (
                slot3_total > 0 and slot3_identity == slot3_total
            ),
            "no_nsi_textablated_learning_gap_fails": no_nsi_learning.get("passed") is True
            and no_nsi_learning.get("checks", {}).get("initial_policy_selection_supports_learned_head_claim")
            is False,
        },
        "metrics": {
            "full_patch_candidate_selection_accuracy": full_accuracy,
            "no_nsi_textablated_patch_candidate_selection_accuracy": no_nsi_accuracy,
            "full_minus_no_nsi_textablated": full_accuracy - no_nsi_accuracy,
            "source_overlap_identity_text_ablated_accuracy": source_ablated,
            "raw_source_overlap_accuracy": raw_source,
            "runtime_identity_heuristic_accuracy": identity_heuristic,
            "slot_distribution": {
                "training": training.get("slot_intent_distribution"),
                "baseline_by_expected_slot": by_slot,
            },
        },
        "artifacts": {name: _artifact(path) for name, path in artifact_paths.items()},
        "packages": {
            "full": _package(package_dir),
            "no_nsi_textablated": _package(no_nsi_package_dir),
        },
        "next_step": {
            "phase": "phase2af_hardened_structural_sidecar_pressure",
            "allowed": True,
            "constraints": [
                "non-sealed public-repo traces only",
                "do not use sealed failures to construct or tune the split",
                "same-intent candidates must share visible lexical/source-overlap cues",
                "pretrain gate must reject raw source-overlap ceiling and raw all-zero controls",
                "claim remains bounded until full beats text-ablated no-NSI and measured baselines",
            ],
        },
    }


def markdown_summary(manifest: dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    lines = [
        "# Phase2AE Structural Sidecar Freeze",
        "",
        f"- Frozen: `{manifest['frozen']}`",
        f"- Evidence boundary: `{manifest['active_evidence_boundary']}`",
        f"- Full selection accuracy: `{metrics['full_patch_candidate_selection_accuracy']}`",
        f"- No-NSI text-ablated selection accuracy: `{metrics['no_nsi_textablated_patch_candidate_selection_accuracy']}`",
        f"- Full minus no-NSI text-ablated: `{metrics['full_minus_no_nsi_textablated']}`",
        f"- Identity-text-ablated source-overlap: `{metrics['source_overlap_identity_text_ablated_accuracy']}`",
        f"- Raw source-overlap: `{metrics['raw_source_overlap_accuracy']}`",
        f"- Runtime identity heuristic: `{metrics['runtime_identity_heuristic_accuracy']}`",
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
    parser = argparse.ArgumentParser(description="Freeze Phase2AE structural sidecar evidence.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE_DIR))
    parser.add_argument("--no-nsi-package-dir", default=str(DEFAULT_NO_NSI_PACKAGE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = build_phase2ae_structural_sidecar_freeze_manifest(
        report_dir=args.report_dir,
        package_dir=args.package_dir,
        no_nsi_package_dir=args.no_nsi_package_dir,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "phase2ae_structural_sidecar_freeze_manifest.json"
    md_path = output / "phase2ae_structural_sidecar_freeze_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_summary(manifest), encoding="utf-8")
    print(json.dumps({"manifest": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
