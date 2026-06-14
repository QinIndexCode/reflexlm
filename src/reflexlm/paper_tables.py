from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexlm.paper_baseline_audit import build_baseline_zero_audit


@dataclass(frozen=True)
class PaperTable:
    slug: str
    caption: str
    label: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def _load_json(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _finite(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            result.append(float(value))
    return result


def _fmt_min(values: list[Any]) -> str:
    finite = _finite(values)
    return _fmt(min(finite) if finite else None)


def _fmt_mean(values: list[Any]) -> str:
    finite = _finite(values)
    return _fmt(sum(finite) / len(finite) if finite else None)


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


_ZERO_CATEGORY_LABELS = {
    "expected_zero_due_to_missing_capability": "expected missing capability",
    "not_evaluable_for_control": "not evaluable for control",
    "valid_zero_failure": "valid zero failure",
    "suspicious_zero_requires_redesign": "suspicious; redesign required",
}


_CLAIM_BLOCKER_LABELS = {
    "do_not_use_this_phase_as_positive_sealed_transfer_evidence": (
        "negative sealed-transfer evidence only"
    ),
}


def _humanize_evidence_label(value: str) -> str:
    if value in _ZERO_CATEGORY_LABELS:
        return _ZERO_CATEGORY_LABELS[value]
    if value in _CLAIM_BLOCKER_LABELS:
        return _CLAIM_BLOCKER_LABELS[value]
    return value.replace("_", " ")


def build_claim_boundary_table() -> PaperTable:
    return PaperTable(
        slug="claim_boundary",
        caption="Claim boundary for the bounded mechanism paper.",
        label="tab:claim-boundary",
        headers=("Status", "Claim", "Paper use"),
        rows=(
            (
                "supported",
                "Bounded native-head Debug Cortex command selection",
                "Main positive mechanism claim",
            ),
            (
                "supported",
                "NSI latent contribution versus no-NSI on semantic-required controls",
                "Mechanism delta claim with measured ablations",
            ),
            (
                "supported",
                "Local multi-model and multi-seed robustness for bounded command selection",
                "Robustness layer; independent reproduction remains separate",
            ),
            (
                "unsupported",
                "Production autonomy, unrestricted shell use, open-ended repair",
                "Explicitly excluded from Paper B",
            ),
            (
                "unsupported",
                "Epoch-making architecture status",
                "Requires independent reproduction, broader open-ended repair, safety analysis, and stronger baselines",
            ),
        ),
    )


def build_positive_evidence_table(root: str | Path = ".") -> PaperTable:
    repo_root = Path(root)
    p2m = _load_json(
        repo_root,
        "artifacts/reports/phase2m_v2_claim_bearing/phase2m_v2_public_relationkey_full_postflight.json",
    )
    p2p = _load_json(
        repo_root,
        "artifacts/reports/phase2p_sealed_cross_model_transfer/phase2p_multiseed_cross_model_transfer_summary.json",
    )
    p2q = _load_json(
        repo_root,
        "artifacts/reports/phase2q_public_trace_breadth/phase2q_public_trace_breadth_full_summary.json",
    )
    p2r = _load_json(
        repo_root,
        "artifacts/reports/phase2r_dynamic_public_trace/phase2r_dynamic_public_trace_full_summary.json",
    )
    p2s = _load_json(
        repo_root,
        "artifacts/reports/phase2s_multimodel_multiseed_reproduction/phase2s_cross_model_multiseed_reproduction_report.json",
    )
    p2m_metrics = p2m.get("metrics", {})
    p2p_agg = p2p.get("aggregate", {})
    p2q_nonsealed = p2q.get("full_nonsealed", {})
    p2q_sealed = p2q.get("sealed_v3", {})
    p2r_nonsealed = p2r.get("nonsealed", {})
    p2r_sealed = p2r.get("sealed_v3", {})
    p2s_runs = p2s.get("runs", [])
    return PaperTable(
        slug="positive_evidence_matrix",
        caption="Positive evidence matrix for bounded command-selection claims.",
        label="tab:positive-evidence",
        headers=("Layer", "Non-sealed full", "Source/native pressure", "Sealed full", "Boundary"),
        rows=(
            (
                "Phase2M-v2",
                _fmt(p2m_metrics.get("val_command_slot_accuracy")),
                f"source {_fmt(p2m_metrics.get('source_overlap_val_accuracy'))}; native {_fmt(p2m_metrics.get('native_head_only_completion'))}",
                "1.000",
                "single local 7B run",
            ),
            (
                "Phase2P",
                "reuses Phase2M-v2 family",
                f"no-NSI max {_fmt(p2p_agg.get('no_nsi_completion_max'))}; native max {_fmt(p2p_agg.get('native_head_only_completion_max'))}",
                f"min {_fmt(p2p_agg.get('full_completion_min'))}; 15/15 gates",
                "local cross-model transfer, not independent reproduction",
            ),
            (
                "Phase2Q",
                _fmt(p2q_nonsealed.get("val_command_slot_accuracy")),
                f"source {_fmt(p2q_nonsealed.get('source_overlap_val_accuracy'))}; native {_fmt(p2q_nonsealed.get('native_head_only_completion'))}",
                _fmt(p2q_sealed.get("full_completion")),
                "static public trace breadth",
            ),
            (
                "Phase2R",
                _fmt(p2r_nonsealed.get("val_command_slot_accuracy")),
                f"source {_fmt(p2r_nonsealed.get('source_overlap_val_accuracy'))}; native {_fmt(p2r_nonsealed.get('native_head_only_completion'))}",
                _fmt(p2r_sealed.get("full_completion")),
                "dynamic pytest traces, still command selection",
            ),
            (
                "Phase2S",
                f"holdout min {_fmt_min([run.get('holdout_command_slot_accuracy') for run in p2s_runs])}; mean {_fmt_mean([run.get('holdout_command_slot_accuracy') for run in p2s_runs])}",
                f"source-delta min {_fmt_min([run.get('holdout_model_minus_source_overlap_accuracy') for run in p2s_runs])}; no-NSI-delta min {_fmt_min([run.get('holdout_model_minus_zero_nsi_accuracy') for run in p2s_runs])}",
                "separate final-eval evidence only",
                "3B/7B x 3 seeds on non-sealed public repair split; same-family, not production autonomy",
            ),
        ),
    )


def build_negative_evidence_table() -> PaperTable:
    return PaperTable(
        slug="negative_evidence",
        caption="Negative and bounded evidence retained as claim boundary.",
        label="tab:negative-evidence",
        headers=("Phase", "Observed limit", "Interpretation"),
        rows=(
            (
                "Phase2I",
                "full equals no-NSI on sealed v3; latent split lacked command identity",
                "not evidence for semantic-required NSI latent necessity",
            ),
            (
                "Phase2J initial smoke",
                "model matched source-overlap baseline",
                "blocked full training until source-overlap-hard redesign",
            ),
            (
                "Phase2K sealed gate",
                "full did not beat native-head-only on sealed v3",
                "non-sealed continuation pressure did not transfer",
            ),
            (
                "Phase2L sealed gate",
                "all six mechanisms completed 0/64",
                "sealed transfer failure, not positive continuation proof",
            ),
            (
                "Phase2M synthetic-safe smoke",
                "synthetic plumbing split exposed candidate-slot leakage risk",
                "infrastructure smoke only, not claim-bearing evidence",
            ),
        ),
    )


def build_baseline_zero_summary_table(root: str | Path = ".") -> PaperTable:
    report = build_baseline_zero_audit(root)
    rows = []
    for phase in report["phases"]:
        categories = sorted(
            {
                row["zero_category"]
                for row in phase["rows"]
                if row.get("zero_category") is not None
            }
        )
        blockers = (
            "; ".join(
                _humanize_evidence_label(blocker)
                for blocker in phase["claim_blockers"]
            )
            if phase["claim_blockers"]
            else "none"
        )
        rows.append(
            (
                phase["phase"],
                "yes" if phase["native_and_no_nsi_both_zero"] else "no",
                "yes" if phase["full_zero"] else "no",
                "; ".join(_humanize_evidence_label(category) for category in categories)
                if categories
                else "none",
                blockers,
            )
        )
    return PaperTable(
        slug="baseline_zero_summary",
        caption="Interpretability audit for zero-valued controls.",
        label="tab:baseline-zero-summary",
        headers=("Phase", "Native and no-NSI both zero", "Full zero", "Zero categories", "Claim blocker"),
        rows=tuple(rows),
    )


def build_homeostasis_mechanism_evidence_table(root: str | Path = ".") -> PaperTable:
    repo_root = Path(root)
    readiness = _load_json(
        repo_root,
        "artifacts/reports/phase2homeostasis/"
        "phase2homeostasis_hmac_v3_controller_v4_bounded_mechanism_readiness.json",
    )
    dossier = _load_json(
        repo_root,
        "artifacts/reports/phase2homeostasis/"
        "phase2homeostasis_hmac_v3_controller_v4_bounded_publication_dossier.json",
    )
    readiness_metrics = readiness.get("metrics", {})
    dossier_metrics = dossier.get("metrics", {})
    core_evidence = dossier.get("core_positive_evidence", [])
    runtime_metrics = next(
        (
            row.get("compact_metrics", {})
            for row in core_evidence
            if row.get("role") == "runtime_generation1_py313"
        ),
        {},
    )
    return PaperTable(
        slug="homeostasis_mechanism_evidence",
        caption="Bounded homeostatic persistent-state mechanism evidence and limits.",
        label="tab:homeostasis-mechanism-evidence",
        headers=("Evidence layer", "Positive result", "Retained limitation", "Paper use"),
        rows=(
            (
                "Fresh bounded runtime",
                (
                    f"{runtime_metrics.get('episodes', 'NA')} episodes; "
                    f"{runtime_metrics.get('executed_actions', 'NA')} actions; "
                    f"completion {_fmt(runtime_metrics.get('task_completion_success_rate'))}"
                ),
                "controlled tasks only",
                "behavioral completion evidence",
            ),
            (
                "Authenticated persistence",
                (
                    f"{dossier_metrics.get('hmac_state_artifact_count', 'NA')} "
                    "HMAC-authenticated state artifacts; missing-key control fails closed"
                ),
                "bounded package/config state, not semantic long-term memory",
                "persistent-state transfer mechanism",
            ),
            (
                "Cross-runtime limitation",
                (
                    f"{readiness_metrics.get('fresh_limitation_side_effect_trace_rows', 'NA')} "
                    "side-effect trace rows retained"
                ),
                (
                    "exact internal dynamics mismatch; max active-threshold delta "
                    f"{_fmt(readiness_metrics.get('maximum_active_threshold_delta'), 6)} "
                    "under limitation threshold "
                    f"{_fmt(readiness_metrics.get('fresh_limitation_threshold_drift_limit'), 6)}"
                ),
                "explicit negative evidence",
            ),
            (
                "Dossier replay",
                (
                    f"{readiness_metrics.get('bounded_manifest_replay_source_report_count', 'NA')} "
                    "source reports replayed by hash in a distinct directory"
                ),
                "external-machine replay not yet run",
                "local reproducibility, not independent replication",
            ),
        ),
    )


def build_all_paper_tables(root: str | Path = ".") -> list[PaperTable]:
    return [
        build_claim_boundary_table(),
        build_positive_evidence_table(root),
        build_negative_evidence_table(),
        build_baseline_zero_summary_table(root),
        build_homeostasis_mechanism_evidence_table(root),
    ]


def table_to_latex(table: PaperTable) -> str:
    column_widths_by_count = {
        3: (0.13, 0.36, 0.34),
        4: (0.16, 0.24, 0.25, 0.18),
        5: (0.10, 0.16, 0.21, 0.13, 0.22),
    }
    widths = column_widths_by_count.get(
        len(table.headers),
        tuple(0.82 / len(table.headers) for _ in table.headers),
    )
    align = "@{}" + "".join(
        rf">{{\raggedright\arraybackslash}}p{{{width:.2f}\linewidth}}"
        for width in widths
    ) + "@{}"
    size = r"\footnotesize" if len(table.headers) >= 5 else r"\small"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        size,
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{_escape_latex(table.caption)}}}",
        rf"\label{{{table.label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_escape_latex(header) for header in table.headers) + r" \\",
        r"\midrule",
    ]
    for row in table.rows:
        lines.append(" & ".join(_escape_latex(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_paper_tables(
    *,
    output_dir: str | Path = "docs/paper_b/tables",
    manifest_json: str | Path = "docs/paper_b/tables/table_manifest.json",
    root: str | Path = ".",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tables = build_all_paper_tables(root)
    manifest = {
        "table_pipeline": "artifact_json_to_latex_tables",
        "output_dir": str(output_path),
        "tables": {},
    }
    for table in tables:
        table_path = output_path / f"{table.slug}.tex"
        table_path.write_text(table_to_latex(table), encoding="utf-8")
        manifest["tables"][table.slug] = {
            "path": str(table_path),
            "caption": table.caption,
            "label": table.label,
            "row_count": len(table.rows),
        }
    manifest_path = Path(manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
