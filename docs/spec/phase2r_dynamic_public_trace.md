# Phase2R Dynamic Public Trace

Phase2R is the next non-sealed pressure benchmark after Phase2Q. Its purpose is narrow: test whether the bounded native nervous-interface mechanism still works when public repository traces include isolated dynamic pytest execution evidence, not only read-only static repository structure.

Phase2R does not claim production autonomy, open-ended repair, unrestricted shell use, or epoch-making architecture status. It only becomes claim-bearing if the dynamic collector, data-health audit, head-state baseline, pretrain gate, and design review all pass before any training.

## Required Boundary

- Sealed v3 remains final evaluation-only and cannot influence Phase2R collection, sampling, profile design, model selection, seed selection, or failure analysis feedback.
- Public repositories must be split-disjoint across train, validation, and holdout.
- Test execution must run in an isolated disposable sandbox, not in the source repository worktree.
- Collection must observe the source repository as read-only; any mutation blocks training.
- Every training row must contain dynamic execution evidence.
- Visible evidence must not contain candidate slot markers, gold labels, hidden hints, or sealed identifiers.
- Source-overlap, native-head-only, continuation-only, prompt-only, and ReAct baselines must be code-measured and below threshold.

## Current Implementation

- Dynamic collector: `src/reflexlm/cli/collect_phase2r_dynamic_public_repo_traces.py`
- Dynamic gate: `src/reflexlm/cli/audit_phase2r_dynamic_public_trace.py`
- Boundary writer: `src/reflexlm/cli/write_phase2r_dynamic_boundary.py`
- Targeted tests: `tests/test_phase2r_dynamic_public_trace.py`

Phase2R should reuse the Phase2M/Phase2Q native-head data path only after dynamic trace collection passes. The adapter/package/sealed sequence remains blocked until non-sealed smoke and full gates pass.

## Completed Result

Phase2R now has a full-density dynamic public-trace result for Qwen2.5-7B seed `13`.

- Collector/data: `1024 / 1024` rows include isolated dynamic pytest execution evidence. Source repositories are observed read-only; execution occurs in disposable sandboxes; sealed v3 is not used for collection, sampling, training, tuning, model selection, or failure feedback.
- Split: eight repo-disjoint public projects with `512 / 256 / 256` train/validation/holdout rows.
- Non-sealed validation: command-slot accuracy `1.000` over `256` rows, source-overlap baseline `0.34375`, `model_minus_source_overlap=0.65625`, native-head-only completion `0.37109375`, and `full_minus_native_head_only=0.62890625`.
- Public holdout: command-slot accuracy `1.000` over `256` rows, source-overlap holdout baseline `0.5`, and `model_minus_source_overlap_holdout=0.5`.
- Sealed v3 final evaluation: full package `64 / 64`, no-NSI `2 / 64`, native-head-only/no-cache `0 / 64`, continuation-only `0 / 64`, prompt-only `0 / 64`, ReAct `0 / 64`, allowlist hallucination `0`, and low-level Qwen calls `0`.

Primary artifacts:

- Full summary: `artifacts/reports/phase2r_dynamic_public_trace/phase2r_dynamic_public_trace_full_summary.md`
- Full postflight: `artifacts/reports/phase2r_dynamic_public_trace/phase2r_qwen2_5_7b/seed13/full/phase2n_phase2r_qwen2_5_7b_full_seed13_r16_alpha32_lr1e-4_len256.postflight.json`
- Holdout postflight: `artifacts/reports/phase2r_dynamic_public_trace/phase2r_qwen2_5_7b/seed13/full/phase2n_phase2r_qwen2_5_7b_full_seed13_r16_alpha32_lr1e-4_len256.holdout_postflight.json`
- Sealed table: `artifacts/reports/phase2r_external_trace_v3_semantic_required/phase2r_external_trace_v3_exact_baseline_table.md`
- Sealed gate: `artifacts/reports/phase2r_external_trace_v3_semantic_required/phase2r_external_trace_v3_gate.json`

This upgrades Phase2R to bounded dynamic public-trace evidence. It still does not prove production autonomy, open-ended repair, unrestricted shell use, independent external reproduction, modern coding-agent superiority, or epoch-making architecture status.
