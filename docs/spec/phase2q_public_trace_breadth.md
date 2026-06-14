# Phase2Q Public Trace Breadth

## Purpose

Phase2Q tests whether the Phase2M/Phase2P bounded mechanism evidence survives a wider public/read-only repository trace setting before claiming broader real-repo generalization. It is a preregistered non-sealed pressure step, not a sealed-v3 tuning loop.

## Boundary

- Sealed v3 remains final-evaluation-only and must not influence data construction, model choice, seed choice, hyperparameters, failure analysis, or sampling.
- Phase2Q smoke success only permits full non-sealed training.
- Package and sealed final evaluation are blocked until full non-sealed validation and holdout gates pass.
- Production autonomy, open-ended repair, unrestricted shell use, independent external reproduction, and epoch-making architecture claims remain unsupported.

## Public Trace Breadth Gate

The breadth gate requires:

- At least eight public/read-only repositories.
- Repo-disjoint train, validation, and holdout splits.
- At least four train repos, two validation repos, and two holdout repos.
- At least `96` train rows, `48` validation rows, and `48` holdout rows.
- Structured watch keys enabled.
- Test-body behavior summaries suppressed.
- No collection rejections.
- No sealed references, gold hints, candidate-slot markers, or candidate index markers in runtime-visible text.
- Measured source-overlap, prompt, ReAct, continuation, and native-head-only baselines, not declared-only baselines.
- Source-overlap and native-head-only baselines below the registered threshold.

## Current Smoke Evidence

The current Phase2Q public trace breadth split passes data health, breadth, pretrain, head-state source-overlap, and design maturity gates. It uses four train repos (`pallets_click`, `pallets_itsdangerous`, `pallets_jinja`, `pytest_pluggy`), two validation repos (`pallets_markupsafe`, `pytest_iniconfig`), and two holdout repos (`pypa_packaging`, `pallets_flask`).

The Qwen2.5-7B seed `13` smoke run passes non-sealed postflight with validation command-slot accuracy `1.000` over `48` records, source-overlap validation baseline `0.3125`, and `model_minus_source_overlap=0.6875`. The run uses LoRA r16/alpha32, lr `1e-4`, max length `256`, max train records `128`, pairwise disabled, and additive latent fusion.

The initial smoke pass authorized full-density training. The full-density run then used `512` train rows, `256` validation rows, and `256` public holdout rows. It passed full non-sealed validation with command-slot accuracy `1.000`, source-overlap validation baseline `0.34765625`, and `full_minus_native_head_only=0.66796875`; it also passed holdout with command-slot accuracy `1.000` against source-overlap holdout baseline `0.3828125`.

After package, sealed `external_trace_v3_semantic_required` passed for the Phase2Q package: full `64 / 64`, prompt-only `0 / 64`, ReAct `0 / 64`, no-NSI `0 / 64`, native-head-only/no-cache `0 / 64`, and continuation-only `0 / 64`, with allowlist hallucination `0` and low-level Qwen calls `0`.

This upgrades Phase2Q from smoke-only to bounded public/read-only trace breadth evidence. It still does not prove production autonomy, open-ended dynamic debugging, unrestricted shell use, or independent external reproduction.

## Full Non-Sealed Inputs

After the smoke pass, the full non-sealed run must use Phase2Q-specific controls:

- Validation: `artifacts/datasets/phase2q_public_trace_breadth_heads/val.jsonl`
- Native-head-only zero-NSI validation control: `artifacts/datasets/phase2q_public_trace_breadth_heads/val.native_head_only_zero_nsi.jsonl`
- Holdout: `artifacts/datasets/phase2q_public_trace_breadth_holdout_heads/val.jsonl`

These inputs must not be replaced with Phase2M-v2 relation-key control files. The zero-NSI control is derived from the Phase2Q validation head split by erasing `nsi_reference` and adding `native_head_only_zero_nsi_control` to `runtime_overrides`; it does not change the gold command, candidate commands, or runtime-visible text.
