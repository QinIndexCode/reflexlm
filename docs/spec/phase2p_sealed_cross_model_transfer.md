# Phase2P Sealed Cross-Model Transfer

## Goal

Phase2P tests one narrow question: whether the bounded semantic-required Debug Cortex/native nervous-interface mechanism transfers across local backbone families and seeds on the sealed `external_trace_v3_semantic_required` benchmark.

This phase does not test production autonomy, unrestricted shell generation, JSON motor-output generality, GUI/robotics, or open-ended software repair.

## Boundary

- Sealed v3 is final evaluation only.
- Sealed results must not affect training data, sampling, hyperparameters, model selection, or seed selection.
- Prompt-only and ReAct text baselines are adapter-seed independent; seed29/47 small-model gates reuse the preregistered seed13 text baseline for the same base model.
- Any failed gate would freeze a transfer failure and stop claim upgrade.

## Registered Model/Seed Matrix

| Model | Family | Seeds | Evidence role |
|---|---|---|---|
| Qwen2.5-1.5B | Qwen2.5 | `13`, `29`, `47` | small same-family sealed transfer |
| Qwen2.5-3B | Qwen2.5 | `13`, `29`, `47` | medium same-family sealed transfer |
| Qwen2.5-7B | Qwen2.5 | `13`, `29`, `47` | reference sealed transfer |
| TinyLlama-1.1B | TinyLlama | `13`, `29`, `47` | non-Qwen sealed transfer probe |
| SmolLM2-360M | SmolLM2 | `13`, `29`, `47` | non-Qwen sealed transfer probe |

## Gate

- Full completion `>= 0.85`.
- Full minus no-NSI `>= 0.15`.
- Full minus native-head-only/no-cache `>= 0.10`.
- Full minus continuation-only `>= 0.15`.
- Full minus best text baseline `>= 0.15`.
- Allowlist hallucination `0`.
- Low-level Qwen calls `0`.
- Qwen/native-head calls only on Debug Cortex route.

## Result

The summary artifact `artifacts/reports/phase2p_sealed_cross_model_transfer/phase2p_multiseed_cross_model_transfer_summary.md` records all `15 / 15` gates passing:

- Full completion mean and minimum are `1.000`.
- Prompt-only and ReAct maxima are `0.000`.
- Native-head-only/no-cache and continuation-only maxima are `0.000`.
- no-NSI maximum is `0.6875`.
- Minimum `full_minus_no_nsi` is `0.3125`.
- Minimum `full_minus_native_head_only` and `full_minus_continuation_only` are both `1.000`.
- Allowlist hallucination is `0`.
- Low-level Qwen calls are `0`.

## Interpretation

Supported: sealed cross-model transfer for the bounded semantic-required Debug Cortex/native nervous-interface mechanism across the registered local backbones and seeds.

Still unsupported: production autonomy, open-ended debugging generalization, unrestricted shell use, broad public real-repo coverage, independent external reproduction, or epoch-making architecture status.
