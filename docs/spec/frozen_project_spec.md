# Phase 1 Frozen Project Spec

This file freezes the repository-wide facts that are allowed to drive implementation. It is derived from the Chinese research document and intentionally separates facts from plans.

## Canonical facts

- The canonical research source is [native_synaptic_interface_research.md](../../native_synaptic_interface_research.md).
- Phase 1 scope is limited to terminal, process, filesystem, and time state.
- Phase 1 does not include GUI perception, robotics, or full-scale LLM training.
- The fixed Phase 1 action space is:
  - `WAIT`
  - `READ_STDOUT`
  - `READ_STDERR`
  - `READ_FILE`
  - `RUN_COMMAND`
  - `STOP_PROCESS`
  - `ASK_USER`
  - `REFRESH_STATE`
  - `BLOCK`
  - `DONE`
- `RUN_COMMAND` must select from a predefined allowlist. Free-form shell synthesis is out of scope.
- The Phase 1 task suite consists of:
  - blocking input detection
  - test failure reflex
  - process hang detection
  - dangerous action interception
  - external file change reflex
  - common error recovery routines
- The evaluation metrics are:
  - reaction latency
  - token-equivalent cost
  - model calls
  - recovery success rate
  - false reflex rate
  - dangerous action block rate
  - long-run stability
  - state hallucination rate
  - stale state action rate
  - task completion rate

## Initial plans now implemented or superseded

- The project now contains generated Phase 1, Phase 1B, and Phase 2 artifacts under `artifacts/`.
- The current strongest small-model evidence is `nsi_v20_debug_lexical_tiny` on the Phase 1B debug-v3 `wide_ood` scenario-heldout split.
- `Qwen/Qwen2.5-7B-Instruct` prompt-only, ReAct, shared-adapter, and hybrid evaluations have been run locally on the fixed Phase 1B debug-v3 test split.
- The originally planned `Qwen/Qwen2.5-1.5B-Instruct` fusion stage is not the current main evidence path; the completed local large-model evidence is the 7B QLoRA/hybrid validation.

## Explicit non-facts and current boundaries

- The completed 7B validation does not prove a full large-model NSI advantage.
- The small-model results do not prove production robustness, open-ended debugging, GUI operation, robotics, free-form shell generation, or AGI-like cognition.
- A result must not be treated as evidence unless it is backed by a run directory, JSON summary, and fixed-split evaluation artifact.
- The previous English draft must not be treated as evidence where it contained predicted values, placeholder tables, or pre-filled success criteria.
