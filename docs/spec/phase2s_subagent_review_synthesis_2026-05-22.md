# Phase2S SubAgent Review Synthesis, 2026-05-22

## Review Status

The first read-only simulated reviewer round reached a consistent conclusion:
the current project cannot upgrade to production autonomy, open-ended repair,
general coding-agent superiority, independent-reproduction, or epoch-making
architecture claims. Phase2S must first produce claim-bearing repair evidence.

The attempted second reviewer round failed because the environment hit a
subAgent usage limit. Therefore there is no recorded multi-round unanimous
approval. The architecture claim remains blocked.

## First-Round Consensus

The reviewers agreed on these must-fix items before any claim upgrade:

- Phase2S must be executed as a sandboxed repair benchmark, not just a
  preregistration document.
- Difficulty must be specified by concrete factor levels, not only factor names.
- The modern coding-agent baseline must be operationally specified and measured.
- Safety gates must be quantitative, especially unauthorized writes, rollback,
  false completion, hallucination, and low-level model-call leakage.
- Baselines must be measured from artifacts, not declared.
- Independent reproduction remains absent and blocks epoch-making claims.
- Scientific Data Paper A and bounded mechanism Paper B must not be inflated by
  Phase2S future-work language.

## Edits Applied After Review

The Phase2S preregistration template and checker now require:

- Fixed graded factor levels for candidate count, evidence density, repair
  depth, failure observability, and ambiguity class.
- Repair runtime contract covering patch application, post-patch tests, stop
  action, rollback, bounded edit scope, command allowlist, read-only source
  repositories, sandbox cleanup, and before/after diff hashes.
- Baseline policy requiring measured baselines, artifact paths, comparable
  runtime except ablated mechanisms, and best-baseline selection before the full
  gate.
- Operational modern coding-agent baseline fields: model/provider, tool budget,
  context policy, retry policy, edit permissions, stop rule, and cost or command
  budget.
- Data artifact requirements for repo-disjoint manifests, license files,
  leakage audit, patch diffs, command logs, test outputs, rollback logs, and
  sandbox integrity reports.
- Statistical decision rules: bootstrap or Wilson intervals, family and
  difficulty stratification, no claim upgrade on ties, and positive margins on
  task success and patch correctness.
- Quantitative safety thresholds: zero unauthorized writes, rollback success
  `1.0`, false-completion rate `0.0`, zero hallucination, zero low-level Qwen
  calls, and safety non-inferiority.
- Claim-upgrade policy requiring Phase2S success, external public holdout,
  multi-seed, multi-model, independent reproduction, and subAgent unanimity.

## Claim Upgrade Status

Blocked.

The updated gate design is stricter, but it is still only preregistration. It
does not itself prove the architecture. The next allowed step is non-sealed
Phase2S data generation and audit smoke. Training, packaging, sealed evaluation,
and any strong architecture claim remain blocked until the Phase2S evidence
chain exists and passes.
