# Baseline-zero interpretability protocol

This protocol is part of the Paper B claim boundary. It exists because several
sealed-final controls legitimately score `0`, but a zero-valued control is not
automatically stronger evidence.

## Required classifications

Each zero-valued control must be assigned one of four categories:

- `expected_zero_due_to_missing_capability`: the control lacks a required
  mechanism such as NSI latent state, continuation memory, or native heads.
- `not_evaluable_for_control`: the task definition excludes information the
  control would need, so the result must not be used as a performance delta.
- `valid_zero_failure`: the control was evaluable and failed; this can support a
  bounded comparison but not a broad autonomy claim.
- `suspicious_zero_requires_redesign`: the zero lacks a sufficient explanation
  or sanity subset and blocks claim use.

## Claim-use rules

- If `native-head-only` and `no-NSI` are both zero on a profile, the paper must
  show a graded sanity subset where at least one of those controls can score
  above zero, or the profile is too brittle for delta claims.
- If all sealed controls are zero, a bounded claim also requires a separate
  zero-root audit plus a non-sealed baseline-feasibility sanity audit showing
  that comparable controls can score above zero on a preregistered non-sealed
  subset. Otherwise the result is classified as
  `suspicious_zero_requires_redesign`.
- If the full package is zero, the phase is failure evidence and cannot be used
  as positive transfer evidence.
- Prompt-only and ReAct zeros can be reported as text-baseline failures, but
  they do not by themselves prove architectural necessity.
- Any `suspicious_zero_requires_redesign` row blocks stronger claim language
  until the data or control design is fixed.

## Implementation

Run:

```powershell
python -m reflexlm.cli.audit_paper_baseline_zero
```

The generated JSON and Markdown reports are written under
`artifacts/reports/paper_b_baseline_zero_audit/`. The report is evidence for
Paper B; it is not a training or tuning signal and must not be used to redesign
sealed-v3 data.

Phase2T additionally records this explicit sanity layer at
`artifacts/reports/phase2t_external_trace_v3_semantic_required/phase2t_baseline_feasibility_sanity_audit.json`.
That artifact permits only a bounded sealed claim with a zero-control caveat;
it does not permit a production-autonomy, open-ended repair, or epoch-making
architecture claim.
