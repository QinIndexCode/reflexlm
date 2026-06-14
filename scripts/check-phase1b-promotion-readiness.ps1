$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"

$reportDir = "artifacts/reports/phase1b_wide_ood_scenario_holdout_seed313_observable_v2_final"
$candidate = "nsi_v18_reflex_micro_observable_hash_hard6"
$baseline = "flat_v3_slot_focus"
$output = Join-Path $reportDir "promotion_readiness.$candidate.json"

python -m reflexlm.cli.check_promotion_readiness `
  --all-summary (Join-Path $reportDir "gain_matrix_summary.flat_v3_slot_focus.json") `
  --reflex-layer-summary (Join-Path $reportDir "gain_matrix_summary.reflex_layer.flat_v3_slot_focus.json") `
  --debug-cortex-summary (Join-Path $reportDir "gain_matrix_summary.debug_cortex.flat_v3_slot_focus.json") `
  --candidate-label $candidate `
  --baseline-label $baseline `
  --phase2-pause-lock "artifacts/control/phase2_7b.paused" `
  --require-all-seed-gates `
  --min-reflex-layer-completion 0.95 `
  --min-common-recovery-completion 0.95 `
  --output-json $output

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host "Promotion readiness report written to $output"
