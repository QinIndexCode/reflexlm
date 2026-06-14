param(
    [string]$PythonExe = "python",
    [string]$OutputDataset = "artifacts\datasets\phase1b_wide_ood_scenario_holdout_seed313",
    [int]$DatasetSeed = 313,
    [string]$LegacyTrainDataset = "artifacts\datasets\phase1_harder_observable_ood_fingerprint\train.jsonl",
    [string]$RunRoot = "artifacts\runs_phase1b_wide_ood",
    [string]$ReportRoot = "artifacts\reports",
    [string[]]$Seeds = @("13", "29", "47"),
    [int]$Epochs = 3,
    [int]$BatchSize = 8,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
function Resolve-RepoPath([string]$PathValue, [bool]$MustExist = $true) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        if ($MustExist) {
            return (Resolve-Path $PathValue -ErrorAction Stop).Path
        }
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    $fullPath = Join-Path $repoRoot $PathValue
    if ($MustExist) {
        return (Resolve-Path $fullPath -ErrorAction Stop).Path
    }
    return [System.IO.Path]::GetFullPath($fullPath)
}

$pythonPath = if ($PythonExe -eq "python") { "python" } else { Resolve-RepoPath $PythonExe }
$datasetPath = Resolve-RepoPath $OutputDataset $false
$legacyTrainPath = Resolve-RepoPath $LegacyTrainDataset
$runRootPath = Resolve-RepoPath $RunRoot $false
$reportRootPath = Resolve-RepoPath $ReportRoot $false
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }

New-Item -ItemType Directory -Force -Path $datasetPath, $runRootPath, $reportRootPath | Out-Null

& $pythonPath -m reflexlm.cli.generate_dataset `
    --output $datasetPath `
    --seed $DatasetSeed `
    --profile wide_ood `
    --split-strategy scenario_holdout
if ($LASTEXITCODE -ne 0) {
    throw "Phase 1B dataset generation failed."
}

$leakageReport = Join-Path $reportRootPath "phase1b_wide_ood_scenario_holdout_seed$DatasetSeed`_leakage.json"
& $pythonPath -m reflexlm.cli.analyze_dataset_leakage `
    --dataset-dir $datasetPath `
    --output-json $leakageReport
if ($LASTEXITCODE -ne 0) {
    throw "Phase 1B leakage analysis failed."
}

$wideTrain = Join-Path $datasetPath "train.jsonl"
$wideTest = Join-Path $datasetPath "test.jsonl"
$matrixLabels = "flat_v3_slot_focus,nsi_v8_reflex_micro,nsi_v9_reflex_micro_no_task_route,nsi_v10_reflex_micro_no_semantic_slots"
$seedArg = ($Seeds -join ",")

$zeroShotReport = Join-Path $reportRootPath "phase1b_wide_ood_zero_shot_seed$DatasetSeed"
$zeroShotRunRoot = Join-Path $runRootPath "zero_shot_seed$DatasetSeed"
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run-small-model-gain-iteration.ps1") `
    -PythonExe $pythonPath `
    -TrainDataset $legacyTrainPath `
    -TestDataset $wideTest `
    -EnvProfile wide_ood `
    -Seeds $seedArg `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -Device $Device `
    -MatrixLabels $matrixLabels `
    -SummaryBaselines flat_v3_slot_focus `
    -ReportDir $zeroShotReport `
    -RunRoot $zeroShotRunRoot
if ($LASTEXITCODE -ne 0) {
    throw "Phase 1B zero-shot matrix failed."
}

$scenarioReport = Join-Path $reportRootPath "phase1b_wide_ood_scenario_holdout_seed$DatasetSeed"
$scenarioRunRoot = Join-Path $runRootPath "scenario_holdout_seed$DatasetSeed"
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run-small-model-gain-iteration.ps1") `
    -PythonExe $pythonPath `
    -TrainDataset $wideTrain `
    -TestDataset $wideTest `
    -EnvProfile wide_ood `
    -Seeds $seedArg `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -Device $Device `
    -MatrixLabels $matrixLabels `
    -SummaryBaselines flat_v3_slot_focus `
    -ReportDir $scenarioReport `
    -RunRoot $scenarioRunRoot
if ($LASTEXITCODE -ne 0) {
    throw "Phase 1B scenario-heldout matrix failed."
}

$manifest = [ordered]@{
    completed_at = (Get-Date).ToString("o")
    dataset = $datasetPath
    dataset_seed = $DatasetSeed
    leakage_report = $leakageReport
    zero_shot_report = $zeroShotReport
    scenario_holdout_report = $scenarioReport
    matrix_labels = $matrixLabels.Split(",")
    seeds = $Seeds
    epochs = $Epochs
    batch_size = $BatchSize
    device = $Device
    phase2_pause_file = (Join-Path $repoRoot "artifacts\control\phase2_7b.paused")
}
$manifestPath = Join-Path $reportRootPath "phase1b_wide_ood_seed$DatasetSeed`_manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "Phase 1B manifest: $manifestPath"
Write-Host "Phase 1B leakage report: $leakageReport"
Write-Host "Phase 1B zero-shot report: $zeroShotReport"
Write-Host "Phase 1B scenario-heldout report: $scenarioReport"
