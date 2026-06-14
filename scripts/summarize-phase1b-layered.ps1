param(
    [string]$PythonExe = "python",
    [string]$FixedReportDir = "artifacts\reports\phase1b_wide_ood_scenario_holdout_seed313",
    [string]$ExtraReportDir = "artifacts\reports\phase1b_wide_ood_scenario_holdout_seed313_extra",
    [string]$BaselineLabel = "flat_v3_slot_focus"
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
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }

function Invoke-ScopeSummary([string]$ReportDir, [string]$Scope, [string]$HardTaskSet) {
    $reportPath = Resolve-RepoPath $ReportDir
    $evalArgs = Get-ChildItem $reportPath -Filter "*.eval.json" | ForEach-Object {
        @("--eval-json", $_.FullName)
    }
    if ($evalArgs.Count -eq 0) {
        throw "No eval JSON files found in $reportPath"
    }
    $outputPath = Join-Path $reportPath "gain_matrix_summary.$Scope.$BaselineLabel.json"
    & $pythonPath -m reflexlm.cli.summarize_gain_matrix @evalArgs `
        --baseline-label $BaselineLabel `
        --task-scope $Scope `
        --hard-task-set $HardTaskSet `
        --output-json $outputPath
    if ($LASTEXITCODE -ne 0) {
        throw "Layered summary failed for $ReportDir scope $Scope."
    }
    Write-Host "Layered summary: $outputPath"
}

Invoke-ScopeSummary $FixedReportDir "reflex_layer" "reflex_layer"
Invoke-ScopeSummary $FixedReportDir "debug_cortex" "phase1_all"
Invoke-ScopeSummary $ExtraReportDir "reflex_layer" "reflex_layer"
Invoke-ScopeSummary $ExtraReportDir "debug_cortex" "phase1_all"
