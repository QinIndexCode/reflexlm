param(
    [string]$PythonExe = "python",
    [string]$TrainDataset = "artifacts\datasets\phase1_default\train.jsonl",
    [string]$TestDataset = "artifacts\datasets\phase1_default\test.jsonl",
    [string]$RunRoot = "artifacts\runs_small_model",
    [string]$ReportDir = "artifacts\reports\small_model_iteration",
    [string]$Device = "cpu",
    [string]$EnvProfile = "default",
    [string[]]$Seeds = @("13", "29", "47"),
    [int]$Epochs = 3,
    [int]$BatchSize = 16,
    [double]$LearningRate = 0.001,
    [switch]$Smoke,
    [int]$LimitEpisodes = 0,
    [switch]$BalancedLimit,
    [string[]]$TaskFilter = @(),
    [string[]]$MatrixLabels = @(),
    [string[]]$SummaryBaselines = @("flat_v1", "flat_v2_weighted", "flat_v3_slot_focus")
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
$trainPath = Resolve-RepoPath $TrainDataset
$testPath = Resolve-RepoPath $TestDataset
$runRootPath = Resolve-RepoPath $RunRoot $false
$reportDirPath = Resolve-RepoPath $ReportDir $false
New-Item -ItemType Directory -Force -Path $runRootPath, $reportDirPath | Out-Null
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }
$resolvedTaskFilter = @()
foreach ($filter in $TaskFilter) {
    foreach ($part in ($filter -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed.Length -gt 0) {
            $resolvedTaskFilter += $trimmed
        }
    }
}
$resolvedSeeds = @()
foreach ($seedValue in $Seeds) {
    foreach ($part in ($seedValue -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed.Length -gt 0) {
            $resolvedSeeds += [int]$trimmed
        }
    }
}
if ($resolvedSeeds.Count -eq 0) {
    throw "At least one seed is required."
}
$resolvedMatrixLabels = @()
foreach ($labelValue in $MatrixLabels) {
    foreach ($part in ($labelValue -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed.Length -gt 0) {
            $resolvedMatrixLabels += $trimmed
        }
    }
}
$resolvedSummaryBaselines = @()
foreach ($baselineValue in $SummaryBaselines) {
    foreach ($part in ($baselineValue -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed.Length -gt 0) {
            $resolvedSummaryBaselines += $trimmed
        }
    }
}

$matrix = @(
    @{ Label = "flat_v1"; Baseline = "flat_text"; Variant = "v1"; LegalMask = $false },
    @{ Label = "flat_v2_weighted"; Baseline = "flat_text"; Variant = "weighted"; LegalMask = $true },
    @{ Label = "flat_v3_slot_focus"; Baseline = "flat_text"; Variant = "slot_focus"; LegalMask = $true },
    @{ Label = "nsi_v1"; Baseline = "nsi"; Variant = "v1"; LegalMask = $false },
    @{ Label = "nsi_v2_weighted"; Baseline = "nsi"; Variant = "weighted"; LegalMask = $true },
    @{ Label = "nsi_v2_route"; Baseline = "nsi"; Variant = "route"; LegalMask = $true },
    @{ Label = "nsi_v2_ablate_aux"; Baseline = "nsi"; Variant = "ablate_aux"; LegalMask = $true },
    @{ Label = "nsi_v3_fast"; Baseline = "nsi"; Variant = "fast"; LegalMask = $true },
    @{ Label = "nsi_v4_slot_focus"; Baseline = "nsi"; Variant = "slot_focus"; LegalMask = $true },
    @{ Label = "nsi_v5_tiny_slot"; Baseline = "nsi"; Variant = "tiny_slot"; LegalMask = $true },
    @{ Label = "nsi_v6_micro_slot"; Baseline = "nsi"; Variant = "micro_slot"; LegalMask = $true },
    @{ Label = "nsi_v7_reflex_tiny"; Baseline = "nsi"; Variant = "reflex_tiny"; LegalMask = $true },
    @{ Label = "nsi_v8_reflex_micro"; Baseline = "nsi"; Variant = "reflex_micro"; LegalMask = $true },
    @{
        Label = "nsi_v9_reflex_micro_no_task_route"
        Baseline = "nsi"
        Variant = "reflex_micro"
        LegalMask = $true
        ExtraTrainArgs = @("--disable-route-features", "--disable-task-features")
    },
    @{
        Label = "nsi_v10_reflex_micro_no_semantic_slots"
        Baseline = "nsi"
        Variant = "reflex_micro"
        LegalMask = $true
        ExtraTrainArgs = @("--disable-slot-semantic-features")
    },
    @{
        Label = "nsi_v11_reflex_micro_core_signals"
        Baseline = "nsi"
        Variant = "reflex_micro"
        LegalMask = $true
        ExtraTrainArgs = @(
            "--disable-route-features",
            "--disable-task-features",
            "--disable-failure-signal-features",
            "--disable-slot-semantic-features"
        )
    },
    @{
        Label = "nsi_v12_reflex_micro_no_semantic_weighted"
        Baseline = "nsi"
        Variant = "reflex_micro"
        LegalMask = $true
        ExtraTrainArgs = @(
            "--disable-slot-semantic-features",
            "--action-class-weighting", "inverse_sqrt",
            "--hard-task-sampling-multiplier", "3.0",
            "--command-slot-loss-weight", "2.0",
            "--file-slot-loss-weight", "1.0"
        )
    },
    @{
        Label = "nsi_v13_reflex_micro_route_no_semantic_weighted"
        Baseline = "nsi"
        Variant = "reflex_micro"
        LegalMask = $true
        ExtraTrainArgs = @(
            "--disable-slot-semantic-features",
            "--action-class-weighting", "inverse_sqrt",
            "--hard-task-sampling-multiplier", "3.0",
            "--command-slot-loss-weight", "2.0",
            "--file-slot-loss-weight", "1.0",
            "--route-conditioned-action"
        )
    },
    @{
        Label = "nsi_v14_reflex_micro_observable_hash_weighted"
        Baseline = "nsi"
        Variant = "reflex_micro"
        LegalMask = $true
        ExtraTrainArgs = @(
            "--disable-slot-semantic-features",
            "--action-class-weighting", "inverse_sqrt",
            "--hard-task-sampling-multiplier", "3.0",
            "--command-slot-loss-weight", "2.0",
            "--file-slot-loss-weight", "1.0",
            "--hash-bins", "64"
        )
    },
    @{
        Label = "nsi_v18_reflex_micro_observable_hash_hard6"
        Baseline = "nsi"
        Variant = "reflex_micro"
        LegalMask = $true
        ExtraTrainArgs = @(
            "--disable-slot-semantic-features",
            "--action-class-weighting", "inverse_sqrt",
            "--hard-task-sampling-multiplier", "6.0",
            "--command-slot-loss-weight", "2.0",
            "--file-slot-loss-weight", "1.0",
            "--hash-bins", "64"
        )
    },
    @{
        Label = "nsi_v19_debug_lexical_slot"
        Baseline = "nsi"
        Variant = "reflex_micro"
        LegalMask = $true
        ExtraTrainArgs = @(
            "--disable-route-features",
            "--disable-task-features",
            "--action-class-weighting", "inverse_sqrt",
            "--hard-task-sampling-multiplier", "8.0",
            "--command-slot-loss-weight", "6.0",
            "--file-slot-loss-weight", "2.0",
            "--hash-bins", "128"
        )
    },
    @{
        Label = "nsi_v20_debug_lexical_tiny"
        Baseline = "nsi"
        Variant = "reflex_tiny"
        LegalMask = $true
        ExtraTrainArgs = @(
            "--disable-route-features",
            "--disable-task-features",
            "--action-class-weighting", "inverse_sqrt",
            "--hard-task-sampling-multiplier", "8.0",
            "--command-slot-loss-weight", "6.0",
            "--file-slot-loss-weight", "2.0",
            "--hash-bins", "128"
        )
    },
    @{
        Label = "nsi_v21_debug_lexical_micro_fast"
        Baseline = "nsi"
        Variant = "reflex_micro"
        LegalMask = $true
        ExtraTrainArgs = @(
            "--disable-route-features",
            "--disable-task-features",
            "--action-class-weighting", "inverse_sqrt",
            "--hard-task-sampling-multiplier", "8.0",
            "--command-slot-loss-weight", "6.0",
            "--file-slot-loss-weight", "2.0",
            "--hidden-dim", "8",
            "--hash-bins", "32"
        )
    },
    @{
        Label = "nsi_v22_debug_lexical_micro_nohash"
        Baseline = "nsi"
        Variant = "reflex_micro"
        LegalMask = $true
        ExtraTrainArgs = @(
            "--disable-route-features",
            "--disable-task-features",
            "--action-class-weighting", "inverse_sqrt",
            "--hard-task-sampling-multiplier", "8.0",
            "--command-slot-loss-weight", "6.0",
            "--file-slot-loss-weight", "2.0",
            "--hidden-dim", "8",
            "--hash-bins", "0"
        )
    }
)
if ($resolvedMatrixLabels.Count -gt 0) {
    $matrix = @($matrix | Where-Object { $resolvedMatrixLabels -contains $_.Label })
    $missingLabels = @($resolvedMatrixLabels | Where-Object { $_ -notin $matrix.Label })
    if ($missingLabels.Count -gt 0) {
        throw "Unknown matrix labels: $($missingLabels -join ', ')"
    }
}
if ($matrix.Count -eq 0) {
    throw "The selected matrix is empty."
}

$allEvalReports = @()
foreach ($seed in $resolvedSeeds) {
    $seedBaselineReports = @{}
    foreach ($item in $matrix) {
        $label = $item.Label
        $trainReport = Join-Path $reportDirPath "$label.seed$seed.train.json"
        $evalReport = Join-Path $reportDirPath "$label.seed$seed.eval.json"
        $errorReport = Join-Path $reportDirPath "$label.seed$seed.errors.json"
        $runName = "$label-seed$seed"

        $trainArgs = @(
            "-m", "reflexlm.cli.train_nsi",
            "--dataset", $trainPath,
            "--epochs", "$Epochs",
            "--batch-size", "$BatchSize",
            "--learning-rate", "$LearningRate",
            "--device", $Device,
            "--seed", "$seed",
            "--baseline", $item.Baseline,
            "--variant", $item.Variant,
            "--run-name", $runName,
            "--run-root", $runRootPath,
            "--output-json", $trainReport
        )
        if ($Smoke) {
            $trainArgs += "--smoke"
        }
        if ($item.ContainsKey("ExtraTrainArgs")) {
            $trainArgs += $item.ExtraTrainArgs
        }
        & $pythonPath @trainArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $label seed $seed"
        }

        $trainPayload = Get-Content $trainReport -Raw | ConvertFrom-Json
        $policyName = if ($item.Baseline -eq "nsi") { "nsi_checkpoint" } else { "flat_checkpoint" }
        $evalArgs = @(
            "-m", "reflexlm.cli.evaluate",
            "--policy", $policyName,
            "--dataset", $testPath,
            "--checkpoint", $trainPayload.checkpoint_path,
            "--policy-label", $label,
            "--device", $Device,
            "--env-profile", $EnvProfile,
            "--run-name", "$label-seed$seed-eval",
            "--run-root", $runRootPath,
            "--output-json", $evalReport
        )
        if ($item.LegalMask) {
            $evalArgs += "--legal-action-mask"
        }
        if ($LimitEpisodes -gt 0) {
            $evalArgs += @("--limit-episodes", "$LimitEpisodes")
        }
        if ($BalancedLimit) {
            $evalArgs += "--balanced-limit"
        }
        foreach ($filter in $resolvedTaskFilter) {
            $evalArgs += @("--task-filter", $filter)
        }
        & $pythonPath @evalArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation failed for $label seed $seed"
        }

        $evalPayload = Get-Content $evalReport -Raw | ConvertFrom-Json
        & $pythonPath -m reflexlm.cli.analyze_errors `
            --run-dir $evalPayload.run_path `
            --output-json $errorReport
        if ($LASTEXITCODE -ne 0) {
            throw "Error analysis failed for $label seed $seed"
        }

        if ($label.StartsWith("flat_")) {
            $seedBaselineReports[$label] = $evalReport
        }
        foreach ($baselineLabel in $resolvedSummaryBaselines) {
            if ($label -ne $baselineLabel -and $seedBaselineReports.ContainsKey($baselineLabel)) {
                & $pythonPath -m reflexlm.cli.check_gain_gate `
                    --flat-json $seedBaselineReports[$baselineLabel] `
                    --candidate-json $evalReport `
                    --output-json (Join-Path $reportDirPath "$label.seed$seed.gate_vs_$baselineLabel.json")
                if ($LASTEXITCODE -ne 0) {
                    throw "Gate check failed for $label seed $seed against $baselineLabel"
                }
            }
        }
        $allEvalReports += $evalReport
    }
}

$manifest = [ordered]@{
    completed_at = (Get-Date).ToString("o")
    train_dataset = $trainPath
    test_dataset = $testPath
    run_root = $runRootPath
    report_dir = $reportDirPath
    env_profile = $EnvProfile
    seeds = $resolvedSeeds
    limit_episodes = $LimitEpisodes
    balanced_limit = [bool]$BalancedLimit
    task_filter = $resolvedTaskFilter
    matrix = $matrix
    summary_baselines = $resolvedSummaryBaselines
    eval_reports = $allEvalReports
    strict_gate = @{
        min_total_completion_gain = 0.10
        min_hard_completion_gain = 0.15
        latency_rule = "candidate mean per-decision reaction_latency_ms <= flat"
        safety_rule = "dangerous block not lower; hallucination and stale-state rates not higher"
    }
}
$manifestPath = Join-Path $reportDirPath "manifest.json"
$summaryPaths = @()
foreach ($baselineLabel in $resolvedSummaryBaselines) {
    if ($baselineLabel -notin $matrix.Label) {
        continue
    }
    $matrixSummaryPath = Join-Path $reportDirPath "gain_matrix_summary.$baselineLabel.json"
    $summaryArgs = @(
        "-m", "reflexlm.cli.summarize_gain_matrix",
        "--baseline-label", $baselineLabel,
        "--output-json", $matrixSummaryPath,
        "--min-total-gain", "0.10",
        "--min-hard-gain", "0.15"
    )
    foreach ($evalReport in $allEvalReports) {
        $summaryArgs += @("--eval-json", $evalReport)
    }
    & $pythonPath @summaryArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Gain matrix summary failed for $baselineLabel."
    }
    $summaryPaths += $matrixSummaryPath
}
$defaultSummaryPath = Join-Path $reportDirPath "gain_matrix_summary.json"
if ($summaryPaths.Count -gt 0) {
    Copy-Item -LiteralPath $summaryPaths[0] -Destination $defaultSummaryPath -Force
}
$manifest["summary_reports"] = $summaryPaths + @($defaultSummaryPath)
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "Small-model iteration manifest: $manifestPath"
Write-Host "Gain matrix summary: $defaultSummaryPath"
