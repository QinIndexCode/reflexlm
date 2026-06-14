param(
    [ValidateSet("prepare", "canary", "matrix", "fastmatrix", "baselines", "evaluate", "fastevaluate", "gate", "fastgate", "all")]
    [string]$Stage = "prepare",
    [string]$PythonExe = ".venv312-gpu\Scripts\python.exe",
    [string]$ModelDir = "artifacts\models\Qwen2.5-7B-Instruct",
    [string]$TrainJsonl = "artifacts\datasets\phase1b_wide_ood_scenario_holdout_seed313_debug_v3\train.jsonl",
    [string]$ValJsonl = "artifacts\datasets\phase1b_wide_ood_scenario_holdout_seed313_debug_v3\val.jsonl",
    [string]$TestJsonl = "artifacts\datasets\phase1b_wide_ood_scenario_holdout_seed313_debug_v3\test.jsonl",
    [string]$NsiCheckpoint = "artifacts\runs_phase1b_wide_ood\scenario_holdout_seed313_pause_validation\training\20260513T111746Z-nsi-v20-debug-lexical-tiny-seed13-54519139\model.pt",
    [string]$SftRoot = "artifacts\datasets\phase2b_sft_unified_nsi_state_v2",
    [string]$FastSftRoot = "artifacts\datasets\phase2b_sft_unified_nsi_state_v2_fast",
    [string]$AdapterRoot = "artifacts\adapters\phase2b_unified_qwen7b",
    [string]$RunRoot = "artifacts\runs_phase2b_unified_qwen7b",
    [string]$ReportDir = "artifacts\reports\phase2b_unified_qwen7b",
    [string]$EnvProfile = "wide_ood",
    [string]$PromptOnlyEval = "",
    [string]$ReactEval = "",
    [string]$ReflexEval = "",
    [int]$FastTrainRows = 1536,
    [int]$FastValRows = 384,
    [double]$ParseRetryGrowth = 1.0,
    [switch]$AllowLongRun,
    [switch]$OverridePause
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

$pythonPath = Resolve-RepoPath $PythonExe
$modelPath = Resolve-RepoPath $ModelDir
$trainPath = Resolve-RepoPath $TrainJsonl
$valPath = Resolve-RepoPath $ValJsonl
$testPath = Resolve-RepoPath $TestJsonl
$nsiCheckpointPath = Resolve-RepoPath $NsiCheckpoint
$sftRootPath = Resolve-RepoPath $SftRoot $false
$fastSftRootPath = Resolve-RepoPath $FastSftRoot $false
$adapterRootPath = Resolve-RepoPath $AdapterRoot $false
$runRootPath = Resolve-RepoPath $RunRoot $false
$reportDirPath = Resolve-RepoPath $ReportDir $false
$promptOnlyEvalPath = if ($PromptOnlyEval) { Resolve-RepoPath $PromptOnlyEval $false } else { Join-Path $reportDirPath "baseline_qwen7b_prompt_only_eval.json" }
$reactEvalPath = if ($ReactEval) { Resolve-RepoPath $ReactEval $false } else { Join-Path $reportDirPath "baseline_qwen7b_react_eval.json" }
$reflexEvalPath = if ($ReflexEval) { Resolve-RepoPath $ReflexEval $false } else { Join-Path $reportDirPath "baseline_reflex_nsi_eval.json" }
$controlDir = Resolve-RepoPath "artifacts\control" $false
$pauseMarker = Join-Path $controlDir "phase2_7b.paused"
$jsonMotorDisabledMarker = Join-Path $controlDir "phase2b_json_motor.disabled"
$lockPath = Join-Path $controlDir "phase2b_unified_qwen7b.lock"

New-Item -ItemType Directory -Force -Path $sftRootPath, $fastSftRootPath, $adapterRootPath, $runRootPath, $reportDirPath, $controlDir | Out-Null
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }

function Assert-LongRunAllowed() {
    if (-not $AllowLongRun) {
        throw "Long 7B execution requires -AllowLongRun. Use Stage=prepare for non-training setup."
    }
    if ((Test-Path $jsonMotorDisabledMarker) -and -not $OverridePause) {
        throw "Phase 2B JSON motor validation is disabled by $jsonMotorDisabledMarker. Use the Phase 2C native head path instead."
    }
    if ((Test-Path $pauseMarker) -and -not $OverridePause) {
        throw "Phase 2 7B is paused by $pauseMarker. Pass -OverridePause only after explicit approval."
    }
}

function Enter-Lock() {
    if (Test-Path $lockPath) {
        throw "Phase 2B lock already exists at $lockPath. Refusing duplicate execution."
    }
    [ordered]@{
        pid = $PID
        stage = $Stage
        started_at = (Get-Date).ToString("o")
        script = $PSCommandPath
    } | ConvertTo-Json | Set-Content -Path $lockPath -Encoding UTF8
}

function Exit-Lock() {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}

function Build-SftDataset() {
    $manifestPath = Join-Path $sftRootPath "manifest.json"
    if (Test-Path $manifestPath) {
        Write-Host "Using existing nsi_state_v2 SFT manifest: $manifestPath"
        return
    }
    & $pythonPath -m reflexlm.cli.build_sft_dataset `
        --train-jsonl $trainPath `
        --val-jsonl $valPath `
        --output-dir $sftRootPath `
        --prompt-style nsi_state_v2 `
        --synapse-checkpoint $nsiCheckpointPath `
        --synapse-device cpu `
        --run-root $runRootPath `
        --output-json (Join-Path $reportDirPath "phase2b_sft_manifest.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 2B nsi_state_v2 SFT materialization failed."
    }
}

function Build-FastSftDataset() {
    Build-SftDataset
    $fastTrain = Join-Path $fastSftRootPath "train.jsonl"
    $fastVal = Join-Path $fastSftRootPath "val.jsonl"
    if ((Test-Path $fastTrain) -and (Test-Path $fastVal)) {
        Write-Host "Using existing fast nsi_state_v2 SFT subset: $fastSftRootPath"
        return
    }
    & $pythonPath -m reflexlm.cli.filter_sft_dataset `
        --train-jsonl (Join-Path $sftRootPath "nsi_state_v2\shared\train.jsonl") `
        --val-jsonl (Join-Path $sftRootPath "nsi_state_v2\shared\val.jsonl") `
        --output-dir $fastSftRootPath `
        --max-train-rows $FastTrainRows `
        --max-val-rows $FastValRows `
        --balance-key task_type `
        --balance-key route_name `
        --balance-key action_type `
        --seed 13 `
        --run-root $runRootPath `
        --run-name "phase2b_fast_sft_subset" `
        --output-json (Join-Path $reportDirPath "phase2b_fast_sft_subset.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 2B fast SFT subset materialization failed."
    }
}

function Run-Phase2BBaselines() {
    Assert-LongRunAllowed
    if (-not (Test-Path $reflexEvalPath)) {
        & $pythonPath -m reflexlm.cli.evaluate `
            --policy nsi_checkpoint `
            --dataset $testPath `
            --checkpoint $nsiCheckpointPath `
            --device cpu `
            --legal-action-mask `
            --env-profile $EnvProfile `
            --run-name "phase2b_reflex_nsi_baseline_eval" `
            --run-root $runRootPath `
            --output-json $reflexEvalPath
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 2B reflex baseline evaluation failed."
        }
    } else {
        Write-Host "Using existing reflex baseline eval: $reflexEvalPath"
    }

    if (-not (Test-Path $promptOnlyEvalPath)) {
        & $pythonPath -m reflexlm.cli.evaluate `
            --policy prompt_only `
            --dataset $testPath `
            --model-name $modelPath `
            --quantization 4bit `
            --cpu-offload `
            --max-new-tokens 96 `
            --max-time-s 20 `
            --max-retries 1 `
            --parse-retry-growth $ParseRetryGrowth `
            --env-profile $EnvProfile `
            --run-name "phase2b_qwen7b_prompt_only_baseline_eval" `
            --run-root $runRootPath `
            --output-json $promptOnlyEvalPath
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 2B prompt-only baseline evaluation failed."
        }
    } else {
        Write-Host "Using existing prompt-only baseline eval: $promptOnlyEvalPath"
    }

    if (-not (Test-Path $reactEvalPath)) {
        & $pythonPath -m reflexlm.cli.evaluate `
            --policy react `
            --dataset $testPath `
            --model-name $modelPath `
            --quantization 4bit `
            --cpu-offload `
            --max-new-tokens 96 `
            --max-time-s 20 `
            --max-retries 1 `
            --parse-retry-growth $ParseRetryGrowth `
            --env-profile $EnvProfile `
            --run-name "phase2b_qwen7b_react_baseline_eval" `
            --run-root $runRootPath `
            --output-json $reactEvalPath
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 2B ReAct baseline evaluation failed."
        }
    } else {
        Write-Host "Using existing ReAct baseline eval: $reactEvalPath"
    }
}

function Run-Phase2BGate($configs, [string]$auditTrainJsonl, [string]$auditValJsonl, [string]$auditLabel) {
    $generalizationAuditPath = Join-Path $reportDirPath ("phase2b_" + $auditLabel + "_generalization_audit.json")
    $overfitAuditPath = Join-Path $reportDirPath ("phase2b_" + $auditLabel + "_overfit_audit.json")
    if (-not (Test-Path $generalizationAuditPath)) {
        & $pythonPath -m reflexlm.cli.analyze_phase2b_generalization `
            --train-sft-jsonl $auditTrainJsonl `
            --val-sft-jsonl $auditValJsonl `
            --test-jsonl $testPath `
            --dataset-dir (Split-Path $testPath -Parent) `
            --synapse-checkpoint $nsiCheckpointPath `
            --synapse-device cpu `
            --output-json $generalizationAuditPath
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 2B generalization audit failed."
        }
    } else {
        Write-Host "Using existing Phase 2B generalization audit: $generalizationAuditPath"
    }
    if (-not (Test-Path $overfitAuditPath)) {
        $overfitArgs = @(
            "-m", "reflexlm.cli.analyze_phase2b_overfit",
            "--train-sft-jsonl", $auditTrainJsonl,
            "--val-sft-jsonl", $auditValJsonl,
            "--test-jsonl", $testPath,
            "--dataset-dir", (Split-Path $testPath -Parent),
            "--synapse-checkpoint", $nsiCheckpointPath,
            "--synapse-device", "cpu",
            "--output-json", $overfitAuditPath,
            "--no-fail"
        )
        foreach ($config in $configs) {
            $trainSummaryPath = Join-Path $reportDirPath ("train_" + $config.Name + ".json")
            if (Test-Path $trainSummaryPath) {
                $overfitArgs += @("--train-summary", $trainSummaryPath)
            }
        }
        & $pythonPath @overfitArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 2B overfit audit generation failed."
        }
    } else {
        Write-Host "Using existing Phase 2B overfit audit: $overfitAuditPath"
    }
    $evalPaths = @()
    foreach ($config in $configs) {
        $evalPath = Join-Path $reportDirPath ("eval_" + $config.Name + ".json")
        if (Test-Path $evalPath) {
            $evalPaths += $evalPath
        }
    }
    if ($evalPaths.Count -eq 0) {
        throw "No Phase 2B unified eval files found in $reportDirPath."
    }
    $gateArgs = @(
        "-m", "reflexlm.cli.check_phase2b_gates",
        "--prompt-only-eval", $promptOnlyEvalPath,
        "--react-eval", $reactEvalPath,
        "--reflex-eval", $reflexEvalPath,
        "--generalization-audit", $generalizationAuditPath,
        "--overfit-audit", $overfitAuditPath,
        "--output-json", (Join-Path $reportDirPath "phase2b_gate_report.json"),
        "--no-fail"
    )
    foreach ($evalPath in $evalPaths) {
        $gateArgs += @("--unified-eval", $evalPath)
    }
    & $pythonPath @gateArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 2B gate report generation failed."
    }
}

$matrix = @(
    [ordered]@{ Name = "r16_alpha32_lr1e-4_len512_ep2"; Rank = 16; Alpha = 32; LearningRate = "0.0001"; MaxLength = 512; Epochs = 2 },
    [ordered]@{ Name = "r32_alpha64_lr1e-4_len512_ep2"; Rank = 32; Alpha = 64; LearningRate = "0.0001"; MaxLength = 512; Epochs = 2 },
    [ordered]@{ Name = "r16_alpha32_lr7e-5_len768_ep2"; Rank = 16; Alpha = 32; LearningRate = "0.00007"; MaxLength = 768; Epochs = 2 }
)

$fastMatrix = @(
    [ordered]@{ Name = "fast_r8_alpha16_lr2e-4_len384_ep1"; Rank = 8; Alpha = 16; LearningRate = "0.0002"; MaxLength = 384; Epochs = 1; GradAccum = 8 },
    [ordered]@{ Name = "fast_r16_alpha32_lr2e-4_len384_ep1"; Rank = 16; Alpha = 32; LearningRate = "0.0002"; MaxLength = 384; Epochs = 1; GradAccum = 8 },
    [ordered]@{ Name = "fast_r16_alpha32_lr1e-4_len512_ep1"; Rank = 16; Alpha = 32; LearningRate = "0.0001"; MaxLength = 512; Epochs = 1; GradAccum = 8 }
)

function Run-TrainMatrix($configs, [string]$trainJsonl, [string]$valJsonl) {
    Assert-LongRunAllowed
    foreach ($config in $configs) {
        $adapterDir = Join-Path $adapterRootPath $config.Name
        $gradAccum = if ($config.Contains("GradAccum")) { $config.GradAccum } else { 16 }
        & $pythonPath -m reflexlm.cli.train_qwen_qlora `
            --base-model-name $modelPath `
            --train-jsonl $trainJsonl `
            --val-jsonl $valJsonl `
            --output-dir $adapterDir `
            --adapter-name ("phase2b_" + $config.Name) `
            --quantization 4bit `
            --epochs $config.Epochs `
            --micro-batch-size 1 `
            --gradient-accumulation-steps $gradAccum `
            --max-length $config.MaxLength `
            --learning-rate $config.LearningRate `
            --lora-rank $config.Rank `
            --lora-alpha $config.Alpha `
            --device cuda `
            --run-root $runRootPath `
            --output-json (Join-Path $reportDirPath ("train_" + $config.Name + ".json"))
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 2B matrix training failed for $($config.Name)."
        }
    }
}

function Run-EvalMatrix($configs) {
    Assert-LongRunAllowed
    foreach ($config in $configs) {
        $adapterDir = Join-Path $adapterRootPath $config.Name
        if (-not (Test-Path (Join-Path $adapterDir "adapter_model.safetensors"))) {
            Write-Warning "Skipping evaluation for missing adapter: $adapterDir"
            continue
        }
        & $pythonPath -m reflexlm.cli.evaluate `
            --policy hybrid_synaptic_qwen `
            --dataset $testPath `
            --model-name $modelPath `
            --adapter-path $adapterDir `
            --nsi-checkpoint $nsiCheckpointPath `
            --prompt-style nsi_state_v2 `
            --quantization 4bit `
            --cpu-offload `
            --device cuda `
            --nsi-device cpu `
            --max-new-tokens 96 `
            --max-time-s 20 `
            --parse-retry-growth $ParseRetryGrowth `
            --confidence-threshold 0.72 `
            --prediction-error-threshold 0.45 `
            --risk-threshold 0.70 `
            --env-profile $EnvProfile `
            --run-name ("phase2b_unified_eval_" + $config.Name) `
            --run-root $runRootPath `
            --output-json (Join-Path $reportDirPath ("eval_" + $config.Name + ".json"))
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 2B evaluation failed for $($config.Name)."
        }
    }
}

Enter-Lock
try {
    Build-SftDataset

    $sharedTrain = Join-Path $sftRootPath "nsi_state_v2\shared\train.jsonl"
    $sharedVal = Join-Path $sftRootPath "nsi_state_v2\shared\val.jsonl"

    if ($Stage -in @("canary", "all")) {
        Assert-LongRunAllowed
        & $pythonPath -m reflexlm.cli.qwen_tiny_overfit `
            --base-model-name $modelPath `
            --source-jsonl $sharedTrain `
            --output-root (Join-Path $adapterRootPath "canary") `
            --run-root $runRootPath `
            --output-json (Join-Path $reportDirPath "phase2b_unified_canary.json") `
            --max-examples 64 `
            --min-loss-drop 0.20 `
            --min-allowlist-valid-rate 0.98 `
            --quantization 4bit `
            --epochs 2 `
            --micro-batch-size 1 `
            --gradient-accumulation-steps 4 `
            --max-length 512 `
            --lora-rank 8 `
            --lora-alpha 16
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 2B canary failed."
        }
    }

    if ($Stage -in @("matrix", "all")) {
        Run-TrainMatrix $matrix $sharedTrain $sharedVal
    }

    if ($Stage -in @("fastmatrix")) {
        Build-FastSftDataset
        Run-TrainMatrix $fastMatrix (Join-Path $fastSftRootPath "train.jsonl") (Join-Path $fastSftRootPath "val.jsonl")
    }

    if ($Stage -in @("baselines", "all")) {
        Run-Phase2BBaselines
    }

    if ($Stage -in @("evaluate", "all")) {
        Run-EvalMatrix $matrix
    }

    if ($Stage -in @("fastevaluate")) {
        Run-EvalMatrix $fastMatrix
    }

    if ($Stage -in @("gate", "all")) {
        Run-Phase2BGate $matrix $sharedTrain $sharedVal "full"
    }

    if ($Stage -in @("fastgate")) {
        Build-FastSftDataset
        Run-Phase2BGate $fastMatrix (Join-Path $fastSftRootPath "train.jsonl") (Join-Path $fastSftRootPath "val.jsonl") "fast"
    }
} finally {
    Exit-Lock
}
