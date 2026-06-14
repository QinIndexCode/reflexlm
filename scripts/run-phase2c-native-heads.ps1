param(
    [ValidateSet("prepare", "canary", "train", "evaluate", "gate", "all")]
    [string]$Stage = "prepare",
    [string]$PythonExe = ".venv312-qwen7b-stable\Scripts\python.exe",
    [string]$ModelDir = "artifacts\models\Qwen2.5-7B-Instruct",
    [string]$SourceTrainJsonl = "artifacts\datasets\phase1b_wide_ood_scenario_holdout_seed313_debug_v3\train.jsonl",
    [string]$SourceValJsonl = "artifacts\datasets\phase1b_wide_ood_scenario_holdout_seed313_debug_v3\val.jsonl",
    [string]$SourceTestJsonl = "artifacts\datasets\phase1b_wide_ood_scenario_holdout_seed313_debug_v3\test.jsonl",
    [string]$HeadDatasetRoot = "artifacts\datasets\phase2c_native_head_wide_ood_scenario_holdout_seed313",
    [string]$NsiCheckpoint = "artifacts\runs_phase1b_wide_ood\scenario_holdout_seed313_pause_validation\training\20260513T111746Z-nsi-v20-debug-lexical-tiny-seed13-54519139\model.pt",
    [string]$AdapterRoot = "artifacts\adapters\phase2c_native_heads",
    [string]$RunRoot = "artifacts\runs_phase2c_native_nervous",
    [string]$ReportDir = "artifacts\reports\phase2c_native_nervous",
    [string]$EnvProfile = "wide_ood",
    [string]$AdapterName = "phase2c_native_heads_r16_alpha32_lr1e-4_len512_ep1",
    [double]$LearningRate = 0.0001,
    [int]$Epochs = 1,
    [int]$MaxLength = 512,
    [int]$LoraRank = 16,
    [int]$LoraAlpha = 32,
    [int]$GradientAccumulationSteps = 8,
    [int]$ProgressLogIntervalSteps = 50,
    [double]$CommandIntentLossWeight = 0.5,
    [double]$CommandSlotLossWeight = 0.3,
    [int]$DebugCommandOversample = 1,
    [switch]$BalanceDebugCommandIntents,
    [switch]$UsePairwiseCommandReranker,
    [ValidateSet("replace", "residual")]
    [string]$PairwiseCommandFusion = "residual",
    [ValidateSet("all", "ambiguous_intent")]
    [string]$PairwiseCommandPolicy = "all",
    [int]$PairwiseCommandMaxLength = 0,
    [int]$PairwiseCommandTopK = 0,
    [ValidateSet("backbone", "features_only")]
    [string]$CommandCandidateEncoder = "backbone",
    [int]$CanaryTrainRecords = 64,
    [int]$CanaryValRecords = 64,
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
$sourceTrainPath = Resolve-RepoPath $SourceTrainJsonl
$sourceValPath = Resolve-RepoPath $SourceValJsonl
$sourceTestPath = Resolve-RepoPath $SourceTestJsonl
$headDatasetRootPath = Resolve-RepoPath $HeadDatasetRoot $false
$headTrainPath = Join-Path $headDatasetRootPath "train.jsonl"
$headValPath = Join-Path $headDatasetRootPath "val.jsonl"
$headTestPath = Join-Path $headDatasetRootPath "test.jsonl"
$nsiCheckpointPath = Resolve-RepoPath $NsiCheckpoint
$adapterRootPath = Resolve-RepoPath $AdapterRoot $false
$adapterPath = Join-Path $adapterRootPath $AdapterName
$runRootPath = Resolve-RepoPath $RunRoot $false
$reportDirPath = Resolve-RepoPath $ReportDir $false
$controlDir = Resolve-RepoPath "artifacts\control" $false
$pauseMarker = Join-Path $controlDir "phase2c_native_heads.paused"
$lockPath = Join-Path $controlDir "phase2c_native_heads.lock"
$activeConfigPath = Join-Path $controlDir "phase2c_native_heads.active.json"
$datasetManifestPath = Join-Path $reportDirPath "phase2c_head_dataset_manifest.json"
$trainSummaryPath = Join-Path $reportDirPath ($AdapterName + ".training_summary.json")
$evalPath = Join-Path $reportDirPath ($AdapterName + ".eval.json")
$gatePath = Join-Path $reportDirPath ($AdapterName + ".gate.json")
$referenceEvalPath = Join-Path $reportDirPath "nsi_v20_reflex_reference_eval.json"

New-Item -ItemType Directory -Force -Path $headDatasetRootPath, $adapterRootPath, $runRootPath, $reportDirPath, $controlDir | Out-Null
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }

function Assert-LongRunAllowed() {
    if (-not $AllowLongRun) {
        throw "Phase 2C 7B execution requires -AllowLongRun. Use Stage=prepare for non-training setup."
    }
    if ((Test-Path $pauseMarker) -and -not $OverridePause) {
        throw "Phase 2C native-head training is paused by $pauseMarker. Pass -OverridePause only after explicit approval."
    }
}

function Get-RunTargets([bool]$Canary) {
    $targetAdapterName = if ($Canary) { $AdapterName + "_canary" } else { $AdapterName }
    [ordered]@{
        adapter_name = $targetAdapterName
        adapter_path = Join-Path $adapterRootPath $targetAdapterName
        train_summary_path = Join-Path $reportDirPath ($targetAdapterName + ".training_summary.json")
        eval_path = Join-Path $reportDirPath ($targetAdapterName + ".eval.json")
        gate_path = Join-Path $reportDirPath ($targetAdapterName + ".gate.json")
    }
}

function Enter-Lock() {
    if (Test-Path $lockPath) {
        throw "Phase 2C lock already exists at $lockPath. Refusing duplicate execution."
    }
    $activeTargets = Get-RunTargets ($Stage -eq "canary")
    [ordered]@{
        updated_at = (Get-Date).ToString("o")
        stage = $Stage
        requested_adapter_name = $AdapterName
        adapter_name = $activeTargets.adapter_name
        adapter_path = $activeTargets.adapter_path
        max_length = $MaxLength
        gradient_accumulation_steps = $GradientAccumulationSteps
        progress_log_interval_steps = $ProgressLogIntervalSteps
        learning_rate = $LearningRate
        epochs = $Epochs
        lora_rank = $LoraRank
        lora_alpha = $LoraAlpha
        command_intent_loss_weight = $CommandIntentLossWeight
        command_slot_loss_weight = $CommandSlotLossWeight
        debug_command_oversample = $DebugCommandOversample
        balance_debug_command_intents = [bool]$BalanceDebugCommandIntents
        use_pairwise_command_reranker = [bool]$UsePairwiseCommandReranker
        pairwise_command_fusion = $PairwiseCommandFusion
        pairwise_command_policy = $PairwiseCommandPolicy
        pairwise_command_max_length = $PairwiseCommandMaxLength
        pairwise_command_top_k = $PairwiseCommandTopK
        command_candidate_encoder = $CommandCandidateEncoder
        train_summary_path = $activeTargets.train_summary_path
        eval_path = $activeTargets.eval_path
        gate_path = $activeTargets.gate_path
        run_root = $runRootPath
        report_dir = $reportDirPath
        command_template = ".\scripts\run-phase2c-native-heads.ps1 -AdapterName `"$($activeTargets.adapter_name)`" -MaxLength $MaxLength -GradientAccumulationSteps $GradientAccumulationSteps -CommandIntentLossWeight $CommandIntentLossWeight -CommandSlotLossWeight $CommandSlotLossWeight -DebugCommandOversample $DebugCommandOversample -PairwiseCommandFusion $PairwiseCommandFusion -Stage <stage> -AllowLongRun"
    } | ConvertTo-Json | Set-Content -Path $activeConfigPath -Encoding UTF8
    [ordered]@{
        pid = $PID
        stage = $Stage
        adapter_name = $AdapterName
        started_at = (Get-Date).ToString("o")
        script = $PSCommandPath
    } | ConvertTo-Json | Set-Content -Path $lockPath -Encoding UTF8
}

function Exit-Lock() {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}

function Build-HeadDataset() {
    $manifestPath = Join-Path $headDatasetRootPath "manifest.json"
    if ((Test-Path $manifestPath) -and (Test-Path $datasetManifestPath)) {
        Write-Host "Using existing Phase 2C head dataset: $manifestPath"
        return
    }
    & $pythonPath -m reflexlm.cli.build_phase2c_head_dataset `
        --train-jsonl $sourceTrainPath `
        --val-jsonl $sourceValPath `
        --test-jsonl $sourceTestPath `
        --output-dir $headDatasetRootPath `
        --synapse-checkpoint $nsiCheckpointPath `
        --synapse-device cpu `
        --run-root $runRootPath `
        --output-json $datasetManifestPath
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 2C head dataset materialization failed."
    }
}

function Run-NativeHeadTraining([bool]$Canary) {
    Assert-LongRunAllowed
    $targets = Get-RunTargets $Canary
    $args = @(
        "-m", "reflexlm.cli.train_phase2c_native_heads",
        "--base-model-name", $modelPath,
        "--train-jsonl", $headTrainPath,
        "--val-jsonl", $headValPath,
        "--output-dir", $targets.adapter_path,
        "--adapter-name", $targets.adapter_name,
        "--quantization", "4bit",
        "--learning-rate", $LearningRate,
        "--epochs", $Epochs,
        "--micro-batch-size", "1",
        "--gradient-accumulation-steps", $GradientAccumulationSteps,
        "--max-length", $MaxLength,
        "--progress-log-interval-steps", $ProgressLogIntervalSteps,
        "--lora-rank", $LoraRank,
        "--lora-alpha", $LoraAlpha,
        "--command-intent-loss-weight", $CommandIntentLossWeight,
        "--command-slot-loss-weight", $CommandSlotLossWeight,
        "--debug-command-oversample", $DebugCommandOversample,
        "--pairwise-command-fusion", $PairwiseCommandFusion,
        "--pairwise-command-policy", $PairwiseCommandPolicy,
        "--command-candidate-encoder", $CommandCandidateEncoder,
        "--device", "cuda",
        "--run-root", $runRootPath,
        "--output-json", $targets.train_summary_path
    )
    if ($Canary) {
        $args += @("--max-train-records", $CanaryTrainRecords, "--max-val-records", $CanaryValRecords)
    }
    if ($BalanceDebugCommandIntents) {
        $args += @("--balance-debug-command-intents")
    }
    if ($UsePairwiseCommandReranker) {
        $args += @("--use-pairwise-command-reranker")
    }
    if ($PairwiseCommandMaxLength -gt 0) {
        $args += @("--pairwise-command-max-length", $PairwiseCommandMaxLength)
    }
    if ($PairwiseCommandTopK -gt 0) {
        $args += @("--pairwise-command-top-k", $PairwiseCommandTopK)
    }
    & $pythonPath @args
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 2C native-head training failed."
    }
}

function Run-NativeHeadEvaluation() {
    Assert-LongRunAllowed
    if (-not (Test-Path $adapterPath)) {
        throw "Missing Phase 2C adapter path: $adapterPath"
    }
    & $pythonPath -m reflexlm.cli.evaluate `
        --policy qwen_native_heads `
        --dataset $sourceTestPath `
        --model-name $modelPath `
        --native-head-path $adapterPath `
        --nsi-checkpoint $nsiCheckpointPath `
        --quantization 4bit `
        --device cuda `
        --nsi-device cpu `
        --native-head-max-length $MaxLength `
        --env-profile $EnvProfile `
        --run-name ($AdapterName + "_eval") `
        --run-root $runRootPath `
        --output-json $evalPath
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 2C native-head evaluation failed."
    }
}

function Run-Gate() {
    if (-not (Test-Path $evalPath)) {
        throw "Missing Phase 2C eval report: $evalPath"
    }
    $gateArgs = @(
        "-m", "reflexlm.cli.check_phase2c_gates",
        "--eval-json", $evalPath,
        "--dataset-manifest-json", $datasetManifestPath,
        "--reference-eval-json", $referenceEvalPath,
        "--output-json", $gatePath,
        "--no-fail"
    )
    if (Test-Path $trainSummaryPath) {
        $gateArgs += @("--train-summary-json", $trainSummaryPath)
    }
    & $pythonPath @gateArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 2C gate check command failed."
    }
}

Enter-Lock
try {
    switch ($Stage) {
        "prepare" { Build-HeadDataset }
        "canary" { Build-HeadDataset; Run-NativeHeadTraining $true }
        "train" { Build-HeadDataset; Run-NativeHeadTraining $false }
        "evaluate" { Run-NativeHeadEvaluation }
        "gate" { Run-Gate }
        "all" { Build-HeadDataset; Run-NativeHeadTraining $true; Run-NativeHeadTraining $false; Run-NativeHeadEvaluation; Run-Gate }
    }
} finally {
    Exit-Lock
}
