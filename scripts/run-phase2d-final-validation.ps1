param(
    [ValidateSet("freeze", "prepare", "train", "package", "evaluate-fixed", "evaluate-debug", "evaluate-quasi", "evaluate-latent", "evaluate-prompt-quasi", "evaluate-react-quasi", "evaluate-baselines", "evaluate-ablation", "evaluate-latent-ablation", "evaluate-native-head-only", "evaluate-continuation-only", "baseline-table", "archive", "gate", "all")]
    [string]$Stage = "prepare",
    [string]$ControlName = "phase2d_final_validation",
    [string]$PythonExe = ".venv312-qwen7b-stable\Scripts\python.exe",
    [string]$ModelDir = "artifacts\models\Qwen2.5-7B-Instruct",
    [string]$ConfigYaml = "configs\phase2d_final_validation.yaml",
    [string]$SourceTrainJsonl = "artifacts\datasets\phase1b_wide_ood_scenario_holdout_seed313_debug_v3\train.jsonl",
    [string]$SourceValJsonl = "artifacts\datasets\phase1b_wide_ood_scenario_holdout_seed313_debug_v3\val.jsonl",
    [string]$SourceTestJsonl = "artifacts\datasets\phase1b_wide_ood_scenario_holdout_seed313_debug_v3\test.jsonl",
    [string]$NsiCheckpoint = "artifacts\runs_phase1b_wide_ood\scenario_holdout_seed313_pause_validation\training\20260513T111746Z-nsi-v20-debug-lexical-tiny-seed13-54519139\model.pt",
    [string]$HeadDatasetRoot = "artifacts\datasets\phase2d_native_head_final",
    [string]$DebugOodRoot = "artifacts\datasets\phase2d_debug_ood_v2",
    [string]$QuasiRealRoot = "artifacts\datasets\phase2d_quasi_real_terminal_v1",
    [string]$LatentSensitiveRoot = "artifacts\datasets\phase2f_latent_sensitive",
    [string]$AdapterRoot = "artifacts\adapters\phase2d_native_heads",
    [string]$PackageRoot = "artifacts\packages\phase2d_native_nervous",
    [string]$RunRoot = "artifacts\runs_phase2d_final_validation",
    [string]$ReportDir = "artifacts\reports\phase2d_final_validation",
    [string]$Phase2IDataAuditJson = "",
    [string]$AdapterName = "phase2d_native_heads_balanced_r16_alpha32_lr1e-4_len256_ep1",
    [double]$LearningRate = 0.0001,
    [int]$Epochs = 1,
    [int]$MaxLength = 256,
    [int]$LoraRank = 16,
    [int]$LoraAlpha = 32,
    [int]$GradientAccumulationSteps = 4,
    [double]$CommandIntentLossWeight = 1.0,
    [double]$CommandSlotLossWeight = 1.5,
    [int]$MaxTrainRecords = 0,
    [int]$MaxValRecords = 0,
    [int]$ProgressLogIntervalSteps = 50,
    [int]$DebugEpisodesPerScenario = 8,
    [int]$QuasiEpisodesPerScenario = 8,
    [int]$LatentEpisodesPerScenario = 8,
    [string]$ExtraTrainProfile = "",
    [string]$ExtraValProfile = "",
    [string]$ExtraTrainRoot = "artifacts\datasets\phase2e_debug_transition_train_v1",
    [string]$ExtraValRoot = "artifacts\datasets\phase2e_debug_transition_val_v1",
    [int]$ExtraTrainEpisodesPerScenario = 24,
    [int]$ExtraValEpisodesPerScenario = 8,
    [switch]$UsePairwiseCommandReranker,
    [ValidateSet("replace", "residual")]
    [string]$PairwiseCommandFusion = "residual",
    [ValidateSet("all", "ambiguous_intent")]
    [string]$PairwiseCommandPolicy = "all",
    [int]$PairwiseCommandMaxLength = 0,
    [int]$PairwiseCommandTopK = 0,
    [ValidateSet("backbone", "features_only")]
    [string]$CommandCandidateEncoder = "backbone",
    [ValidateSet("concat", "additive")]
    [string]$LatentFusion = "additive",
    [switch]$AllowLongRun,
    [switch]$SkipPhase2IPrePackageGate,
    [switch]$OverridePause
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent

function Resolve-RepoPath([string]$PathValue, [bool]$MustExist = $true) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        if ($MustExist) { return (Resolve-Path $PathValue -ErrorAction Stop).Path }
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    $fullPath = Join-Path $repoRoot $PathValue
    if ($MustExist) { return (Resolve-Path $fullPath -ErrorAction Stop).Path }
    return [System.IO.Path]::GetFullPath($fullPath)
}

$pythonPath = Resolve-RepoPath $PythonExe
$modelPath = Resolve-RepoPath $ModelDir
$configPath = Resolve-RepoPath $ConfigYaml
$sourceTrainPath = Resolve-RepoPath $SourceTrainJsonl
$sourceValPath = Resolve-RepoPath $SourceValJsonl
$sourceTestPath = Resolve-RepoPath $SourceTestJsonl
$nsiCheckpointPath = Resolve-RepoPath $NsiCheckpoint
$headDatasetRootPath = Resolve-RepoPath $HeadDatasetRoot $false
$debugRootPath = Resolve-RepoPath $DebugOodRoot $false
$quasiRootPath = Resolve-RepoPath $QuasiRealRoot $false
$latentRootPath = Resolve-RepoPath $LatentSensitiveRoot $false
$extraTrainRootPath = Resolve-RepoPath $ExtraTrainRoot $false
$extraValRootPath = Resolve-RepoPath $ExtraValRoot $false
$adapterRootPath = Resolve-RepoPath $AdapterRoot $false
$adapterPath = Join-Path $adapterRootPath $AdapterName
$packageRootPath = Resolve-RepoPath $PackageRoot $false
$packagePath = Join-Path $packageRootPath $AdapterName
$ablationPackagePath = Join-Path $packageRootPath ($AdapterName + "_no_nsi_latent")
$nativeHeadOnlyPackagePath = Join-Path $packageRootPath ($AdapterName + "_native_head_only")
$continuationOnlyPackagePath = Join-Path $packageRootPath ($AdapterName + "_continuation_only")
$runRootPath = Resolve-RepoPath $RunRoot $false
$reportDirPath = Resolve-RepoPath $ReportDir $false
$controlDir = Resolve-RepoPath "artifacts\control" $false
$pauseMarker = Join-Path $controlDir ($ControlName + ".paused")
$lockPath = Join-Path $controlDir ($ControlName + ".lock")
$activePath = Join-Path $controlDir ($ControlName + ".active.json")
$frozenConfigPath = Join-Path $reportDirPath "phase2d_final_config.freeze.json"
$headManifestPath = Join-Path $reportDirPath "phase2d_head_dataset_manifest.json"
$debugManifestPath = Join-Path $debugRootPath "manifest.json"
$quasiManifestPath = Join-Path $quasiRootPath "manifest.json"
$latentManifestPath = Join-Path $latentRootPath "manifest.json"
$debugAuditPath = Join-Path $reportDirPath "phase2d_debug_ood_v2_dataset_audit.json"
$quasiAuditPath = Join-Path $reportDirPath "phase2d_quasi_real_dataset_audit.json"
$latentAuditPath = Join-Path $reportDirPath "phase2f_latent_sensitive_dataset_audit.json"
$trainSummaryPath = Join-Path $reportDirPath ($AdapterName + ".training_summary.json")
$fixedEvalPath = Join-Path $reportDirPath ($AdapterName + ".fixed_eval.json")
$debugEvalPath = Join-Path $reportDirPath ($AdapterName + ".debug_ood_v2_eval.json")
$quasiEvalPath = Join-Path $reportDirPath ($AdapterName + ".quasi_real_eval.json")
$latentEvalPath = Join-Path $reportDirPath ($AdapterName + ".latent_sensitive_eval.json")
$promptQuasiEvalPath = Join-Path $reportDirPath "prompt_only_7b.quasi_real_eval.json"
$reactQuasiEvalPath = Join-Path $reportDirPath "react_7b.quasi_real_eval.json"
$ablationEvalPath = Join-Path $reportDirPath ($AdapterName + ".no_nsi_latent_eval.json")
$latentAblationEvalPath = Join-Path $reportDirPath ($AdapterName + ".latent_sensitive_no_nsi_latent_eval.json")
$nativeHeadOnlyEvalPath = Join-Path $reportDirPath ($AdapterName + ".native_head_only_debug_ood_v2_eval.json")
$continuationOnlyEvalPath = Join-Path $reportDirPath ($AdapterName + ".continuation_only_debug_ood_v2_eval.json")
$baselineTablePath = Join-Path $reportDirPath "phase2f_exact_baseline_table.json"
$gatePath = Join-Path $reportDirPath "phase2d_final_gate.json"
$phase2iPrePackageGatePath = Join-Path $reportDirPath "phase2i_prepackage_gate.json"

New-Item -ItemType Directory -Force -Path $headDatasetRootPath, $debugRootPath, $quasiRootPath, $latentRootPath, $extraTrainRootPath, $extraValRootPath, $adapterRootPath, $packageRootPath, $runRootPath, $reportDirPath, $controlDir | Out-Null
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }

function Assert-LongRunAllowed() {
    if (-not $AllowLongRun) {
        throw "Phase2D long-running stage requires -AllowLongRun."
    }
    if ((Test-Path $pauseMarker) -and -not $OverridePause) {
        throw "$ControlName is paused by $pauseMarker."
    }
}

function Enter-Lock() {
    if (Test-Path $lockPath) {
        throw "$ControlName lock already exists at $lockPath. Refusing duplicate execution."
    }
    [ordered]@{
        updated_at = (Get-Date).ToString("o")
        control_name = $ControlName
        stage = $Stage
        adapter_name = $AdapterName
        adapter_path = $adapterPath
        package_path = $packagePath
        ablation_package_path = $ablationPackagePath
        max_length = $MaxLength
        gradient_accumulation_steps = $GradientAccumulationSteps
        max_train_records = $MaxTrainRecords
        max_val_records = $MaxValRecords
        progress_log_interval_steps = $ProgressLogIntervalSteps
        extra_train_profile = $ExtraTrainProfile
        extra_val_profile = $ExtraValProfile
        use_pairwise_command_reranker = [bool]$UsePairwiseCommandReranker
        pairwise_command_fusion = $PairwiseCommandFusion
        pairwise_command_policy = $PairwiseCommandPolicy
        pairwise_command_max_length = $PairwiseCommandMaxLength
        pairwise_command_top_k = $PairwiseCommandTopK
        command_candidate_encoder = $CommandCandidateEncoder
        latent_fusion = $LatentFusion
        extra_train_root = $extraTrainRootPath
        extra_val_root = $extraValRootPath
        frozen_config_path = $frozenConfigPath
        fixed_eval_path = $fixedEvalPath
        debug_eval_path = $debugEvalPath
        quasi_eval_path = $quasiEvalPath
        latent_eval_path = $latentEvalPath
        prompt_quasi_eval_path = $promptQuasiEvalPath
        react_quasi_eval_path = $reactQuasiEvalPath
        ablation_eval_path = $ablationEvalPath
        latent_ablation_eval_path = $latentAblationEvalPath
        gate_path = $gatePath
        phase2i_prepackage_gate_path = $phase2iPrePackageGatePath
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $activePath -Encoding UTF8
    [ordered]@{
        pid = $PID
        control_name = $ControlName
        stage = $Stage
        adapter_name = $AdapterName
        started_at = (Get-Date).ToString("o")
        script = $PSCommandPath
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $lockPath -Encoding UTF8
}

function Exit-Lock() {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}

function Get-Phase2DFileSha256([string]$PathValue) {
    $stream = [System.IO.File]::OpenRead($PathValue)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            return ([System.BitConverter]::ToString($sha.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
        } finally {
            $sha.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Freeze-Config() {
    $hash = Get-Phase2DFileSha256 $configPath
    [ordered]@{
        frozen_at = (Get-Date).ToString("o")
        control_name = $ControlName
        config_yaml = $configPath
        config_hash = $hash
        adapter_name = $AdapterName
        learning_rate = $LearningRate
        epochs = $Epochs
        max_length = $MaxLength
        lora_rank = $LoraRank
        lora_alpha = $LoraAlpha
        gradient_accumulation_steps = $GradientAccumulationSteps
        progress_log_interval_steps = $ProgressLogIntervalSteps
        command_intent_loss_weight = $CommandIntentLossWeight
        command_slot_loss_weight = $CommandSlotLossWeight
        use_pairwise_command_reranker = [bool]$UsePairwiseCommandReranker
        pairwise_command_fusion = $PairwiseCommandFusion
        pairwise_command_policy = $PairwiseCommandPolicy
        pairwise_command_max_length = $PairwiseCommandMaxLength
        pairwise_command_top_k = $PairwiseCommandTopK
        command_candidate_encoder = $CommandCandidateEncoder
        latent_fusion = $LatentFusion
        max_train_records = $MaxTrainRecords
        max_val_records = $MaxValRecords
        no_test_feedback = $true
        fixed_test_jsonl = $sourceTestPath
        debug_ood_v2_jsonl = Join-Path $debugRootPath "challenge.jsonl"
        quasi_real_terminal_v1_jsonl = Join-Path $quasiRootPath "challenge.jsonl"
        latent_sensitive_jsonl = Join-Path $latentRootPath "challenge.jsonl"
        extra_train_profile = $ExtraTrainProfile
        extra_train_jsonl = if ($ExtraTrainProfile) { Join-Path $extraTrainRootPath "challenge.jsonl" } else { $null }
        extra_val_profile = $ExtraValProfile
        extra_val_jsonl = if ($ExtraValProfile) { Join-Path $extraValRootPath "challenge.jsonl" } else { $null }
        overfit_guards = [ordered]@{
            no_final_test_feedback_tuning = $true
            extra_profiles_disjoint_from_final_profiles = $true
            hidden_hint_serialization_forbidden = $true
            exact_final_command_hardcoding_forbidden = $true
        }
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $frozenConfigPath -Encoding UTF8
}

function Test-IsPhase2IContext() {
    return (
        $ControlName -like "*phase2i*" -or
        $ConfigYaml -like "*phase2i*" -or
        $AdapterName -like "*phase2i*"
    )
}

function Get-Phase2IDataAuditPath() {
    if ($Phase2IDataAuditJson) {
        return (Resolve-RepoPath $Phase2IDataAuditJson)
    }
    $audit = Get-ChildItem -Path $reportDirPath -Filter "phase2i_data_health_audit*.json" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $audit) {
        throw "Phase2I pre-package gate requires phase2i_data_health_audit*.json under $reportDirPath."
    }
    return $audit.FullName
}

function Assert-Phase2IPrePackageGate() {
    if ($SkipPhase2IPrePackageGate -or -not (Test-IsPhase2IContext)) {
        return
    }
    if (-not (Test-Path $trainSummaryPath)) {
        throw "Phase2I pre-package gate requires training summary at $trainSummaryPath."
    }
    $expectedTrainRecords = if ($MaxTrainRecords -gt 0) { $MaxTrainRecords } else { 1024 }
    $expectedValRecords = if ($MaxValRecords -gt 0) { $MaxValRecords } else { 512 }
    $expectedPairwiseCommandMaxLength = if ($PairwiseCommandMaxLength -gt 0) { $PairwiseCommandMaxLength } else { $MaxLength }
    $expectedPairwiseCommandTopKArgs = @()
    if ($PairwiseCommandTopK -gt 0) {
        $expectedPairwiseCommandTopKArgs = @("--expected-pairwise-command-top-k", $PairwiseCommandTopK)
    }
    $dataAuditPath = Get-Phase2IDataAuditPath
    & $pythonPath -m reflexlm.cli.check_phase2i_prepackage_gates `
        --training-summary-json $trainSummaryPath `
        --data-audit-json $dataAuditPath `
        --expected-adapter-name $AdapterName `
        --expected-train-records $expectedTrainRecords `
        --expected-val-records $expectedValRecords `
        --expected-command-candidate-feature-dim 24 `
        --expected-pairwise-command-fusion residual `
        --expected-pairwise-command-policy $PairwiseCommandPolicy `
        --expected-pairwise-command-max-length $expectedPairwiseCommandMaxLength `
        @expectedPairwiseCommandTopKArgs `
        --expected-command-candidate-encoder $CommandCandidateEncoder `
        --expected-latent-fusion additive `
        --min-val-command-slot-accuracy 0.85 `
        --output-json $phase2iPrePackageGatePath
    if ($LASTEXITCODE -ne 0) {
        throw "Phase2I pre-package gate failed. Refusing to package $AdapterName."
    }
}

function Prepare-Datasets() {
    Freeze-Config
    $headDatasetArgs = @(
        "-m", "reflexlm.cli.build_phase2c_head_dataset",
        "--train-jsonl", $sourceTrainPath,
        "--val-jsonl", $sourceValPath,
        "--test-jsonl", $sourceTestPath,
        "--output-dir", $headDatasetRootPath,
        "--synapse-checkpoint", $nsiCheckpointPath,
        "--synapse-device", "cpu",
        "--run-root", $runRootPath,
        "--output-json", $headManifestPath
    )
    if ($ExtraTrainProfile) {
        & $pythonPath -m reflexlm.cli.generate_debug_cortex_challenge --output $extraTrainRootPath --profile $ExtraTrainProfile --episodes-per-scenario $ExtraTrainEpisodesPerScenario
        if ($LASTEXITCODE -ne 0) { throw "$ControlName extra train generation failed." }
        $headDatasetArgs += @("--extra-train-jsonl", (Join-Path $extraTrainRootPath "challenge.jsonl"))
    }
    if ($ExtraValProfile) {
        & $pythonPath -m reflexlm.cli.generate_debug_cortex_challenge --output $extraValRootPath --profile $ExtraValProfile --episodes-per-scenario $ExtraValEpisodesPerScenario
        if ($LASTEXITCODE -ne 0) { throw "$ControlName extra val generation failed." }
        $headDatasetArgs += @("--extra-val-jsonl", (Join-Path $extraValRootPath "challenge.jsonl"))
    }
    & $pythonPath @headDatasetArgs
    if ($LASTEXITCODE -ne 0) { throw "Phase2D head dataset materialization failed." }
    & $pythonPath -m reflexlm.cli.generate_debug_cortex_challenge --output $debugRootPath --profile debug_ood_v2 --episodes-per-scenario $DebugEpisodesPerScenario
    if ($LASTEXITCODE -ne 0) { throw "Phase2D debug_ood_v2 generation failed." }
    & $pythonPath -m reflexlm.cli.generate_debug_cortex_challenge --output $quasiRootPath --profile quasi_real_terminal --episodes-per-scenario $QuasiEpisodesPerScenario
    if ($LASTEXITCODE -ne 0) { throw "Phase2D quasi_real generation failed." }
    & $pythonPath -m reflexlm.cli.generate_debug_cortex_challenge --output $latentRootPath --profile phase2f_latent_sensitive --episodes-per-scenario $LatentEpisodesPerScenario
    if ($LASTEXITCODE -ne 0) { throw "Phase2F latent-sensitive generation failed." }
    & $pythonPath -m reflexlm.cli.audit_phase2d_dataset `
        --dataset-jsonl (Join-Path $debugRootPath "challenge.jsonl") `
        --train-head-jsonl (Join-Path $headDatasetRootPath "train.jsonl") `
        --manifest-json $debugManifestPath `
        --output-json $debugAuditPath
    if ($LASTEXITCODE -ne 0) { throw "Phase2D debug_ood_v2 audit failed." }
    & $pythonPath -m reflexlm.cli.audit_phase2d_dataset `
        --dataset-jsonl (Join-Path $quasiRootPath "challenge.jsonl") `
        --train-head-jsonl (Join-Path $headDatasetRootPath "train.jsonl") `
        --manifest-json $quasiManifestPath `
        --output-json $quasiAuditPath
    if ($LASTEXITCODE -ne 0) { throw "Phase2D quasi_real audit failed." }
    & $pythonPath -m reflexlm.cli.audit_phase2d_dataset `
        --dataset-jsonl (Join-Path $latentRootPath "challenge.jsonl") `
        --train-head-jsonl (Join-Path $headDatasetRootPath "train.jsonl") `
        --manifest-json $latentManifestPath `
        --output-json $latentAuditPath
    if ($LASTEXITCODE -ne 0) { throw "Phase2F latent-sensitive audit failed." }
}

function Train-FinalAdapter() {
    Assert-LongRunAllowed
    $trainArgs = @(
        "-m", "reflexlm.cli.train_phase2c_native_heads",
        "--base-model-name", $modelPath,
        "--train-jsonl", (Join-Path $headDatasetRootPath "train.jsonl"),
        "--val-jsonl", (Join-Path $headDatasetRootPath "val.jsonl"),
        "--output-dir", $adapterPath,
        "--adapter-name", $AdapterName,
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
        "--debug-command-oversample", "1",
        "--balance-debug-command-intents",
        "--pairwise-command-fusion", $PairwiseCommandFusion,
        "--pairwise-command-policy", $PairwiseCommandPolicy,
        "--command-candidate-encoder", $CommandCandidateEncoder,
        "--latent-fusion", $LatentFusion,
        "--device", "cuda",
        "--run-root", $runRootPath,
        "--output-json", $trainSummaryPath
    )
    if ($UsePairwiseCommandReranker) {
        $trainArgs += @("--use-pairwise-command-reranker")
    }
    if ($PairwiseCommandMaxLength -gt 0) {
        $trainArgs += @("--pairwise-command-max-length", $PairwiseCommandMaxLength)
    }
    if ($PairwiseCommandTopK -gt 0) {
        $trainArgs += @("--pairwise-command-top-k", $PairwiseCommandTopK)
    }
    if ($MaxTrainRecords -gt 0) {
        $trainArgs += @("--max-train-records", $MaxTrainRecords)
    }
    if ($MaxValRecords -gt 0) {
        $trainArgs += @("--max-val-records", $MaxValRecords)
    }
    & $pythonPath @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Phase2D final adapter training failed." }
}

function Build-Packages() {
    Assert-Phase2IPrePackageGate
    & $pythonPath -m reflexlm.cli.build_phase2d_policy_package `
        --output-dir $packagePath `
        --base-model-name $modelPath `
        --native-head-path $adapterPath `
        --low-level-checkpoint-path $nsiCheckpointPath `
        --quantization 4bit `
        --max-length $MaxLength `
        --nsi-device cpu `
        --device cuda `
        --policy-label $AdapterName
    if ($LASTEXITCODE -ne 0) { throw "Phase2D package build failed." }
    & $pythonPath -m reflexlm.cli.build_phase2d_policy_package `
        --output-dir $ablationPackagePath `
        --base-model-name $modelPath `
        --native-head-path $adapterPath `
        --low-level-checkpoint-path $nsiCheckpointPath `
        --quantization 4bit `
        --max-length $MaxLength `
        --nsi-device cpu `
        --device cuda `
        --policy-label ($AdapterName + "_no_nsi_latent") `
        --zero-nsi-latent
    if ($LASTEXITCODE -ne 0) { throw "Phase2D no-NSI-latent package build failed." }
    & $pythonPath -m reflexlm.cli.build_phase2d_policy_package `
        --output-dir $nativeHeadOnlyPackagePath `
        --base-model-name $modelPath `
        --native-head-path $adapterPath `
        --low-level-checkpoint-path $nsiCheckpointPath `
        --quantization 4bit `
        --max-length $MaxLength `
        --nsi-device cpu `
        --device cuda `
        --policy-label ($AdapterName + "_native_head_only") `
        --disable-continuation-cache
    if ($LASTEXITCODE -ne 0) { throw "Phase2F native-head-only package build failed." }
    & $pythonPath -m reflexlm.cli.build_phase2d_policy_package `
        --output-dir $continuationOnlyPackagePath `
        --base-model-name $modelPath `
        --native-head-path $adapterPath `
        --low-level-checkpoint-path $nsiCheckpointPath `
        --quantization 4bit `
        --max-length $MaxLength `
        --nsi-device cpu `
        --device cuda `
        --policy-label ($AdapterName + "_continuation_only") `
        --disable-native-head-calls
    if ($LASTEXITCODE -ne 0) { throw "Phase2F continuation-only package build failed." }
}

function Evaluate-Package([string]$DatasetPath, [string]$EnvProfile, [string]$RunName, [string]$OutputJson, [string]$Package) {
    Assert-LongRunAllowed
    & $pythonPath -m reflexlm.cli.evaluate `
        --policy phase2d_native_package `
        --policy-package-path $Package `
        --dataset $DatasetPath `
        --env-profile $EnvProfile `
        --run-name $RunName `
        --run-root $runRootPath `
        --output-json $OutputJson
    if ($LASTEXITCODE -ne 0) { throw "Phase2D evaluation failed: $RunName" }
}

function Evaluate-TextBaseline([string]$PolicyName, [string]$RunName, [string]$OutputJson) {
    Assert-LongRunAllowed
    & $pythonPath -m reflexlm.cli.evaluate `
        --policy $PolicyName `
        --dataset (Join-Path $quasiRootPath "challenge.jsonl") `
        --model-name $modelPath `
        --quantization 4bit `
        --device cuda `
        --env-profile quasi_real_terminal `
        --max-new-tokens 96 `
        --max-time-s 20 `
        --max-retries 1 `
        --run-name $RunName `
        --run-root $runRootPath `
        --output-json $OutputJson
    if ($LASTEXITCODE -ne 0) { throw "Phase2D text baseline evaluation failed: $RunName" }
}

function Run-Gate() {
    $gateArgs = @(
        "-m", "reflexlm.cli.check_phase2d_gates",
        "--fixed-eval-json", $fixedEvalPath,
        "--debug-ood-eval-json", $debugEvalPath,
        "--quasi-real-eval-json", $quasiEvalPath,
        "--prompt-quasi-eval-json", $promptQuasiEvalPath,
        "--react-quasi-eval-json", $reactQuasiEvalPath,
        "--no-nsi-latent-eval-json", $ablationEvalPath,
        "--config-json", $frozenConfigPath,
        "--output-json", $gatePath,
        "--no-fail"
    )
    if ((Test-Path $latentEvalPath) -and (Test-Path $latentAblationEvalPath)) {
        $gateArgs += @(
            "--latent-sensitive-eval-json", $latentEvalPath,
            "--latent-sensitive-no-nsi-eval-json", $latentAblationEvalPath
        )
    } elseif ($ControlName -like "phase2f*") {
        throw "Phase2F gate requires latent-sensitive normal and no-NSI-latent evaluations."
    }
    & $pythonPath @gateArgs
    if ($LASTEXITCODE -ne 0) { throw "Phase2D gate command failed." }
}

function Build-BaselineTable() {
    & $pythonPath -m reflexlm.cli.build_phase2f_baseline_table `
        --report-dir $reportDirPath `
        --output-json $baselineTablePath `
        --output-md (Join-Path $reportDirPath "phase2f_exact_baseline_table.md")
    if ($LASTEXITCODE -ne 0) { throw "Phase2F baseline table build failed." }
}

function Archive-Phase2F() {
    & $pythonPath -m reflexlm.cli.archive_phase2f_evidence `
        --report-dir $reportDirPath `
        --package-root $packageRootPath `
        --paper-path (Join-Path $repoRoot "paper_draft.md") `
        --output-dir (Join-Path $repoRoot "artifacts\archives\phase2f_rich_latent_fusion_20260517")
    if ($LASTEXITCODE -ne 0) { throw "Phase2F evidence archive failed." }
}

Enter-Lock
try {
    switch ($Stage) {
        "freeze" { Freeze-Config }
        "prepare" { Prepare-Datasets }
        "train" { Prepare-Datasets; Train-FinalAdapter }
        "package" { Build-Packages }
        "evaluate-fixed" { Evaluate-Package $sourceTestPath "wide_ood" ($AdapterName + "_fixed") $fixedEvalPath $packagePath }
        "evaluate-debug" { Evaluate-Package (Join-Path $debugRootPath "challenge.jsonl") "debug_ood_v2" ($AdapterName + "_debug_ood_v2") $debugEvalPath $packagePath }
        "evaluate-quasi" { Evaluate-Package (Join-Path $quasiRootPath "challenge.jsonl") "quasi_real_terminal" ($AdapterName + "_quasi_real") $quasiEvalPath $packagePath }
        "evaluate-latent" { Evaluate-Package (Join-Path $latentRootPath "challenge.jsonl") "phase2f_latent_sensitive" ($AdapterName + "_latent_sensitive") $latentEvalPath $packagePath }
        "evaluate-prompt-quasi" { Evaluate-TextBaseline "prompt_only" "phase2d_prompt_only_7b_quasi_real" $promptQuasiEvalPath }
        "evaluate-react-quasi" { Evaluate-TextBaseline "react" "phase2d_react_7b_quasi_real" $reactQuasiEvalPath }
        "evaluate-baselines" {
            Evaluate-TextBaseline "prompt_only" "phase2d_prompt_only_7b_quasi_real" $promptQuasiEvalPath
            Evaluate-TextBaseline "react" "phase2d_react_7b_quasi_real" $reactQuasiEvalPath
        }
        "evaluate-ablation" { Evaluate-Package (Join-Path $debugRootPath "challenge.jsonl") "debug_ood_v2" ($AdapterName + "_no_nsi_latent_debug_ood_v2") $ablationEvalPath $ablationPackagePath }
        "evaluate-latent-ablation" { Evaluate-Package (Join-Path $latentRootPath "challenge.jsonl") "phase2f_latent_sensitive" ($AdapterName + "_latent_sensitive_no_nsi_latent") $latentAblationEvalPath $ablationPackagePath }
        "evaluate-native-head-only" { Evaluate-Package (Join-Path $debugRootPath "challenge.jsonl") "debug_ood_v2" ($AdapterName + "_native_head_only_debug_ood_v2") $nativeHeadOnlyEvalPath $nativeHeadOnlyPackagePath }
        "evaluate-continuation-only" { Evaluate-Package (Join-Path $debugRootPath "challenge.jsonl") "debug_ood_v2" ($AdapterName + "_continuation_only_debug_ood_v2") $continuationOnlyEvalPath $continuationOnlyPackagePath }
        "baseline-table" { Build-BaselineTable }
        "archive" { Archive-Phase2F }
        "gate" { Run-Gate }
        "all" {
            Prepare-Datasets
            Train-FinalAdapter
            Build-Packages
            Evaluate-Package $sourceTestPath "wide_ood" ($AdapterName + "_fixed") $fixedEvalPath $packagePath
            Evaluate-Package (Join-Path $debugRootPath "challenge.jsonl") "debug_ood_v2" ($AdapterName + "_debug_ood_v2") $debugEvalPath $packagePath
            Evaluate-Package (Join-Path $quasiRootPath "challenge.jsonl") "quasi_real_terminal" ($AdapterName + "_quasi_real") $quasiEvalPath $packagePath
            Evaluate-Package (Join-Path $latentRootPath "challenge.jsonl") "phase2f_latent_sensitive" ($AdapterName + "_latent_sensitive") $latentEvalPath $packagePath
            Evaluate-TextBaseline "prompt_only" "phase2d_prompt_only_7b_quasi_real" $promptQuasiEvalPath
            Evaluate-TextBaseline "react" "phase2d_react_7b_quasi_real" $reactQuasiEvalPath
            Evaluate-Package (Join-Path $debugRootPath "challenge.jsonl") "debug_ood_v2" ($AdapterName + "_no_nsi_latent_debug_ood_v2") $ablationEvalPath $ablationPackagePath
            Evaluate-Package (Join-Path $latentRootPath "challenge.jsonl") "phase2f_latent_sensitive" ($AdapterName + "_latent_sensitive_no_nsi_latent") $latentAblationEvalPath $ablationPackagePath
            Evaluate-Package (Join-Path $debugRootPath "challenge.jsonl") "debug_ood_v2" ($AdapterName + "_native_head_only_debug_ood_v2") $nativeHeadOnlyEvalPath $nativeHeadOnlyPackagePath
            Evaluate-Package (Join-Path $debugRootPath "challenge.jsonl") "debug_ood_v2" ($AdapterName + "_continuation_only_debug_ood_v2") $continuationOnlyEvalPath $continuationOnlyPackagePath
            Run-Gate
            Build-BaselineTable
            Archive-Phase2F
        }
    }
} finally {
    Exit-Lock
}
