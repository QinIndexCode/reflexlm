param(
    [ValidateSet("smoke", "full", "all")]
    [string]$Stage = "all",
    [string]$PythonExe = ".venv312-qwen7b-stable\Scripts\python.exe",
    [string]$ModelKey = "qwen2_5_3b",
    [string]$ModelDir = "artifacts\models\Qwen2.5-3B-Instruct",
    [int]$Seed = 17,
    [string]$HeadDatasetRoot = "artifacts\datasets\phase2s_public_repair_r128_heads",
    [string]$HoldoutDatasetRoot = "artifacts\datasets\phase2s_public_repair_r128_holdout_heads",
    [string]$AdapterRoot = "artifacts\adapters\phase2s_multimodel_multiseed_heads",
    [string]$RunRoot = "artifacts\runs_phase2s_multimodel_multiseed",
    [string]$ReportDir = "artifacts\reports\phase2s_multimodel_multiseed_reproduction",
    [string]$DataHealthJson = "artifacts\reports\phase2s_public_repair_r128\phase2s_public_repair_r128_data_health.json",
    [string]$PretrainGateJson = "artifacts\reports\phase2s_public_repair_r128\phase2s_public_repair_r128_pretrain_gate.json",
    [string]$HeadManifestJson = "artifacts\reports\phase2s_public_repair_r128\phase2s_public_repair_r128_head_dataset_manifest.json"
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

function Invoke-Checked([string]$Description, [string]$Exe, [string[]]$Arguments) {
    Write-Host "==> $Description"
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Get-HoldoutRecordCount([string]$ManifestPath) {
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    return [int]$manifest.splits.val.rows
}

$pythonPath = Resolve-RepoPath $PythonExe
$modelPath = Resolve-RepoPath $ModelDir
$headRoot = Resolve-RepoPath $HeadDatasetRoot
$holdoutRoot = Resolve-RepoPath $HoldoutDatasetRoot
$adapterRootPath = Resolve-RepoPath $AdapterRoot $false
$runRootPath = Resolve-RepoPath $RunRoot $false
$reportRootPath = Resolve-RepoPath $ReportDir $false
$dataHealthPath = Resolve-RepoPath $DataHealthJson
$pretrainGatePath = Resolve-RepoPath $PretrainGateJson
$headManifestPath = Resolve-RepoPath $HeadManifestJson
$trainJsonl = Resolve-RepoPath (Join-Path $HeadDatasetRoot "train.jsonl")
$valJsonl = Resolve-RepoPath (Join-Path $HeadDatasetRoot "val.jsonl")
$valZeroNsiJsonl = Resolve-RepoPath (Join-Path $HeadDatasetRoot "val.zero_nsi.jsonl")
$holdoutJsonl = Resolve-RepoPath (Join-Path $HoldoutDatasetRoot "val.jsonl")
$holdoutZeroNsiJsonl = Resolve-RepoPath (Join-Path $HoldoutDatasetRoot "val.zero_nsi.jsonl")
$holdoutManifest = Resolve-RepoPath (Join-Path $HoldoutDatasetRoot "manifest.json")
$holdoutRecords = Get-HoldoutRecordCount $holdoutManifest

New-Item -ItemType Directory -Force -Path $adapterRootPath, $runRootPath, $reportRootPath | Out-Null
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }

function Run-Phase2SStage([ValidateSet("smoke", "full")] [string]$RunStage) {
    $maxTrain = if ($RunStage -eq "smoke") { 128 } else { 1024 }
    $maxVal = if ($RunStage -eq "smoke") { 512 } else { 768 }
    $progressInterval = if ($RunStage -eq "smoke") { 10 } else { 50 }
    $adapterName = "phase2s_${ModelKey}_${RunStage}_seed${Seed}_r16_alpha32_lr1e-4_len256_featureonly"
    $adapterDir = Join-Path $adapterRootPath $adapterName
    $stageRunRoot = Join-Path $runRootPath (Join-Path $ModelKey "seed$Seed\$RunStage")
    $stageReportDir = Join-Path $reportRootPath (Join-Path $ModelKey "seed$Seed\$RunStage")
    New-Item -ItemType Directory -Force -Path $adapterDir, $stageRunRoot, $stageReportDir | Out-Null

    $summaryPath = Join-Path $stageReportDir "$adapterName.training_summary.json"
    $zeroNsiPath = Join-Path $stageReportDir "$adapterName.command_slot_diagnostics_zero_nsi.json"
    $postflightPath = Join-Path $stageReportDir "$adapterName.postflight.json"
    $holdoutPath = Join-Path $stageReportDir "$adapterName.holdout_diagnostics.json"
    $holdoutZeroNsiPath = Join-Path $stageReportDir "$adapterName.holdout_diagnostics_zero_nsi.json"
    $holdoutPostflightPath = Join-Path $stageReportDir "$adapterName.full_holdout_postflight.json"

    $trainArgs = @(
        "-m", "reflexlm.cli.train_phase2c_native_heads",
        "--base-model-name", $modelPath,
        "--train-jsonl", $trainJsonl,
        "--val-jsonl", $valJsonl,
        "--output-dir", $adapterDir,
        "--adapter-name", $adapterName,
        "--quantization", "4bit",
        "--learning-rate", "0.0001",
        "--epochs", "1",
        "--micro-batch-size", "1",
        "--gradient-accumulation-steps", "4",
        "--max-length", "256",
        "--lora-rank", "16",
        "--lora-alpha", "32",
        "--seed", "$Seed",
        "--device", "cuda",
        "--max-train-records", "$maxTrain",
        "--max-val-records", "$maxVal",
        "--progress-log-interval-steps", "$progressInterval",
        "--command-slot-loss-weight", "2.0",
        "--command-candidate-encoder", "features_only",
        "--latent-fusion", "additive",
        "--run-root", $stageRunRoot,
        "--output-json", $summaryPath
    )
    Invoke-Checked "Phase2S $ModelKey seed$Seed $RunStage train" $pythonPath $trainArgs

    if ($RunStage -eq "smoke") {
        $diagArgs = @(
            "-m", "reflexlm.cli.diagnose_phase2i_command_slots",
            "--adapter-dir", $adapterDir,
            "--val-jsonl", $valZeroNsiJsonl,
            "--training-summary", $summaryPath,
            "--base-model-name", $modelPath,
            "--quantization", "4bit",
            "--device", "cuda:0",
            "--max-length", "256",
            "--max-records", "512",
            "--batch-size", "4",
            "--output-json", $zeroNsiPath,
            "--no-records"
        )
        Invoke-Checked "Phase2S $ModelKey seed$Seed smoke zero-NSI diagnostic" $pythonPath $diagArgs
        $postArgs = @(
            "-m", "reflexlm.cli.audit_phase2s_open_repair", "smoke-postflight",
            "--training-summary-json", $summaryPath,
            "--data-health-json", $dataHealthPath,
            "--pretrain-gate-json", $pretrainGatePath,
            "--head-manifest-json", $headManifestPath,
            "--zero-nsi-diagnostics-json", $zeroNsiPath,
            "--min-model-minus-zero-nsi", "0.15",
            "--output-json", $postflightPath
        )
        Invoke-Checked "Phase2S $ModelKey seed$Seed smoke postflight" $pythonPath $postArgs
    }
    else {
        $holdoutArgs = @(
            "-m", "reflexlm.cli.diagnose_phase2i_command_slots",
            "--adapter-dir", $adapterDir,
            "--val-jsonl", $holdoutJsonl,
            "--training-summary", $summaryPath,
            "--base-model-name", $modelPath,
            "--quantization", "4bit",
            "--device", "cuda:0",
            "--max-length", "256",
            "--max-records", "$holdoutRecords",
            "--batch-size", "4",
            "--output-json", $holdoutPath,
            "--no-records"
        )
        Invoke-Checked "Phase2S $ModelKey seed$Seed full holdout diagnostic" $pythonPath $holdoutArgs
        $holdoutZeroArgs = @(
            "-m", "reflexlm.cli.diagnose_phase2i_command_slots",
            "--adapter-dir", $adapterDir,
            "--val-jsonl", $holdoutZeroNsiJsonl,
            "--training-summary", $summaryPath,
            "--base-model-name", $modelPath,
            "--quantization", "4bit",
            "--device", "cuda:0",
            "--max-length", "256",
            "--max-records", "$holdoutRecords",
            "--batch-size", "4",
            "--output-json", $holdoutZeroNsiPath,
            "--no-records"
        )
        Invoke-Checked "Phase2S $ModelKey seed$Seed full holdout zero-NSI diagnostic" $pythonPath $holdoutZeroArgs
        $fullPostArgs = @(
            "-m", "reflexlm.cli.audit_phase2s_open_repair", "full-holdout-postflight",
            "--training-summary-json", $summaryPath,
            "--data-health-json", $dataHealthPath,
            "--pretrain-gate-json", $pretrainGatePath,
            "--head-manifest-json", $headManifestPath,
            "--holdout-diagnostics-json", $holdoutPath,
            "--holdout-zero-nsi-diagnostics-json", $holdoutZeroNsiPath,
            "--min-holdout-model-minus-zero-nsi", "0.15",
            "--output-json", $holdoutPostflightPath
        )
        Invoke-Checked "Phase2S $ModelKey seed$Seed full holdout postflight" $pythonPath $fullPostArgs
    }
}

if ($Stage -eq "smoke" -or $Stage -eq "all") {
    Run-Phase2SStage "smoke"
}
if ($Stage -eq "full" -or $Stage -eq "all") {
    Run-Phase2SStage "full"
}

Write-Host "Phase2S $ModelKey seed$Seed $Stage complete."
