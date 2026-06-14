param(
    [string]$PythonExe = ".venv312-gpu\Scripts\python.exe",
    [string]$ModelDir = "artifacts\models\Qwen2.5-7B-Instruct",
    [string]$DatasetRoot = "artifacts\datasets\phase2_sft",
    [string]$RunRoot = "artifacts\runs_phase2",
    [string]$ReportDir = "artifacts\reports",
    [string]$AdapterRoot = "artifacts\adapters",
    [string]$Phase1Dataset = "artifacts\datasets\phase1_default\test.jsonl",
    [string]$NsiCheckpoint = "artifacts\runs_phase1\training\20260512T114823Z-nsi-full-gpu-383997ac\model.pt",
    [string]$EnvProfile = "default",
    [switch]$SkipTraining,
    [switch]$SkipEvaluation
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
$datasetRootPath = Resolve-RepoPath $DatasetRoot
$runRootPath = Resolve-RepoPath $RunRoot $false
$reportDirPath = Resolve-RepoPath $ReportDir $false
$adapterRootPath = Resolve-RepoPath $AdapterRoot $false
$phase1DatasetPath = Resolve-RepoPath $Phase1Dataset
$nsiCheckpointPath = Resolve-RepoPath $NsiCheckpoint
$controlDir = Resolve-RepoPath "artifacts\control" $false
$pauseMarker = Join-Path $controlDir "phase2_7b.paused"
$lockPath = Join-Path $controlDir "phase2_7b.lock"

New-Item -ItemType Directory -Force -Path $runRootPath, $reportDirPath, $adapterRootPath, $controlDir | Out-Null

function Assert-NotPaused() {
    if (Test-Path $pauseMarker) {
        throw "Phase 2 7B pipeline is paused by $pauseMarker. Small-model strict gain evidence is required before resuming."
    }
}

function Enter-Phase2Lock() {
    if (Test-Path $lockPath) {
        throw "Phase 2 7B lock already exists at $lockPath. Refusing to start a duplicate run."
    }
    $payload = [ordered]@{
        pid = $PID
        started_at = (Get-Date).ToString("o")
        script = $PSCommandPath
    }
    $payload | ConvertTo-Json | Set-Content -Path $lockPath -Encoding UTF8
}

function Exit-Phase2Lock() {
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}

Assert-NotPaused
Enter-Phase2Lock
try {
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }

Write-Host "Checking local 7B model completeness..."
& (Resolve-RepoPath "scripts\check-qwen7b-download.ps1") -ModelDir $modelPath | Out-Host

$manifest = Get-Content (Resolve-RepoPath "configs\models\qwen2.5-7b-instruct-manifest.json") -Raw | ConvertFrom-Json
foreach ($entry in $manifest.required_files) {
    $targetPath = Join-Path $modelPath $entry.name
    if (-not (Test-Path $targetPath)) {
        throw "Missing required model file: $($entry.name)"
    }
    if ((Get-Item $targetPath).Length -ne [int64]$entry.size_bytes) {
        throw "Incomplete model file: $($entry.name)"
    }
}

$sharedTrain = Join-Path $datasetRootPath "synapse_augmented\shared\train.jsonl"
$sharedVal = Join-Path $datasetRootPath "synapse_augmented\shared\val.jsonl"
$routeDatasetRoot = Join-Path $datasetRootPath "synapse_augmented\by_route"
$sharedAdapterDir = Join-Path $adapterRootPath "qwen7b_shared_synapse"
$routeAdapterRoot = Join-Path $adapterRootPath "qwen7b_route_experts"
$routeMapPath = Join-Path $routeAdapterRoot "adapter_map.json"

if (-not $SkipTraining) {
    Assert-NotPaused
    & $pythonPath -m reflexlm.cli.train_qwen_qlora `
        --base-model-name $modelPath `
        --train-jsonl $sharedTrain `
        --val-jsonl $sharedVal `
        --output-dir $sharedAdapterDir `
        --adapter-name qwen7b_shared_synapse `
        --quantization 4bit `
        --epochs 1 `
        --micro-batch-size 1 `
        --gradient-accumulation-steps 16 `
        --max-length 256 `
        --lora-rank 8 `
        --lora-alpha 16 `
        --device cuda `
        --run-root $runRootPath `
        --output-json (Join-Path $reportDirPath "qwen7b_shared_synapse_train.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Shared adapter training failed."
    }

    Assert-NotPaused
    & $pythonPath -m reflexlm.cli.train_route_experts `
        --base-model-name $modelPath `
        --dataset-root $routeDatasetRoot `
        --output-root $routeAdapterRoot `
        --quantization 4bit `
        --epochs 1 `
        --micro-batch-size 1 `
        --gradient-accumulation-steps 16 `
        --max-length 256 `
        --lora-rank 8 `
        --lora-alpha 16 `
        --device cuda `
        --run-root $runRootPath `
        --output-json (Join-Path $reportDirPath "qwen7b_route_experts_train.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Route expert training failed."
    }
}

if (-not $SkipEvaluation) {
    Assert-NotPaused
    & $pythonPath -m reflexlm.cli.evaluate `
        --policy prompt_only `
        --dataset $phase1DatasetPath `
        --model-name $modelPath `
        --quantization 4bit `
        --cpu-offload `
        --max-new-tokens 96 `
        --max-time-s 20 `
        --env-profile $EnvProfile `
        --run-name qwen7b_prompt_only_eval `
        --run-root $runRootPath `
        --output-json (Join-Path $reportDirPath "qwen7b_prompt_only_eval.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Prompt-only evaluation failed."
    }

    Assert-NotPaused
    & $pythonPath -m reflexlm.cli.evaluate `
        --policy react `
        --dataset $phase1DatasetPath `
        --model-name $modelPath `
        --quantization 4bit `
        --cpu-offload `
        --max-new-tokens 96 `
        --max-time-s 20 `
        --env-profile $EnvProfile `
        --run-name qwen7b_react_eval `
        --run-root $runRootPath `
        --output-json (Join-Path $reportDirPath "qwen7b_react_eval.json")
    if ($LASTEXITCODE -ne 0) {
        throw "ReAct evaluation failed."
    }

    Assert-NotPaused
    & $pythonPath -m reflexlm.cli.evaluate `
        --policy qwen_adapter `
        --dataset $phase1DatasetPath `
        --model-name $modelPath `
        --adapter-path $sharedAdapterDir `
        --policy-label qwen7b_shared_adapter `
        --quantization 4bit `
        --cpu-offload `
        --max-new-tokens 96 `
        --max-time-s 20 `
        --env-profile $EnvProfile `
        --run-name qwen7b_shared_adapter_eval `
        --run-root $runRootPath `
        --output-json (Join-Path $reportDirPath "qwen7b_shared_adapter_eval.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Shared adapter evaluation failed."
    }

    Assert-NotPaused
    & $pythonPath -m reflexlm.cli.evaluate `
        --policy hybrid_synaptic_qwen `
        --dataset $phase1DatasetPath `
        --model-name $modelPath `
        --adapter-path $sharedAdapterDir `
        --adapter-map-json $routeMapPath `
        --nsi-checkpoint $nsiCheckpointPath `
        --quantization 4bit `
        --cpu-offload `
        --device cuda `
        --nsi-device cpu `
        --max-new-tokens 96 `
        --max-time-s 20 `
        --confidence-threshold 0.72 `
        --prediction-error-threshold 0.45 `
        --risk-threshold 0.70 `
        --env-profile $EnvProfile `
        --run-name qwen7b_hybrid_eval `
        --run-root $runRootPath `
        --output-json (Join-Path $reportDirPath "qwen7b_hybrid_eval.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Hybrid evaluation failed."
    }

    $phase1ComparisonPath = Join-Path $reportDirPath "phase1_local_comparison.json"
    if (Test-Path $phase1ComparisonPath) {
        Assert-NotPaused
        $phase1Comparison = Get-Content $phase1ComparisonPath -Raw | ConvertFrom-Json
        $promptEval = Get-Content (Join-Path $reportDirPath "qwen7b_prompt_only_eval.json") -Raw | ConvertFrom-Json
        $reactEval = Get-Content (Join-Path $reportDirPath "qwen7b_react_eval.json") -Raw | ConvertFrom-Json
        $sharedEval = Get-Content (Join-Path $reportDirPath "qwen7b_shared_adapter_eval.json") -Raw | ConvertFrom-Json
        $hybridEval = Get-Content (Join-Path $reportDirPath "qwen7b_hybrid_eval.json") -Raw | ConvertFrom-Json

        & $pythonPath -m reflexlm.cli.compare_runs `
            --run-dir $phase1Comparison.runs.rule_oracle.run_dir `
            --run-dir $phase1Comparison.runs.nsi_small_model.run_dir `
            --run-dir $phase1Comparison.runs.flat_text_small_model.run_dir `
            --run-dir $promptEval.run_path `
            --run-dir $reactEval.run_path `
            --run-dir $sharedEval.run_path `
            --run-dir $hybridEval.run_path `
            --reference-label nsi_small_model `
            --label-order nsi_small_model `
            --label-order qwen_prompt_only_7b `
            --label-order qwen_react_7b `
            --label-order rule_oracle `
            --label-order flat_text_small_model `
            --label-order qwen7b_shared_adapter `
            --label-order hybrid_synaptic_qwen7b `
            --run-name phase2_qwen7b_full_comparison `
            --run-root $runRootPath `
            --output-json (Join-Path $reportDirPath "phase2_qwen7b_full_comparison.json") `
            --output-markdown-dir (Join-Path $reportDirPath "phase2_qwen7b_full_comparison")
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 2 comparison generation failed."
        }
    }
}
} finally {
    Exit-Phase2Lock
}
