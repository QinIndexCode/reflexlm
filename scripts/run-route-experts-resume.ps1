param(
    [string]$PythonExe = ".venv312-qwen7b-stable\Scripts\python.exe",
    [string]$ModelDir = "artifacts\models\Qwen2.5-7B-Instruct",
    [string]$DatasetRoot = "artifacts\datasets\phase2_sft_debug_v3_v20\synapse_augmented\by_route",
    [string]$RunRoot = "artifacts\runs_phase2_debug_v3_v20_stable_full",
    [string]$ReportDir = "artifacts\reports\phase2_debug_v3_v20_stable_full",
    [string]$AdapterRoot = "artifacts\adapters\debug_v3_v20_stable_full\qwen7b_route_experts"
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

New-Item -ItemType Directory -Force -Path $runRootPath, $reportDirPath, $adapterRootPath | Out-Null
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }

& $pythonPath -m reflexlm.cli.train_route_experts `
    --base-model-name $modelPath `
    --dataset-root $datasetRootPath `
    --output-root $adapterRootPath `
    --quantization 4bit `
    --epochs 1 `
    --micro-batch-size 1 `
    --gradient-accumulation-steps 16 `
    --max-length 256 `
    --lora-rank 8 `
    --lora-alpha 16 `
    --device cuda `
    --run-root $runRootPath `
    --output-json (Join-Path $reportDirPath "qwen7b_route_experts_train.json") `
    --resume-existing
if ($LASTEXITCODE -ne 0) {
    throw "Route expert resume training failed."
}
