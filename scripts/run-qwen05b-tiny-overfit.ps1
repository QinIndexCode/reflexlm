param(
    [string]$PythonExe = ".venv312-gpu\Scripts\python.exe",
    [string]$BaseModelName = "Qwen/Qwen2.5-0.5B-Instruct",
    [string]$SourceJsonl = "artifacts\datasets\phase2_sft\synapse_augmented\shared\train.jsonl",
    [string]$OutputRoot = "artifacts\qwen05b_tiny_overfit",
    [string]$RunRoot = "artifacts\runs_small_model",
    [string]$OutputJson = "artifacts\reports\qwen05b_tiny_overfit_gate.json"
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
$sourcePath = Resolve-RepoPath $SourceJsonl
$outputRootPath = Resolve-RepoPath $OutputRoot $false
$runRootPath = Resolve-RepoPath $RunRoot $false
$outputJsonPath = Resolve-RepoPath $OutputJson $false
New-Item -ItemType Directory -Force -Path (Split-Path $outputJsonPath -Parent), $outputRootPath, $runRootPath | Out-Null
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }

& $pythonPath -m reflexlm.cli.qwen_tiny_overfit `
    --base-model-name $BaseModelName `
    --source-jsonl $sourcePath `
    --output-root $outputRootPath `
    --run-root $runRootPath `
    --output-json $outputJsonPath `
    --max-examples 64 `
    --min-loss-drop 0.20 `
    --quantization 4bit `
    --epochs 2 `
    --micro-batch-size 1 `
    --gradient-accumulation-steps 4 `
    --max-length 256 `
    --lora-rank 8 `
    --lora-alpha 16
if ($LASTEXITCODE -ne 0) {
    throw "Qwen 0.5B tiny-overfit gate failed to run."
}
