param(
    [int]$PollSeconds = 60,
    [string]$ModelDir = "artifacts\models\Qwen2.5-7B-Instruct"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
function Resolve-RepoPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $PathValue))
}

$modelPath = Resolve-RepoPath $ModelDir
$manifestPath = Resolve-RepoPath "configs\models\qwen2.5-7b-instruct-manifest.json"
$phase2Runner = Resolve-RepoPath "scripts\run-phase2-qwen7b.ps1"
$pauseMarker = Resolve-RepoPath "artifacts\control\phase2_7b.paused"
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

function Assert-NotPaused() {
    if (Test-Path $pauseMarker) {
        throw "Phase 2 7B pipeline is paused by $pauseMarker. Remove that marker only after the small-model promotion gate passes."
    }
}

function Test-ModelComplete([string]$ResolvedModelDir, $ResolvedManifest) {
    $activeStateFiles = Get-ChildItem $ResolvedModelDir -Filter "*.aria2" -ErrorAction SilentlyContinue
    if ($activeStateFiles) {
        return $false
    }
    foreach ($entry in $ResolvedManifest.required_files) {
        $targetPath = Join-Path $ResolvedModelDir $entry.name
        if (-not (Test-Path $targetPath)) {
            return $false
        }
        if ((Get-Item $targetPath).Length -ne [int64]$entry.size_bytes) {
            return $false
        }
    }
    return $true
}

Assert-NotPaused
Write-Host ("[{0}] Waiting for complete local 7B model in {1}" -f (Get-Date -Format s), $modelPath)
while (-not (Test-ModelComplete -ResolvedModelDir $modelPath -ResolvedManifest $manifest)) {
    Assert-NotPaused
    Write-Host ("[{0}] Model not complete yet. Sleeping {1}s." -f (Get-Date -Format s), $PollSeconds)
    Start-Sleep -Seconds $PollSeconds
}

Assert-NotPaused
Write-Host ("[{0}] Local 7B model complete. Starting Phase 2 runner." -f (Get-Date -Format s))
& $phase2Runner
