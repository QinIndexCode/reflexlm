param(
    [string]$ModelDir = "artifacts\models\Qwen2.5-7B-Instruct",
    [string]$ManifestPath = "configs\models\qwen2.5-7b-instruct-manifest.json"
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

$resolvedModelDir = Resolve-RepoPath $ModelDir
$resolvedManifestPath = Resolve-RepoPath $ManifestPath
$manifest = Get-Content $resolvedManifestPath -Raw | ConvertFrom-Json

$rows = foreach ($entry in $manifest.required_files) {
    $targetPath = Join-Path $resolvedModelDir $entry.name
    $ariaStatePath = "$targetPath.aria2"
    $exists = Test-Path $targetPath
    $ariaStateExists = Test-Path $ariaStatePath
    $actualSize = if ($exists) { (Get-Item $targetPath).Length } else { 0 }
    $progress = if ($ariaStateExists) {
        $null
    }
    elseif ($entry.size_bytes -gt 0) {
        [math]::Round(($actualSize / $entry.size_bytes) * 100, 2)
    }
    else {
        $null
    }
    $status = if (-not $exists) {
        "missing"
    }
    elseif ($ariaStateExists) {
        "downloading"
    }
    elseif ($actualSize -eq $entry.size_bytes) {
        "complete"
    }
    elseif ($actualSize -lt $entry.size_bytes) {
        "partial"
    }
    else {
        "oversize"
    }
    [pscustomobject]@{
        File = $entry.name
        Status = $status
        ExpectedBytes = [int64]$entry.size_bytes
        ActualBytes = [int64]$actualSize
        ProgressPercent = $progress
    }
}

$rows | Sort-Object File | Format-Table -AutoSize

$completeCount = ($rows | Where-Object Status -eq "complete").Count
$partialCount = ($rows | Where-Object Status -eq "partial").Count
$missingCount = ($rows | Where-Object Status -eq "missing").Count
$downloadingCount = ($rows | Where-Object Status -eq "downloading").Count

Write-Host ""
Write-Host ("repo_id: {0}" -f $manifest.repo_id)
Write-Host ("revision: {0}" -f $manifest.revision)
Write-Host ("complete_files: {0}/{1}" -f $completeCount, $rows.Count)
Write-Host ("downloading_files: {0}" -f $downloadingCount)
Write-Host ("partial_files: {0}" -f $partialCount)
Write-Host ("missing_files: {0}" -f $missingCount)

$ariaTempFiles = Get-ChildItem $resolvedModelDir -Filter "*.aria2" -ErrorAction SilentlyContinue
if ($ariaTempFiles) {
    Write-Host ""
    Write-Host "note: files with a .aria2 state file are still downloading. Their file length is not authoritative for completion."
    Write-Host ""
    Write-Host "active_aria2_state_files:"
    $ariaTempFiles | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
}
