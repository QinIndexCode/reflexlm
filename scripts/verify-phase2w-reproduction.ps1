param(
    [string]$BundleJson = "artifacts/reports/phase2w_epoch_preregistration/phase2w_reproduction_bundle_manifest.json",
    [string]$OutputJson = "artifacts/reports/phase2w_epoch_preregistration/phase2w_reproduction_bundle_verification.json",
    [switch]$RunCommands
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "src;$env:PYTHONPATH"
} else {
    "src"
}

$argsList = @(
    "-m",
    "reflexlm.cli.verify_phase2w_reproduction_bundle",
    "--bundle-json",
    $BundleJson,
    "--output-json",
    $OutputJson
)

if ($RunCommands) {
    $argsList += "--run-commands"
}

python @argsList
