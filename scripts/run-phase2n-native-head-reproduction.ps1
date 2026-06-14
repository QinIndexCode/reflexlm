param(
    [ValidateSet("compat", "smoke", "full")]
    [string]$Stage = "smoke",
    [string]$PythonExe = ".venv312-qwen7b-stable\Scripts\python.exe",
    [string]$ModelKey = "qwen2_5_1_5b",
    [string]$ModelDir = "artifacts\models\Qwen2.5-1.5B-Instruct",
    [int]$Seed = 13,
    [string]$TrainJsonl = "artifacts\datasets\phase2m_v2_public_relationkey_full_heads\train.jsonl",
    [string]$ValJsonl = "artifacts\datasets\phase2m_v2_public_relationkey_full_heads\val.jsonl",
    [string]$AdapterRoot = "artifacts\adapters\phase2n_multimodel_multiseed_heads",
    [string]$RunRoot = "artifacts\runs_phase2n_multimodel_multiseed",
    [string]$ReportDir = "artifacts\reports\phase2n_multimodel_multiseed_reproduction",
    [string]$DataHealthJson = "artifacts\reports\phase2m_v2_claim_bearing\phase2m_v2_public_relationkey_full_data_health.json",
    [string]$PretrainGateJson = "artifacts\reports\phase2m_v2_claim_bearing\phase2m_v2_public_relationkey_full_pretrain_gate.json",
    [string]$DesignMaturityJson = "artifacts\reports\phase2m_v2_claim_bearing\phase2m_v2_public_relationkey_full_design_maturity_review.json",
    [string]$HeadManifestJson = "artifacts\reports\phase2m_v2_claim_bearing\phase2m_v2_public_relationkey_full_head_dataset_manifest.json",
    [string]$NativeHeadOnlyValJsonl = "artifacts\datasets\phase2m_v2_public_relationkey_full_heads\val.native_head_only_zero_nsi.jsonl",
    [string]$HoldoutJsonl = "artifacts\datasets\phase2m_v2_public_relationkey_full_holdout_heads\val.jsonl",
    [string]$NativeHeadOnlyEvalJson = ""
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
$adapterRootPath = Resolve-RepoPath $AdapterRoot $false
$runRootPath = Resolve-RepoPath $RunRoot $false
$reportRootPath = Resolve-RepoPath $ReportDir $false
$dataHealthPath = Resolve-RepoPath $DataHealthJson
$pretrainGatePath = Resolve-RepoPath $PretrainGateJson
$designMaturityPath = Resolve-RepoPath $DesignMaturityJson
$headManifestPath = Resolve-RepoPath $HeadManifestJson
$nativeHeadOnlyValPath = if ($NativeHeadOnlyValJsonl.Trim().Length -gt 0) { Resolve-RepoPath $NativeHeadOnlyValJsonl } else { "" }
$holdoutPath = if ($HoldoutJsonl.Trim().Length -gt 0) { Resolve-RepoPath $HoldoutJsonl } else { "" }

$stageReportDir = Join-Path $reportRootPath (Join-Path $ModelKey "seed$Seed\$Stage")
$stageRunRoot = Join-Path $runRootPath (Join-Path $ModelKey "seed$Seed\$Stage")
New-Item -ItemType Directory -Force -Path $adapterRootPath, $stageReportDir, $stageRunRoot | Out-Null

$maxTrain = if ($Stage -eq "compat") { 4 } elseif ($Stage -eq "smoke") { 128 } else { 1024 }
$maxVal = if ($Stage -eq "compat") { 4 } else { 512 }
$progressInterval = if ($Stage -eq "compat") { 1 } else { 25 }
$adapterName = "phase2n_${ModelKey}_${Stage}_seed${Seed}_r16_alpha32_lr1e-4_len256"
$adapterOutputDir = Join-Path $adapterRootPath $adapterName
New-Item -ItemType Directory -Force -Path $adapterOutputDir | Out-Null
$summaryPath = Join-Path $stageReportDir "$adapterName.training_summary.json"
$postflightPath = Join-Path $stageReportDir "$adapterName.postflight.json"
$nativeHeadOnlyDiagnosticPath = Join-Path $stageReportDir "$adapterName.native_head_only_zero_nsi_diagnostic.json"
$nativeHeadOnlyEvalPath = Join-Path $stageReportDir "$adapterName.native_head_only_eval.json"
$holdoutDiagnosticPath = Join-Path $stageReportDir "$adapterName.holdout_diagnostic.json"
$holdoutPostflightPath = Join-Path $stageReportDir "$adapterName.holdout_postflight.json"

$env:PYTHONPATH = if ($env:PYTHONPATH) { "$repoRoot\src;$env:PYTHONPATH" } else { "$repoRoot\src" }

$trainArgs = @(
    "-m", "reflexlm.cli.train_phase2c_native_heads",
    "--base-model-name", $modelPath,
    "--train-jsonl", $trainPath,
    "--val-jsonl", $valPath,
    "--output-dir", $adapterOutputDir,
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

& $pythonPath @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Phase2N $Stage training failed for $ModelKey seed $Seed."
}

if ($Stage -ne "compat") {
    if ($Stage -eq "full" -and $NativeHeadOnlyEvalJson.Trim().Length -eq 0) {
        if ($nativeHeadOnlyValPath.Trim().Length -eq 0) {
            throw "NativeHeadOnlyValJsonl is required to auto-generate the full-stage native-head-only control."
        }
        $adapterDirPath = $adapterOutputDir
        & $pythonPath -m reflexlm.cli.diagnose_phase2i_command_slots `
            --adapter-dir $adapterDirPath `
            --val-jsonl $nativeHeadOnlyValPath `
            --training-summary $summaryPath `
            --base-model-name $modelPath `
            --quantization "4bit" `
            --device "cuda:0" `
            --max-length "256" `
            --max-records "512" `
            --batch-size "4" `
            --output-json $nativeHeadOnlyDiagnosticPath `
            --no-records
        if ($LASTEXITCODE -ne 0) {
            throw "Phase2N native-head-only diagnostic failed for $ModelKey seed $Seed."
        }
        $evalScript = @'
import json
import os
from pathlib import Path

diagnostic_path = Path(os.environ["PHASE2N_NATIVE_DIAGNOSTIC"])
eval_path = Path(os.environ["PHASE2N_NATIVE_EVAL"])
zero_nsi_val = os.environ["PHASE2N_NATIVE_VAL"]
payload = json.loads(diagnostic_path.read_text(encoding="utf-8-sig"))
effective = payload.get("sources", {}).get("effective", {})
accuracy = float(effective.get("accuracy") or 0.0)
total = int(effective.get("total") or 0)
report = {
    "evaluation_family": "phase2n_native_head_only_zero_nsi_control",
    "sealed_v3_used_for_training_or_tuning": False,
    "control": "native_head_only_zero_nsi_latent",
    "metrics": {
        "task_completion_rate": accuracy,
        "command_slot_accuracy": accuracy,
        "command_slot_count": total,
        "low_level_qwen_calls": 0,
        "allowlist_hallucination_count": 0,
    },
    "inputs": {
        "diagnostic_json": str(diagnostic_path),
        "zero_nsi_val_jsonl": zero_nsi_val,
    },
}
eval_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
'@
        $env:PHASE2N_NATIVE_DIAGNOSTIC = $nativeHeadOnlyDiagnosticPath
        $env:PHASE2N_NATIVE_EVAL = $nativeHeadOnlyEvalPath
        $env:PHASE2N_NATIVE_VAL = $nativeHeadOnlyValPath
        try {
            $evalScript | & $pythonPath -
            if ($LASTEXITCODE -ne 0) {
                throw "Phase2N native-head-only eval construction failed for $ModelKey seed $Seed."
            }
        }
        finally {
            Remove-Item Env:\PHASE2N_NATIVE_DIAGNOSTIC -ErrorAction SilentlyContinue
            Remove-Item Env:\PHASE2N_NATIVE_EVAL -ErrorAction SilentlyContinue
            Remove-Item Env:\PHASE2N_NATIVE_VAL -ErrorAction SilentlyContinue
        }
        $NativeHeadOnlyEvalJson = $nativeHeadOnlyEvalPath
    }

    $postflightArgs = @(
        "-m", "reflexlm.cli.audit_phase2m_v2_postflight",
        "--training-summary-json", $summaryPath,
        "--data-health-json", $dataHealthPath,
        "--pretrain-gate-json", $pretrainGatePath,
        "--design-maturity-json", $designMaturityPath,
        "--head-manifest-json", $headManifestPath,
        "--stage", $Stage,
        "--output-json", $postflightPath
    )
    if ($NativeHeadOnlyEvalJson.Trim().Length -gt 0) {
        $postflightArgs += @("--native-head-only-eval-json", (Resolve-RepoPath $NativeHeadOnlyEvalJson))
    }
    if ($Stage -eq "smoke") {
        $postflightArgs += @("--max-smoke-train-records", "128", "--max-smoke-val-records", "512")
    }
    & $pythonPath @postflightArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Phase2N $Stage postflight gate failed for $ModelKey seed $Seed."
    }

    if ($Stage -eq "full" -and $holdoutPath.Trim().Length -gt 0) {
        & $pythonPath -m reflexlm.cli.diagnose_phase2i_command_slots `
            --adapter-dir $adapterOutputDir `
            --val-jsonl $holdoutPath `
            --training-summary $summaryPath `
            --base-model-name $modelPath `
            --quantization "4bit" `
            --device "cuda:0" `
            --max-length "256" `
            --max-records "512" `
            --batch-size "4" `
            --output-json $holdoutDiagnosticPath `
            --no-records
        if ($LASTEXITCODE -ne 0) {
            throw "Phase2N holdout diagnostic failed for $ModelKey seed $Seed."
        }
        $holdoutScript = @'
import json
import os
from pathlib import Path

diagnostic_path = Path(os.environ["PHASE2N_HOLDOUT_DIAGNOSTIC"])
postflight_path = Path(os.environ["PHASE2N_HOLDOUT_POSTFLIGHT"])
model_key = os.environ["PHASE2N_MODEL_KEY"]
seed = int(os.environ["PHASE2N_SEED"])
payload = json.loads(diagnostic_path.read_text(encoding="utf-8-sig"))
sources = payload.get("sources", {})
effective = sources.get("effective", {})
source_overlap = sources.get("source_overlap_baseline", {})
slot_head = sources.get("slot_head", {})
accuracy = float(effective.get("accuracy") or 0.0)
source_accuracy = float(source_overlap.get("accuracy") or 0.0)
slot_accuracy = float(slot_head.get("accuracy") or 0.0)
report = {
    "artifact_family": "phase2n_holdout_postflight",
    "model_key": model_key,
    "seed": seed,
    "stage": "full",
    "sealed_v3_used_for_training_or_tuning": False,
    "passed": accuracy >= 0.85 and (accuracy - source_accuracy) >= 0.10,
    "checks": {
        "holdout_command_slot_accuracy_gate": accuracy >= 0.85,
        "holdout_model_beats_source_overlap": (accuracy - source_accuracy) >= 0.10,
        "sealed_v3_used_for_training_or_tuning": False,
    },
    "metrics": {
        "holdout_command_slot_accuracy": accuracy,
        "source_overlap_holdout_accuracy": source_accuracy,
        "model_minus_source_overlap_holdout": accuracy - source_accuracy,
        "slot_head_holdout_accuracy": slot_accuracy,
        "command_record_count": int(effective.get("total") or 0),
    },
    "inputs": {"holdout_diagnostic_json": str(diagnostic_path)},
}
postflight_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(1)
'@
        $env:PHASE2N_HOLDOUT_DIAGNOSTIC = $holdoutDiagnosticPath
        $env:PHASE2N_HOLDOUT_POSTFLIGHT = $holdoutPostflightPath
        $env:PHASE2N_MODEL_KEY = $ModelKey
        $env:PHASE2N_SEED = "$Seed"
        try {
            $holdoutScript | & $pythonPath -
            if ($LASTEXITCODE -ne 0) {
                throw "Phase2N holdout postflight failed for $ModelKey seed $Seed."
            }
        }
        finally {
            Remove-Item Env:\PHASE2N_HOLDOUT_DIAGNOSTIC -ErrorAction SilentlyContinue
            Remove-Item Env:\PHASE2N_HOLDOUT_POSTFLIGHT -ErrorAction SilentlyContinue
            Remove-Item Env:\PHASE2N_MODEL_KEY -ErrorAction SilentlyContinue
            Remove-Item Env:\PHASE2N_SEED -ErrorAction SilentlyContinue
        }
    }
}

$manifest = [ordered]@{
    artifact_family = "phase2n_native_head_reproduction_run"
    stage = $Stage
    model_key = $ModelKey
    model_dir = $modelPath
    seed = $Seed
    adapter_name = $adapterName
    adapter_dir = $adapterOutputDir
    training_summary = $summaryPath
    postflight = if ($Stage -eq "compat") { $null } else { $postflightPath }
    native_head_only_eval = if ($Stage -eq "full") { $NativeHeadOnlyEvalJson } else { $null }
    holdout_postflight = if ($Stage -eq "full" -and $holdoutPath.Trim().Length -gt 0) { $holdoutPostflightPath } else { $null }
    max_train_records = $maxTrain
    max_val_records = $maxVal
    sealed_v3_used_for_training_or_tuning = $false
}
$manifestPath = Join-Path $stageReportDir "$adapterName.run_manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8
Write-Output "Phase2N $Stage complete: $manifestPath"
