param(
    [string]$PythonExe = ".venv312-qwen7b-stable\Scripts\python.exe",
    [string]$ModelRoot = "artifacts\models",
    [string]$ReportDir = "artifacts\reports\phase2n_multimodel_multiseed_reproduction",
    [string[]]$Models = @(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct"
    ),
    [string[]]$IgnorePatterns = @(
        "onnx/*",
        "*.onnx",
        "*.onnx_data",
        "*.gguf",
        "*.tflite",
        "*.mlmodel",
        "*.msgpack",
        "*.h5"
    )
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
$modelRootPath = Resolve-RepoPath $ModelRoot $false
$reportDirPath = Resolve-RepoPath $ReportDir $false
New-Item -ItemType Directory -Force -Path $modelRootPath, $reportDirPath | Out-Null

$manifestPath = Join-Path $reportDirPath "phase2n_model_pull_manifest.json"
$modelListJson = ConvertTo-Json @($Models) -Compress
$ignorePatternsJson = ConvertTo-Json @($IgnorePatterns) -Compress

$downloadScript = @'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

models = json.loads(os.environ["PHASE2N_MODELS_JSON"])
ignore_patterns = json.loads(os.environ["PHASE2N_IGNORE_PATTERNS_JSON"])
model_root = Path(os.environ["PHASE2N_MODEL_ROOT"])
manifest_path = Path(os.environ["PHASE2N_MANIFEST_PATH"])

rows = []
for repo_id in models:
    local_name = repo_id.split("/")[-1]
    local_dir = model_root / local_name
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=ignore_patterns,
    )
    files = sorted(path for path in local_dir.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(local_dir)).replace("\\", "/").encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
    rows.append(
        {
            "repo_id": repo_id,
            "local_dir": str(local_dir),
            "file_count": len(files),
            "size_bytes": sum(path.stat().st_size for path in files),
            "layout_hash": digest.hexdigest(),
            "has_config": (local_dir / "config.json").exists(),
            "has_tokenizer": any((local_dir / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")),
            "ignored_patterns": ignore_patterns,
        }
    )

existing_rows = []
if manifest_path.exists():
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        existing_rows = list(existing.get("models") or [])
    except Exception:
        existing_rows = []

merged = {str(row.get("repo_id")): row for row in existing_rows if row.get("repo_id")}
for row in rows:
    previous = merged.get(row["repo_id"], {})
    history = list(previous.get("pull_history") or [])
    history.append(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "file_count": row["file_count"],
            "size_bytes": row["size_bytes"],
            "layout_hash": row["layout_hash"],
        }
    )
    row["pull_history"] = history
    merged[row["repo_id"]] = row

manifest = {
    "artifact_family": "phase2n_model_pull_manifest",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "models": list(merged.values()),
    "last_requested_models": models,
    "ignored_patterns": ignore_patterns,
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
'@

$env:PHASE2N_MODELS_JSON = $modelListJson
$env:PHASE2N_IGNORE_PATTERNS_JSON = $ignorePatternsJson
$env:PHASE2N_MODEL_ROOT = $modelRootPath
$env:PHASE2N_MANIFEST_PATH = $manifestPath
try {
    $downloadScript | & $pythonPath -
    if ($LASTEXITCODE -ne 0) {
        throw "Phase2N model pull failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:\PHASE2N_MODELS_JSON -ErrorAction SilentlyContinue
    Remove-Item Env:\PHASE2N_IGNORE_PATTERNS_JSON -ErrorAction SilentlyContinue
    Remove-Item Env:\PHASE2N_MODEL_ROOT -ErrorAction SilentlyContinue
    Remove-Item Env:\PHASE2N_MANIFEST_PATH -ErrorAction SilentlyContinue
}
