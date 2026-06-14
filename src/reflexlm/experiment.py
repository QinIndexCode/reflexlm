from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "runs"
MAX_RUN_NAME_SLUG_LENGTH = 48


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stable_config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _slugify(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    slug = cleaned.strip("-") or "run"
    return slug[:MAX_RUN_NAME_SLUG_LENGTH].strip("-") or "run"


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _torch_snapshot() -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None
    snapshot: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        snapshot["device_count"] = int(torch.cuda.device_count())
        snapshot["device_names"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return snapshot


def _nvidia_smi_snapshot() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"gpus": rows}


def collect_environment_snapshot() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": {
            name: _package_version(name)
            for name in [
                "numpy",
                "psutil",
                "pydantic",
                "pyyaml",
                "torch",
                "transformers",
                "accelerate",
                "peft",
                "bitsandbytes",
            ]
        },
        "torch": _torch_snapshot(),
        "nvidia_smi": _nvidia_smi_snapshot(),
    }


class ExperimentRun:
    def __init__(
        self,
        *,
        kind: str,
        name: str,
        path: Path,
        config: dict[str, Any],
    ) -> None:
        self.kind = kind
        self.name = name
        self.path = path
        self.config = _jsonable(config)
        self.created_at = datetime.now(UTC)
        self.config_hash = stable_config_hash(self.config)

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        output_path = self.path / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def write_jsonl(self, relative_path: str | Path, rows: list[dict[str, Any]]) -> Path:
        output_path = self.path / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(_jsonable(row), ensure_ascii=False))
                    handle.write("\n")
        except FileNotFoundError:
            # On Windows, nested run directories can be observed late by a
            # child training process. Recreate and retry once instead of
            # failing an otherwise valid evidence run.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(_jsonable(row), ensure_ascii=False))
                    handle.write("\n")
        return output_path

    def write_text(self, relative_path: str | Path, content: str) -> Path:
        output_path = self.path / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    def finalize(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        finished_at = datetime.now(UTC)
        manifest = {
            "kind": self.kind,
            "name": self.name,
            "path": str(self.path),
            "created_at_utc": self.created_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": round((finished_at - self.created_at).total_seconds(), 3),
            "config_hash": self.config_hash,
        }
        if extra:
            manifest.update(_jsonable(extra))
        self.write_json("run_manifest.json", manifest)
        return manifest


def create_experiment_run(
    *,
    kind: str,
    name: str,
    config: dict[str, Any],
    run_root: str | Path | None = None,
) -> ExperimentRun:
    root = Path(run_root) if run_root is not None else DEFAULT_RUN_ROOT
    slug = _slugify(name)
    config_hash = stable_config_hash(config)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / kind / f"{timestamp}-{slug}-{config_hash[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    run = ExperimentRun(kind=kind, name=name, path=path, config=config)
    run.write_json("config.json", config)
    run.write_json("environment.json", collect_environment_snapshot())
    run.write_text("command.txt", " ".join(sys.argv))
    return run
