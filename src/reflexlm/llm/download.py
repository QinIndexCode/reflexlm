from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from reflexlm.experiment import create_experiment_run


@dataclass(slots=True)
class ModelDownloadConfig:
    repo_id: str
    revision: str = "main"
    cache_dir: str | None = None
    local_dir: str | None = None
    token: str | None = None
    allow_patterns: list[str] | None = None
    filenames: list[str] | None = None
    max_workers: int = 4
    dry_run: bool = False
    retries: int = 3
    retry_backoff_seconds: float = 5.0
    etag_timeout: float = 10.0


def download_model_snapshot(
    config: ModelDownloadConfig,
    *,
    run_root: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download, snapshot_download

    run = create_experiment_run(
        kind="model_download",
        name=run_name or config.repo_id.split("/")[-1],
        config={
            "repo_id": config.repo_id,
            "revision": config.revision,
            "cache_dir": config.cache_dir,
            "local_dir": config.local_dir,
            "allow_patterns": config.allow_patterns,
            "filenames": config.filenames,
            "max_workers": config.max_workers,
            "dry_run": config.dry_run,
            "retries": config.retries,
            "retry_backoff_seconds": config.retry_backoff_seconds,
            "etag_timeout": config.etag_timeout,
        },
        run_root=run_root,
    )

    if config.filenames:
        files: list[dict[str, Any]] = []
        for filename in config.filenames:
            if config.dry_run:
                item = hf_hub_download(
                    repo_id=config.repo_id,
                    filename=filename,
                    revision=config.revision,
                    cache_dir=config.cache_dir,
                    local_dir=config.local_dir,
                    token=config.token,
                    etag_timeout=config.etag_timeout,
                    dry_run=True,
                )
                files.append(
                    {
                        "filename": item.filename,
                        "file_size": item.file_size,
                        "is_cached": item.is_cached,
                        "will_download": item.will_download,
                        "local_path": item.local_path,
                    }
                )
                continue

            attempts: list[dict[str, Any]] = []
            last_error: str | None = None
            local_path: str | None = None
            for attempt in range(1, config.retries + 1):
                started_at = time.time()
                try:
                    resolved = hf_hub_download(
                        repo_id=config.repo_id,
                        filename=filename,
                        revision=config.revision,
                        cache_dir=config.cache_dir,
                        local_dir=config.local_dir,
                        token=config.token,
                        etag_timeout=config.etag_timeout,
                    )
                    local_path = str(resolved)
                    attempts.append(
                        {
                            "attempt": attempt,
                            "status": "ok",
                            "duration_seconds": round(time.time() - started_at, 3),
                        }
                    )
                    break
                except Exception as exc:  # pragma: no cover - network/runtime dependent
                    last_error = f"{type(exc).__name__}: {exc}"
                    attempts.append(
                        {
                            "attempt": attempt,
                            "status": "error",
                            "duration_seconds": round(time.time() - started_at, 3),
                            "error": last_error,
                        }
                    )
                    if attempt >= config.retries:
                        raise
                    time.sleep(config.retry_backoff_seconds * attempt)
            files.append(
                {
                    "filename": filename,
                    "status": "downloaded",
                    "local_path": local_path,
                    "attempts": attempts,
                    "last_error": last_error,
                }
            )

        snapshot_payload = (
            {
                "files": files,
                "missing_file_count": sum(1 for item in files if item.get("will_download")),
                "cached_file_count": sum(1 for item in files if item.get("is_cached")),
                "missing_total_bytes": int(
                    sum((item.get("file_size") or 0) for item in files if item.get("will_download"))
                ),
                "cached_total_bytes": int(
                    sum((item.get("file_size") or 0) for item in files if item.get("is_cached"))
                ),
            }
            if config.dry_run
            else {"files": files}
        )
    else:
        snapshot_path = snapshot_download(
            repo_id=config.repo_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
            local_dir=config.local_dir,
            token=config.token,
            allow_patterns=config.allow_patterns,
            max_workers=config.max_workers,
            dry_run=config.dry_run,
        )
        if config.dry_run:
            files = [
                {
                    "filename": item.filename,
                    "file_size": item.file_size,
                    "is_cached": item.is_cached,
                    "will_download": item.will_download,
                    "local_path": item.local_path,
                }
                for item in snapshot_path
            ]
            snapshot_payload = {
                "files": files,
                "missing_file_count": sum(1 for item in files if item["will_download"]),
                "cached_file_count": sum(1 for item in files if item["is_cached"]),
                "missing_total_bytes": int(
                    sum(item["file_size"] for item in files if item["will_download"])
                ),
                "cached_total_bytes": int(
                    sum(item["file_size"] for item in files if item["is_cached"])
                ),
            }
        else:
            snapshot_payload = str(snapshot_path)
    payload = {
        "repo_id": config.repo_id,
        "revision": config.revision,
        "snapshot_path": snapshot_payload,
        "dry_run": config.dry_run,
    }
    payload["run_manifest"] = run.finalize(payload)
    run.write_json("download_summary.json", payload)
    return payload
