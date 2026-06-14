from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.llm.download import ModelDownloadConfig, download_model_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or resume a Hugging Face model snapshot.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-dir")
    parser.add_argument("--token")
    parser.add_argument("--allow-pattern", action="append")
    parser.add_argument("--file", action="append")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--etag-timeout", type=float, default=10.0)
    parser.add_argument("--run-root")
    parser.add_argument("--run-name")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    payload = download_model_snapshot(
        ModelDownloadConfig(
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_dir=args.local_dir,
            token=args.token,
            allow_patterns=args.allow_pattern,
            filenames=args.file,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            etag_timeout=args.etag_timeout,
        ),
        run_root=args.run_root,
        run_name=args.run_name,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
