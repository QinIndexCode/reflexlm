from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[A-Za-z0-9_./+\-]{8,}"
)
WINDOWS_PATH_RE = re.compile(r"(?i)\b[A-Z]:\\[^\s,;:\"']+")
UNC_PATH_RE = re.compile(r"\\\\[^\s,;:\"']+")
POSIX_PATH_RE = re.compile(r"(?<!\w)/(?:Users|home|root|var/folders)/[^\s,;:\"']+")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _redact_text(value: str) -> str:
    redacted = EMAIL_RE.sub("<REDACTED_EMAIL>", value)
    redacted = SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}=<REDACTED_SECRET>", redacted
    )
    redacted = WINDOWS_PATH_RE.sub("<REDACTED_ABS_PATH>", redacted)
    redacted = UNC_PATH_RE.sub("<REDACTED_ABS_PATH>", redacted)
    redacted = POSIX_PATH_RE.sub("<REDACTED_ABS_PATH>", redacted)
    return redacted


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    return value


def _script_hash(collection_script: str | Path | None) -> str:
    if collection_script is None:
        return _sha256_text(Path(__file__).read_text(encoding="utf-8"))
    script = Path(collection_script)
    return _sha256_text(script.read_text(encoding="utf-8-sig"))


def _candidate_commands(row: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for candidate in row.get("command_candidates", []):
        if isinstance(candidate, str):
            commands.append(candidate)
        elif isinstance(candidate, dict) and candidate.get("command") is not None:
            commands.append(str(candidate["command"]))
    return commands


def normalize_phase2m_row(
    raw: dict[str, Any],
    *,
    split: str,
    row_index: int,
    collection_script_hash: str,
) -> dict[str, Any]:
    if split not in {"train", "val", "holdout"}:
        raise ValueError("split must be train, val, or holdout")
    row = _redact(dict(raw))
    row["split"] = split
    row.setdefault("source_kind", "synthetic_safe_repo")
    row.setdefault("repo_id", row.get("repo_url_or_origin") or f"phase2m_repo_{row_index}")
    row.setdefault("repo_url_or_origin", f"synthetic://phase2m/{row['repo_id']}")
    row.setdefault("license_or_synthetic_origin", "synthetic-safe phase2m trace")
    row["collection_script_hash"] = str(
        row.get("collection_script_hash") or collection_script_hash
    )
    row.setdefault("normalization", {})
    if not isinstance(row["normalization"], dict):
        row["normalization"] = {}
    row["normalization"].update(
        {
            "deterministic": True,
            "redacted_absolute_local_paths": True,
            "redacted_secrets_tokens_and_emails": True,
            "preserved_runtime_visible_evidence": True,
        }
    )
    if not row.get("commit_hash"):
        row["commit_hash"] = _sha256_text(f"{row['repo_id']}:{row_index}")[:40]
    commands = _candidate_commands(row)
    expected = row.get("expected_command")
    if expected is not None and expected not in commands:
        raise ValueError(
            f"expected_command is not present in command_candidates for row {row_index}"
        )
    row.setdefault("baselines", {})
    if not isinstance(row["baselines"], dict):
        raise ValueError(f"baselines must be an object for row {row_index}")
    row.setdefault(
        "trace_id",
        f"{split}:{row.get('repo_id')}:{row.get('commit_hash')}:{row_index}",
    )
    stable_for_hash = {key: value for key, value in row.items() if key != "trace_hash"}
    row["trace_hash"] = _sha256_text(_canonical_json(stable_for_hash))
    return row


def normalize_phase2m_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    collection_script_hash: str,
) -> list[dict[str, Any]]:
    return [
        normalize_phase2m_row(
            row,
            split=split,
            row_index=index,
            collection_script_hash=collection_script_hash,
        )
        for index, row in enumerate(rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize read-only external traces into the Phase2M audit schema."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "holdout"])
    parser.add_argument("--collection-script")
    args = parser.parse_args()

    rows = _read_jsonl(args.input_jsonl)
    normalized = normalize_phase2m_rows(
        rows,
        split=args.split,
        collection_script_hash=_script_hash(args.collection_script),
    )
    _write_jsonl(args.output_jsonl, normalized)
    print(
        json.dumps(
            {
                "input_jsonl": str(Path(args.input_jsonl)),
                "output_jsonl": str(Path(args.output_jsonl)),
                "split": args.split,
                "rows": len(normalized),
                "collection_script_hash": _script_hash(args.collection_script),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
