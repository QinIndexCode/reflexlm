from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from reflexlm.cli.collect_phase2m_external_traces import normalize_phase2m_rows
from reflexlm.cli.collect_phase2m_public_repo_traces import (
    CANDIDATE_COUNTS,
    FORBIDDEN_VISIBLE_RE,
    TestCase,
    _baseline_payload,
    _candidate_command,
    _choose_source_file,
    _clone_or_get_repo,
    _detect_license,
    _discover_source_files,
    _discover_tests,
    _read_json,
    _run_git,
    _test_watch_key,
)
from reflexlm.cli.generate_phase2m_synthetic_safe_traces import (
    AMBIGUITY_CLASSES,
    CONTINUATION_DEPTHS,
    EVIDENCE_DENSITIES,
    TRACE_TYPES,
)


SANDBOX_IGNORE = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "node_modules",
}
ABSOLUTE_PATH_RE = re.compile(
    r"(?i)([A-Z]:[\\/][^\s,;:\"']+|\\\\[A-Za-z0-9_.-]+\\[^\s,;:\"']+|/(?:Users|home|root|var/folders)/[^\s,;:\"']+)"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _script_hash() -> str:
    return _sha256_text(Path(__file__).read_text(encoding="utf-8"))


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9_]+", text.lower()) if token]


def _ignore_sandbox_parts(_: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SANDBOX_IGNORE or name.endswith((".pyc", ".pyo"))}


def _copy_repo_to_sandbox(*, repo: Path, sandbox_root: Path, repo_id: str, row_index: int) -> Path:
    target = (sandbox_root / repo_id / f"row_{row_index:05d}").resolve()
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo, target, ignore=_ignore_sandbox_parts)
    return target


def _shorten(text: str, *, limit: int = 2400) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<TRUNCATED>"


def _redact_execution_text(text: str, *, repo: Path, sandbox: Path) -> str:
    sandbox = sandbox.resolve()
    repo = repo.resolve()
    redacted = text.replace(str(sandbox), "<EXECUTION_SANDBOX>")
    redacted = redacted.replace(str(repo), "<SOURCE_REPO>")
    redacted = redacted.replace(sandbox.as_posix(), "<EXECUTION_SANDBOX>")
    redacted = redacted.replace(repo.as_posix(), "<SOURCE_REPO>")
    redacted = redacted.replace("\\", "/")
    redacted = ABSOLUTE_PATH_RE.sub("<REDACTED_ABS_PATH>", redacted)
    redacted = re.sub(r":\n", " -\n", redacted)
    return _shorten(redacted)


def _pytest_outcome(*, exit_code: int | None, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if exit_code == 0:
        return "passed"
    if exit_code == 1:
        return "failed"
    if exit_code == 2:
        return "interrupted_or_usage_error"
    if exit_code == 3:
        return "internal_error"
    if exit_code == 4:
        return "usage_error"
    if exit_code == 5:
        return "no_tests_collected"
    return "unknown"


def _run_pytest_target(
    *,
    repo: Path,
    sandbox: Path,
    command_target: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    args = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        command_target,
        "--maxfail=1",
        "-p",
        "no:cacheprovider",
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    start = time.perf_counter()
    timed_out = False
    exit_code: int | None
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            args,
            cwd=str(sandbox),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        exit_code = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    duration = time.perf_counter() - start
    return {
        "command_template": "python -m pytest -q <target> --maxfail=1 -p no:cacheprovider",
        "target_hash": _sha256_text(command_target)[:16],
        "exit_code": exit_code,
        "timed_out": timed_out,
        "outcome": _pytest_outcome(exit_code=exit_code, timed_out=timed_out),
        "duration_seconds": round(duration, 3),
        "stdout_excerpt": _redact_execution_text(stdout, repo=repo, sandbox=sandbox),
        "stderr_excerpt": _redact_execution_text(stderr, repo=repo, sandbox=sandbox),
    }


def _runtime_evidence_for_dynamic_case(
    *,
    repo: Path,
    sandbox: Path,
    test: TestCase,
    source_file: str,
    evidence_density: str,
    continuation_depth: str,
    trace_type: str,
    active_watch_key: str,
    execution: dict[str, Any],
) -> dict[str, Any]:
    imported_symbols = list(test.imports[:6])
    evidence: dict[str, Any] = {
        "trace_construction": "dynamic_public_repo_pytest_execution_trace",
        "execution_backend": "pytest_subprocess_isolated_sandbox",
        "pytest_execution": execution,
        "changed_files": [source_file],
        "traceback_symbols": imported_symbols or _tokens(source_file)[:4],
        "watched_files": [test.rel_path],
        "prior_read_summary": (
            "Dynamic public repository trace: an isolated pytest subprocess was observed, "
            f"then related back to source module {source_file} using only public repo evidence."
        ),
        "stale_state_refresh": continuation_depth == "stale_state_refresh",
        "active_watch_key": active_watch_key,
        "active_watch_key_origin": "dynamic_public_repo_pytest_execution_relation",
        "source_repo_observed_read_only": True,
        "execution_sandbox_used": True,
        "execution_sandbox_path_visible": False,
    }
    if evidence_density in {"medium", "high"}:
        evidence["module_owner"] = ".".join(_tokens(source_file)[:3]) or "unknown_module"
    if evidence_density == "high":
        evidence["execution_stdout_signal_hash"] = _sha256_text(
            str(execution.get("stdout_excerpt") or "")
        )[:16]
        evidence["execution_stderr_signal_hash"] = _sha256_text(
            str(execution.get("stderr_excerpt") or "")
        )[:16]
    if trace_type == "changed_file_to_watched_test":
        evidence["watched_files"] = [test.rel_path]
    elif trace_type == "module_ownership_to_command":
        evidence["module_owner"] = ".".join(_tokens(source_file)[:4]) or "unknown_module"
    elif trace_type == "stale_state_refresh":
        evidence["stale_state_refresh"] = True
        evidence["refresh_reason"] = "dynamic execution refreshed a stale watch relation"
    # The sandbox is a disposable execution copy; exposing its absolute path would be a shortcut.
    _ = repo, sandbox
    return evidence


def build_dynamic_public_repo_rows_for_spec(
    spec: dict[str, Any],
    *,
    clone_root: Path,
    sandbox_root: Path,
    rows_per_repo: int,
    timeout_seconds: int,
    no_clone: bool = False,
    structured_watch_keys: bool = True,
    keep_sandboxes: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_id = str(spec.get("repo_id") or Path(str(spec.get("local_path") or "repo")).name)
    split = str(spec.get("split") or "train")
    if split not in {"train", "val", "holdout"}:
        raise ValueError(f"invalid split for {repo_id}: {split}")
    repo = _clone_or_get_repo(spec, clone_root=clone_root, no_clone=no_clone)
    git_status_before = _run_git(repo, ["status", "--short"])
    commit_hash = _run_git(repo, ["rev-parse", "HEAD"])
    repo_url = str(spec.get("repo_url") or spec.get("url") or f"file://{repo.as_posix()}")
    license_name = _detect_license(repo, spec.get("license"))
    test_groups = _discover_tests(repo)
    source_files = _discover_source_files(repo)
    eligible_groups = [
        cases
        for _, cases in sorted(test_groups.items())
        if len(cases) >= max(CANDIDATE_COUNTS)
    ]
    rows: list[dict[str, Any]] = []
    rejected_reasons: list[str] = []
    execution_outcomes: dict[str, int] = {}
    if not eligible_groups:
        rejected_reasons.append("no_test_file_with_four_or_more_pytest_functions")
    attempts = 0
    slot_offsets = {candidate_count: 0 for candidate_count in CANDIDATE_COUNTS}
    max_attempts = max(rows_per_repo * 8, rows_per_repo + 12)
    while eligible_groups and len(rows) < rows_per_repo and attempts < max_attempts:
        index = attempts
        candidate_count = CANDIDATE_COUNTS[index % len(CANDIDATE_COUNTS)]
        evidence_density = EVIDENCE_DENSITIES[index % len(EVIDENCE_DENSITIES)]
        continuation_depth = CONTINUATION_DEPTHS[index % len(CONTINUATION_DEPTHS)]
        ambiguity_class = AMBIGUITY_CLASSES[index % len(AMBIGUITY_CLASSES)]
        trace_type = TRACE_TYPES[index % len(TRACE_TYPES)]
        group = eligible_groups[index % len(eligible_groups)]
        max_start = max(0, len(group) - candidate_count)
        start = (index // len(CANDIDATE_COUNTS)) % (max_start + 1)
        candidates = group[start : start + candidate_count]
        expected_slot = slot_offsets[candidate_count] % candidate_count
        slot_offsets[candidate_count] += 1
        expected = candidates[expected_slot]
        commands = [_candidate_command(test) for test in candidates]
        watch_keys = [
            _test_watch_key(repo_id=repo_id, commit_hash=commit_hash, test=test)
            for test in candidates
        ]
        if len(set(commands)) != len(commands):
            attempts += 1
            continue
        if structured_watch_keys and len(set(watch_keys)) != len(watch_keys):
            attempts += 1
            continue
        sandbox = _copy_repo_to_sandbox(
            repo=repo,
            sandbox_root=sandbox_root,
            repo_id=repo_id,
            row_index=index,
        )
        command_target = commands[expected_slot].replace("python -m pytest -q ", "", 1)
        try:
            execution = _run_pytest_target(
                repo=repo,
                sandbox=sandbox,
                command_target=command_target,
                timeout_seconds=timeout_seconds,
            )
        finally:
            if not keep_sandboxes:
                shutil.rmtree(sandbox, ignore_errors=True)
        execution_outcomes[str(execution["outcome"])] = (
            execution_outcomes.get(str(execution["outcome"]), 0) + 1
        )
        source_file = _choose_source_file(
            source_files=source_files,
            test=expected,
            row_index=index,
        )
        runtime_visible_evidence = _runtime_evidence_for_dynamic_case(
            repo=repo,
            sandbox=sandbox,
            test=expected,
            source_file=source_file,
            evidence_density=evidence_density,
            continuation_depth=continuation_depth,
            trace_type=trace_type,
            active_watch_key=watch_keys[expected_slot],
            execution=execution,
        )
        current_visible_text = (
            "Public repository dynamic regression triage. Same-intent pytest rerun "
            "commands are available from one test module; choose using runtime-visible "
            f"dynamic execution, source, traceback, watched-file, and prior-read evidence. "
            f"repo={repo_id}; density={evidence_density}; ambiguity={ambiguity_class}."
        )
        row = {
            "trace_id": f"{split}:{repo_id}:{commit_hash[:12]}:dynamic:{index}",
            "split": split,
            "source_kind": "public_repo",
            "repo_id": repo_id,
            "repo_url_or_origin": repo_url,
            "commit_hash": commit_hash,
            "license_or_synthetic_origin": license_name,
            "trace_construction_mode": "dynamic_public_repo_pytest_execution_trace",
            "current_visible_text": current_visible_text,
            "runtime_visible_evidence": runtime_visible_evidence,
            "command_candidates": [
                {
                    "command": command,
                    "intent": "test_rerun",
                    "source": "public_repo_dynamic_pytest_target",
                    **({"watch_key": watch_keys[slot]} if structured_watch_keys else {}),
                }
                for slot, command in enumerate(commands)
            ],
            "expected_command": commands[expected_slot],
            "difficulty": {
                "evidence_density": evidence_density,
                "candidate_count": candidate_count,
                "continuation_depth": continuation_depth,
                "ambiguity_class": ambiguity_class,
                "trace_type": trace_type,
            },
        }
        baselines, metadata = _baseline_payload(row)
        row["baselines"] = baselines
        row["baseline_metadata"] = metadata
        row["trace_hash"] = _sha256_text(_canonical_json(row))
        visible_payload = {
            "current_visible_text": row.get("current_visible_text"),
            "runtime_visible_evidence": row.get("runtime_visible_evidence"),
            "command_candidates": row.get("command_candidates"),
        }
        if FORBIDDEN_VISIBLE_RE.search(_canonical_json(visible_payload)):
            rejected_reasons.append("row_contains_forbidden_visible_marker")
        else:
            rows.append(row)
        attempts += 1
    if len(rows) < rows_per_repo:
        rejected_reasons.append(f"insufficient_rows:{len(rows)}<{rows_per_repo}")
    git_status_after = _run_git(repo, ["status", "--short"])
    if git_status_after != git_status_before:
        rejected_reasons.append("source_repo_git_status_changed_after_collection")
    return rows, {
        "repo_id": repo_id,
        "split": split,
        "repo_path": str(repo),
        "repo_url_or_origin": repo_url,
        "commit_hash": commit_hash,
        "license_or_synthetic_origin": license_name,
        "test_files": len(test_groups),
        "source_files": len(source_files),
        "rows_requested": rows_per_repo,
        "rows_emitted": len(rows),
        "rejected_reasons": sorted(set(rejected_reasons)),
        "structured_watch_keys": structured_watch_keys,
        "include_behavior_summary": False,
        "dynamic_execution_rows": len(rows),
        "execution_outcomes": execution_outcomes,
        "source_repo_git_status_before": git_status_before,
        "source_repo_git_status_after": git_status_after,
        "source_repo_read_only_observed": git_status_before == git_status_after,
    }


def collect_phase2r_dynamic_public_repo_traces(
    *,
    repo_specs: list[dict[str, Any]],
    clone_root: str | Path,
    sandbox_root: str | Path,
    output_root: str | Path,
    rows_per_repo: int = 12,
    timeout_seconds: int = 30,
    no_clone: bool = False,
    keep_sandboxes: bool = False,
) -> dict[str, Any]:
    output = Path(output_root)
    script_hash = _script_hash()
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "holdout": []}
    repo_reports: list[dict[str, Any]] = []
    for spec in repo_specs:
        rows, report = build_dynamic_public_repo_rows_for_spec(
            spec,
            clone_root=Path(clone_root),
            sandbox_root=Path(sandbox_root),
            rows_per_repo=rows_per_repo,
            timeout_seconds=timeout_seconds,
            no_clone=no_clone,
            structured_watch_keys=True,
            keep_sandboxes=keep_sandboxes,
        )
        split = report["split"]
        split_rows[split].extend(rows)
        repo_reports.append(report)
    splits: dict[str, dict[str, Any]] = {}
    for split, rows in split_rows.items():
        raw_path = output / f"{split}.raw.jsonl"
        normalized_path = output / f"{split}.jsonl"
        normalized = normalize_phase2m_rows(
            rows,
            split=split,
            collection_script_hash=script_hash,
        )
        _write_jsonl(raw_path, rows)
        _write_jsonl(normalized_path, normalized)
        splits[split] = {
            "raw_path": str(raw_path),
            "path": str(normalized_path),
            "rows": len(normalized),
            "repo_ids": sorted({str(row.get("repo_id")) for row in normalized}),
        }
    total_rows = sum(split["rows"] for split in splits.values())
    manifest = {
        "collector_family": "phase2r_public_repo_dynamic_execution_trace_collector",
        "trace_construction_mode": "dynamic_public_repo_pytest_execution_trace",
        "sealed_v3_used": False,
        "writes_to_collected_repos": False,
        "execution_sandbox_used": True,
        "execution_sandboxes_retained": keep_sandboxes,
        "collection_script_hash": script_hash,
        "rows_per_repo": rows_per_repo,
        "timeout_seconds": timeout_seconds,
        "structured_watch_keys": True,
        "include_behavior_summary": False,
        "splits": splits,
        "repos": repo_reports,
        "dynamic_execution_rows": sum(
            int(repo.get("dynamic_execution_rows") or 0) for repo in repo_reports
        ),
        "source_repo_read_only_observed": all(
            repo.get("source_repo_read_only_observed") is True for repo in repo_reports
        ),
        "dimensions": {
            "evidence_densities": list(EVIDENCE_DENSITIES),
            "candidate_counts": list(CANDIDATE_COUNTS),
            "continuation_depths": list(CONTINUATION_DEPTHS),
            "ambiguity_classes": list(AMBIGUITY_CLASSES),
            "trace_types": list(TRACE_TYPES),
        },
        "total_rows": total_rows,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect dynamic public repository pytest execution traces for Phase2R."
    )
    parser.add_argument("--repo-spec-json", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--sandbox-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rows-per-repo", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--no-clone", action="store_true")
    parser.add_argument("--keep-sandboxes", action="store_true")
    args = parser.parse_args()
    specs = _read_json(args.repo_spec_json)
    if not isinstance(specs, list):
        raise TypeError("--repo-spec-json must contain a JSON list of repo specs")
    manifest = collect_phase2r_dynamic_public_repo_traces(
        repo_specs=specs,
        clone_root=args.clone_root,
        sandbox_root=args.sandbox_root,
        output_root=args.output_root,
        rows_per_repo=args.rows_per_repo,
        timeout_seconds=args.timeout_seconds,
        no_clone=args.no_clone,
        keep_sandboxes=args.keep_sandboxes,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
