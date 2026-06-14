from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2m_external_generalization import (
    BASELINE_METHODS,
    compute_phase2m_baseline_predictions,
)
from reflexlm.cli.collect_phase2m_external_traces import normalize_phase2m_rows
from reflexlm.cli.generate_phase2m_synthetic_safe_traces import (
    AMBIGUITY_CLASSES,
    CANDIDATE_COUNTS,
    CONTINUATION_DEPTHS,
    EVIDENCE_DENSITIES,
    TRACE_TYPES,
)


TEST_FUNC_RE = re.compile(r"^test_[A-Za-z0-9_]+$")
FORBIDDEN_VISIBLE_RE = re.compile(
    r"(?i)(candidate[_-]?\d+|gold|hidden_hint|sealed|external_trace_v3)"
)
PY_CACHE_PARTS = {"__pycache__", ".venv", "venv", ".tox", ".nox", "site-packages"}


@dataclass(frozen=True)
class TestCase:
    rel_path: str
    function_name: str
    body_text: str
    imports: tuple[str, ...]
    line: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


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


def _run_git(repo: Path, args: list[str], *, timeout: int = 120) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def _safe_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _should_skip_path(path: Path) -> bool:
    return any(part in PY_CACHE_PARTS or part.startswith(".") for part in path.parts)


def _is_test_file(path: Path) -> bool:
    name = path.name
    return (name.startswith("test_") or name.endswith("_test.py")) and path.suffix == ".py"


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9_]+", text.lower()) if token]


def _summarize_test_body(body_text: str, function_name: str, *, max_tokens: int) -> str:
    stripped = re.sub(r"#.*", " ", body_text)
    stripped = re.sub(r"(?i)\b" + re.escape(function_name) + r"\b", " ", stripped)
    stripped = re.sub(r"(?i)\bcandidate[_-]?\d+\b", " ", stripped)
    tokens = [
        token
        for token in _tokens(stripped)
        if token
        not in {
            "def",
            "assert",
            "with",
            "pytest",
            "raises",
            "self",
            "true",
            "false",
            "none",
        }
    ]
    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
        if len(deduped) >= max_tokens:
            break
    return " ".join(deduped)


def _extract_imports(module: ast.Module) -> tuple[str, ...]:
    imports: list[str] = []
    for node in module.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
    return tuple(sorted(set(imports)))


def _discover_tests(repo: Path) -> dict[str, list[TestCase]]:
    grouped: dict[str, list[TestCase]] = {}
    for path in sorted(repo.rglob("*.py")):
        rel_path = _safe_rel(path, repo)
        if _should_skip_path(Path(rel_path)) or not _is_test_file(path):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        imports = _extract_imports(module)
        cases: list[TestCase] = []
        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not TEST_FUNC_RE.match(node.name):
                continue
            body_text = ast.get_source_segment(source, node) or node.name
            if FORBIDDEN_VISIBLE_RE.search(node.name) or FORBIDDEN_VISIBLE_RE.search(body_text):
                continue
            cases.append(
                TestCase(
                    rel_path=rel_path,
                    function_name=node.name,
                    body_text=body_text,
                    imports=imports,
                    line=int(getattr(node, "lineno", 1)),
                )
            )
        if len(cases) >= min(CANDIDATE_COUNTS):
            grouped[rel_path] = cases
    return grouped


def _discover_source_files(repo: Path) -> list[str]:
    source_files: list[str] = []
    for path in sorted(repo.rglob("*.py")):
        rel = _safe_rel(path, repo)
        rel_path = Path(rel)
        if _should_skip_path(rel_path) or _is_test_file(path):
            continue
        source_files.append(rel)
    return source_files


def _detect_license(repo: Path, fallback: str | None) -> str:
    if fallback:
        return fallback
    for pattern in ("LICENSE*", "COPYING*", "NOTICE*"):
        for path in sorted(repo.glob(pattern)):
            if path.is_file():
                return path.name
    return "public repository license file not detected"


def _clone_or_get_repo(
    spec: dict[str, Any],
    *,
    clone_root: Path,
    no_clone: bool,
) -> Path:
    local_path = spec.get("local_path")
    if local_path:
        repo = Path(str(local_path)).resolve()
        if not repo.exists():
            raise FileNotFoundError(f"local_path does not exist: {repo}")
        return repo
    if no_clone:
        raise ValueError("repo spec without local_path cannot be used with --no-clone")
    repo_url = str(spec.get("repo_url") or spec.get("url") or "").strip()
    repo_id = str(spec.get("repo_id") or "").strip()
    if not repo_url or not repo_id:
        raise ValueError("repo specs without local_path require repo_id and repo_url")
    clone_root.mkdir(parents=True, exist_ok=True)
    target = clone_root / repo_id
    if not target.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target)],
            check=True,
            timeout=600,
        )
    commit_hash = str(spec.get("commit_hash") or spec.get("commit") or "").strip()
    if commit_hash:
        try:
            subprocess.run(
                ["git", "cat-file", "-e", f"{commit_hash}^{{commit}}"],
                cwd=target,
                check=True,
                capture_output=True,
                timeout=60,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", commit_hash],
                cwd=target,
                check=True,
                timeout=600,
            )
        subprocess.run(
            ["git", "checkout", "--detach", commit_hash],
            cwd=target,
            check=True,
            timeout=120,
        )
    return target.resolve()


def _candidate_command(test: TestCase) -> str:
    return f"python -m pytest -q {test.rel_path}::{test.function_name}"


def _choose_source_file(
    *,
    source_files: list[str],
    test: TestCase,
    row_index: int,
) -> str:
    if not source_files:
        return test.rel_path
    import_tokens = set(test.imports)
    for source in source_files:
        path_tokens = set(_tokens(source))
        if import_tokens & path_tokens:
            return source
    return source_files[row_index % len(source_files)]


def _runtime_evidence_for_case(
    *,
    test: TestCase,
    source_file: str,
    evidence_density: str,
    continuation_depth: str,
    trace_type: str,
    active_watch_key: str | None = None,
    include_behavior_summary: bool = True,
) -> dict[str, Any]:
    summary_tokens = {
        "low": 8,
        "medium": 16,
        "high": 28,
    }[evidence_density]
    imported_symbols = list(test.imports[:6])
    evidence: dict[str, Any] = {
        "trace_construction": "read_only_static_public_repo_trace",
        "changed_files": [source_file],
        "traceback_symbols": imported_symbols or _tokens(source_file)[:4],
        "prior_read_summary": (
            "Read-only public repository trace: prior source inspection linked the "
            f"failure behavior to source module {source_file} without executing tests."
        ),
        "stale_state_refresh": continuation_depth == "stale_state_refresh",
    }
    if include_behavior_summary:
        evidence["test_body_behavior_summary"] = _summarize_test_body(
            test.body_text,
            test.function_name,
            max_tokens=summary_tokens,
        )
        evidence["test_body_function_name_redacted"] = True
    else:
        evidence["test_body_behavior_summary_redacted"] = True
    if active_watch_key:
        evidence["active_watch_key"] = active_watch_key
        evidence["active_watch_key_origin"] = "read_only_public_repo_static_test_relation"
    if evidence_density in {"medium", "high"}:
        evidence["watched_files"] = [test.rel_path]
        evidence["module_owner"] = ".".join(_tokens(source_file)[:3]) or "unknown_module"
    if evidence_density == "high":
        evidence["test_body_summary_token_budget"] = summary_tokens
    if trace_type == "changed_file_to_watched_test":
        evidence["watched_files"] = [test.rel_path]
    elif trace_type == "module_ownership_to_command":
        evidence["module_owner"] = ".".join(_tokens(source_file)[:4]) or "unknown_module"
    elif trace_type == "stale_state_refresh":
        evidence["stale_state_refresh"] = True
        evidence["refresh_reason"] = "previous watch state was stale before source inspection"
    return evidence


def _baseline_payload(row: dict[str, Any]) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    predictions = compute_phase2m_baseline_predictions(row)
    metadata = {
        name: {
            "measured": True,
            "method": method,
            "uses_expected_command": False,
            "uses_sealed_feedback": False,
        }
        for name, method in BASELINE_METHODS.items()
    }
    return predictions, metadata


def _visible_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_visible_text": row.get("current_visible_text"),
        "runtime_visible_evidence": row.get("runtime_visible_evidence"),
        "command_candidates": row.get("command_candidates"),
    }


def _test_watch_key(
    *,
    repo_id: str,
    commit_hash: str,
    test: TestCase,
) -> str:
    body_without_function = test.body_text.replace(test.function_name, "")
    payload = {
        "repo_id": repo_id,
        "commit_hash": commit_hash,
        "rel_path": test.rel_path,
        "function_name": test.function_name,
        "body_hash": _sha256_text(body_without_function)[:24],
        "imports": list(test.imports[:8]),
    }
    return f"watch:{_sha256_text(_canonical_json(payload))[:24]}"


def build_public_repo_rows_for_spec(
    spec: dict[str, Any],
    *,
    clone_root: Path,
    rows_per_repo: int,
    no_clone: bool = False,
    structured_watch_keys: bool = False,
    include_behavior_summary: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_id = str(spec.get("repo_id") or Path(str(spec.get("local_path") or "repo")).name)
    split = str(spec.get("split") or "train")
    if split not in {"train", "val", "holdout"}:
        raise ValueError(f"invalid split for {repo_id}: {split}")
    repo = _clone_or_get_repo(spec, clone_root=clone_root, no_clone=no_clone)
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
        source_file = _choose_source_file(
            source_files=source_files,
            test=expected,
            row_index=index,
        )
        runtime_visible_evidence = _runtime_evidence_for_case(
            test=expected,
            source_file=source_file,
            evidence_density=evidence_density,
            continuation_depth=continuation_depth,
            trace_type=trace_type,
            active_watch_key=watch_keys[expected_slot] if structured_watch_keys else None,
            include_behavior_summary=include_behavior_summary,
        )
        current_visible_text = (
            "Public repository read-only regression triage. Same-intent pytest rerun "
            "commands are available from one test module; choose using only runtime-visible "
            f"source, traceback, watched-file, and prior-read evidence. repo={repo_id}; "
            f"density={evidence_density}; ambiguity={ambiguity_class}."
        )
        row = {
            "trace_id": f"{split}:{repo_id}:{commit_hash[:12]}:{index}",
            "split": split,
            "source_kind": "public_repo",
            "repo_id": repo_id,
            "repo_url_or_origin": repo_url,
            "commit_hash": commit_hash,
            "license_or_synthetic_origin": license_name,
            "trace_construction_mode": "read_only_static_public_repo_trace",
            "current_visible_text": current_visible_text,
            "runtime_visible_evidence": runtime_visible_evidence,
            "command_candidates": [
                {
                    "command": command,
                    "intent": "test_rerun",
                    "source": "public_repo_static_pytest_target",
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
        if FORBIDDEN_VISIBLE_RE.search(_canonical_json(_visible_payload(row))):
            rejected_reasons.append("row_contains_forbidden_visible_marker")
        else:
            rows.append(row)
        attempts += 1
    if len(rows) < rows_per_repo:
        rejected_reasons.append(f"insufficient_rows:{len(rows)}<{rows_per_repo}")
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
        "include_behavior_summary": include_behavior_summary,
    }


def collect_phase2m_public_repo_traces(
    *,
    repo_specs: list[dict[str, Any]],
    clone_root: str | Path,
    output_root: str | Path,
    rows_per_repo: int = 12,
    no_clone: bool = False,
    structured_watch_keys: bool = False,
    include_behavior_summary: bool = True,
) -> dict[str, Any]:
    output = Path(output_root)
    script_hash = _script_hash()
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "holdout": []}
    repo_reports: list[dict[str, Any]] = []
    for spec in repo_specs:
        rows, report = build_public_repo_rows_for_spec(
            spec,
            clone_root=Path(clone_root),
            rows_per_repo=rows_per_repo,
            no_clone=no_clone,
            structured_watch_keys=structured_watch_keys,
            include_behavior_summary=include_behavior_summary,
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
    manifest = {
        "collector_family": "phase2m_public_repo_readonly_trace_collector",
        "trace_construction_mode": "read_only_static_public_repo_trace",
        "sealed_v3_used": False,
        "writes_to_collected_repos": False,
        "collection_script_hash": script_hash,
        "rows_per_repo": rows_per_repo,
        "structured_watch_keys": structured_watch_keys,
        "include_behavior_summary": include_behavior_summary,
        "splits": splits,
        "repos": repo_reports,
        "dimensions": {
            "evidence_densities": list(EVIDENCE_DENSITIES),
            "candidate_counts": list(CANDIDATE_COUNTS),
            "continuation_depths": list(CONTINUATION_DEPTHS),
            "ambiguity_classes": list(AMBIGUITY_CLASSES),
            "trace_types": list(TRACE_TYPES),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect read-only public repository Phase2M traces from pytest targets."
    )
    parser.add_argument("--repo-spec-json", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rows-per-repo", type=int, default=12)
    parser.add_argument("--no-clone", action="store_true")
    parser.add_argument("--structured-watch-keys", action="store_true")
    parser.add_argument("--suppress-test-body-summary", action="store_true")
    args = parser.parse_args()
    specs = _read_json(args.repo_spec_json)
    if not isinstance(specs, list):
        raise TypeError("--repo-spec-json must contain a JSON list of repo specs")
    manifest = collect_phase2m_public_repo_traces(
        repo_specs=specs,
        clone_root=args.clone_root,
        output_root=args.output_root,
        rows_per_repo=args.rows_per_repo,
        no_clone=args.no_clone,
        structured_watch_keys=args.structured_watch_keys,
        include_behavior_summary=not args.suppress_test_body_summary,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
