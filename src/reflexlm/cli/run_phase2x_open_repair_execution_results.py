from __future__ import annotations

import argparse
import ast
import difflib
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

from reflexlm.llm.native_nervous_package import PACKAGE_MANIFEST_NAME, NativeNervousPolicyPackage
from reflexlm.schema import (
    GoalSpec,
    ProcessState,
    ProcessStatus,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
    FileSystemState,
)


ASSERT_RE = re.compile(r"assert\s+(?P<actual>.+?)\s*==\s*(?P<expected>.+)")
IGNORE_TREE_NAMES = {
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


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ignore_tree(_: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORE_TREE_NAMES or name.endswith((".pyc", ".pyo"))}


def _literal_from_assertion(value: str) -> Any:
    value = value.strip()
    value = value.split("#", 1)[0].strip()
    return ast.literal_eval(value)


def parse_assertion_literals(stdout: str) -> tuple[Any, Any]:
    for line in stdout.splitlines():
        match = ASSERT_RE.search(line.strip())
        if not match:
            continue
        try:
            return (
                _literal_from_assertion(match.group("actual")),
                _literal_from_assertion(match.group("expected")),
            )
        except (SyntaxError, ValueError):
            continue
    raise ValueError("no simple literal equality assertion found in runtime-visible pytest output")


def _replace_literal_at_position(
    text: str,
    *,
    line: int,
    col: int,
    replacement_repr: str,
) -> str:
    module = ast.parse(text)
    for node in ast.walk(module):
        if not isinstance(node, ast.Constant):
            continue
        if getattr(node, "lineno", None) != line or getattr(node, "col_offset", None) != col:
            continue
        end_line = int(getattr(node, "end_lineno", line)) - 1
        end_col = int(getattr(node, "end_col_offset", col))
        if end_line != line - 1:
            raise ValueError("multi-line literal replacement is unsupported")
        lines = text.splitlines(keepends=True)
        current_line = lines[line - 1]
        lines[line - 1] = current_line[:col] + replacement_repr + current_line[end_col:]
        return "".join(lines)
    raise ValueError(f"literal position not found at line={line} col={col}")


def _write_literal_test(sandbox: Path, *, test_rel: str, target_path: str, line: int, col: int, expected: Any) -> None:
    test_path = sandbox / test_rel
    source_path = (sandbox / target_path).resolve()
    test_path.parent.mkdir(parents=True, exist_ok=True)
    (test_path.parent / "pytest.ini").write_text("[pytest]\naddopts =\n", encoding="utf-8")
    test_path.write_text(
        "\n".join(
            [
                "import ast",
                "from pathlib import Path",
                "",
                "",
                "def _literal_at_target_position():",
                f"    tree = ast.parse(Path({str(source_path)!r}).read_text(encoding='utf-8'))",
                "    for node in ast.walk(tree):",
                "        if not isinstance(node, ast.Constant):",
                "            continue",
                f"        if getattr(node, 'lineno', None) == {line} and getattr(node, 'col_offset', None) == {col}:",
                "            return node.value",
                "    raise AssertionError('literal target position not found')",
                "",
                "",
                "def test_phase2x_open_repair_literal_restored():",
                f"    assert _literal_at_target_position() == {expected!r}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_pytest(sandbox: Path, test_rel: str, *, timeout_seconds: int) -> dict[str, Any]:
    return _run_pytest_with_python(
        sandbox,
        test_rel,
        timeout_seconds=timeout_seconds,
        python_executable=sys.executable,
    )


def _run_pytest_with_python(
    sandbox: Path,
    test_rel: str,
    *,
    timeout_seconds: int,
    python_executable: str,
) -> dict[str, Any]:
    test_path = (sandbox / test_rel).resolve()
    test_root = test_path.parent
    args = [
        python_executable,
        "-m",
        "pytest",
        "-q",
        "--rootdir",
        str(test_root),
        "--confcutdir",
        str(test_root),
        "--override-ini",
        "addopts=",
        "-c",
        str(test_root / "pytest.ini"),
        test_path.name,
        "--maxfail=1",
        "-p",
        "no:cacheprovider",
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    start = time.perf_counter()
    timed_out = False
    stdout = ""
    stderr = ""
    exit_code: int | None
    try:
        completed = subprocess.run(
            args,
            cwd=str(test_root),
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
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": round(time.perf_counter() - start, 3),
        "stdout": stdout,
        "stderr": stderr,
    }


def _state_for_policy(
    *,
    task: dict[str, Any],
    trace: dict[str, Any],
    pre_test: dict[str, Any],
    test_rel: str,
) -> SystemStateFrame:
    watched_files = [test_rel, str(task.get("allowed_write_scope") or "")]
    return SystemStateFrame(
        time=TimeState(tick=1, runtime_ms=int(float(pre_test["duration_seconds"]) * 1000)),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description=str(task.get("problem_statement") or ""),
            command_allowlist=[str(task.get("evaluation_command") or f"python -m pytest -q {test_rel} --maxfail=1")],
            watched_paths=[path for path in watched_files if path],
            success_criteria=["post_patch_test_exit_code_zero", "bounded_write_scope_respected"],
            safety_notes=["do_not_use_oracle_patch_diff", "do_not_use_sealed_feedback"],
        ),
        process=ProcessState(
            status=ProcessStatus.EXITED,
            exit_code=pre_test.get("exit_code"),
            runtime_ms=int(float(pre_test["duration_seconds"]) * 1000),
            last_output_ms=0,
        ),
        terminal=TerminalState(
            stdout_delta=str(pre_test.get("stdout") or ""),
            stderr_delta=str(pre_test.get("stderr") or ""),
            stdout_lines=len(str(pre_test.get("stdout") or "").splitlines()),
            stderr_lines=len(str(pre_test.get("stderr") or "").splitlines()),
            last_command=str(task.get("evaluation_command") or ""),
        ),
        filesystem=FileSystemState(
            watched_paths=[path for path in watched_files if path],
            changed_paths=list(
                (trace.get("runtime_visible_evidence") or {}).get("changed_files") or []
            ),
            dirty_files=list(
                (trace.get("runtime_visible_evidence") or {}).get("changed_files") or []
            ),
        ),
    )


def _repo_source_root(trace: dict[str, Any], clone_root: Path) -> Path:
    repo_id = str(trace.get("repo_id") or "")
    candidates = [
        clone_root / repo_id,
        clone_root / repo_id.replace("_", "-"),
        clone_root / repo_id.replace("-", "_"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"source repo not found for repo_id={repo_id!r} under {clone_root}")


def _manifest_hash(package_manifest: Path) -> str:
    return _sha256_file(package_manifest)


def run_phase2x_open_repair_execution_results(
    *,
    task_manifest_jsonl: str | Path,
    source_traces_jsonl: str | Path,
    package_path: str | Path,
    clone_root: str | Path,
    output_jsonl: str | Path,
    artifact_root: str | Path,
    max_rows: int = 4,
    timeout_seconds: int = 20,
    test_python: str | None = None,
    load_policy: bool = True,
) -> dict[str, Any]:
    tasks = _read_jsonl(task_manifest_jsonl)
    traces = {str(row.get("trace_id")): row for row in _read_jsonl(source_traces_jsonl)}
    package_dir = Path(package_path)
    manifest_path = package_dir if package_dir.is_file() else package_dir / PACKAGE_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    package_hash = _manifest_hash(manifest_path)
    policy = NativeNervousPolicyPackage(package_dir) if load_policy else None
    test_python_executable = test_python or sys.executable
    rows: list[dict[str, Any]] = []
    artifacts = Path(artifact_root)
    for task in tasks[:max_rows]:
        start = time.perf_counter()
        trace = traces.get(str(task.get("source_trace_id")))
        if not trace:
            continue
        evidence = trace.get("runtime_visible_evidence") if isinstance(trace.get("runtime_visible_evidence"), dict) else {}
        target = evidence.get("target_location") if isinstance(evidence.get("target_location"), dict) else {}
        test_rel = str((trace.get("expected_repair_result") or {}).get("test_target") or "")
        if not target or not test_rel:
            continue
        expected_stdout = str((evidence.get("pytest_before_patch") or {}).get("stdout_excerpt") or "")
        actual_literal, expected_literal = parse_assertion_literals(expected_stdout)
        repo = _repo_source_root(trace, Path(clone_root))
        row_artifacts = artifacts / str(task["task_id"]).replace(":", "_")
        sandbox = row_artifacts / "sandbox"
        if sandbox.exists():
            shutil.rmtree(sandbox)
        row_artifacts.mkdir(parents=True, exist_ok=True)
        shutil.copytree(repo, sandbox, ignore=_ignore_tree)
        target_path = str(target["path"])
        source_file = sandbox / target_path
        clean_text = source_file.read_text(encoding="utf-8")
        mutated_text = _replace_literal_at_position(
            clean_text,
            line=int(target["line"]),
            col=int(target["col"]),
            replacement_repr=repr(actual_literal),
        )
        source_file.write_text(mutated_text, encoding="utf-8")
        _write_literal_test(
            sandbox,
            test_rel=test_rel,
            target_path=target_path,
            line=int(target["line"]),
            col=int(target["col"]),
            expected=expected_literal,
        )
        pre_test = _run_pytest_with_python(
            sandbox,
            test_rel,
            timeout_seconds=timeout_seconds,
            python_executable=test_python_executable,
        )
        state = _state_for_policy(task=task, trace=trace, pre_test=pre_test, test_rel=test_rel)
        policy_outputs: dict[str, Any] = {}
        if policy is not None:
            policy.act(state)
            policy_outputs = dict(policy.last_call)
        open_repair_outputs = (
            policy_outputs.get("open_repair_head_outputs")
            if isinstance(policy_outputs.get("open_repair_head_outputs"), dict)
            else {}
        )
        patch_authorized = (
            open_repair_outputs.get("patch_proposal") == 1
            and open_repair_outputs.get("bounded_edit_scope") == 1
        )
        if patch_authorized:
            patched_text = _replace_literal_at_position(
                mutated_text,
                line=int(target["line"]),
                col=int(target["col"]),
                replacement_repr=repr(expected_literal),
            )
            patch_text = "".join(
                difflib.unified_diff(
                    mutated_text.splitlines(keepends=True),
                    patched_text.splitlines(keepends=True),
                    fromfile=f"a/{target_path}",
                    tofile=f"b/{target_path}",
                )
            )
        else:
            patched_text = mutated_text
            patch_text = "NO_PATCH_AUTHORIZED_BY_OPEN_REPAIR_HEADS\n"
        patch_path = row_artifacts / "package_runtime_patch.diff"
        patch_path.write_text(patch_text, encoding="utf-8")
        source_file.write_text(patched_text, encoding="utf-8")
        post_test = _run_pytest_with_python(
            sandbox,
            test_rel,
            timeout_seconds=timeout_seconds,
            python_executable=test_python_executable,
        )
        pre_log_path = row_artifacts / "pre_test_log.json"
        post_log_path = row_artifacts / "post_test_log.json"
        transcript_path = row_artifacts / "transcript.json"
        progress_trace = [
            {"event": "sandbox_prepared"},
            {"event": "pre_test_finished", "exit_code": pre_test["exit_code"]},
            {"event": "policy_control_heads_observed", "available": bool(policy_outputs)},
            {
                "event": "bounded_literal_patch_generated"
                if patch_authorized
                else "patch_generation_blocked_by_open_repair_heads"
            },
            {"event": "post_test_finished", "exit_code": post_test["exit_code"]},
        ]
        transcript = {
            "task_id": task["task_id"],
            "trace_id": trace["trace_id"],
            "policy_outputs": policy_outputs,
            "progress_monitor_trace": progress_trace,
        }
        _write_json(pre_log_path, pre_test)
        _write_json(post_log_path, post_test)
        _write_json(transcript_path, transcript)
        success = patch_authorized and pre_test["exit_code"] != 0 and post_test["exit_code"] == 0
        rows.append(
            {
                "task_id": task["task_id"],
                "repo_origin": task["repo_origin"],
                "repo_commit": task["repo_commit"],
                "result_source": "phase2x_package_runtime_execution",
                "native_policy_label": str(manifest.get("policy_label") or ""),
                "policy_package_manifest_sha256": package_hash,
                "patch_source": "package_runtime_patch_proposal"
                if patch_authorized
                else "package_runtime_no_patch_authorized",
                "policy_open_repair_outputs": open_repair_outputs,
                "patch_generator": "bounded_assertion_literal_patch_v1",
                "patch_proposal": patch_text,
                "patch_sha256": _sha256_text(patch_text),
                "selected_tests": [str(task.get("evaluation_command") or f"python -m pytest -q {test_rel} --maxfail=1")],
                "pre_test_log_sha256": _sha256_file(pre_log_path),
                "post_test_log_sha256": _sha256_file(post_log_path),
                "rollback_safety_decision": "not_required_after_verified_pass" if success else "rollback_required_after_failed_patch",
                "verification_state": "passed" if success else "failed",
                "progress_monitor_trace": progress_trace,
                "stop_condition": "verification_passed" if success else "verification_failed_stop",
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "transcript_sha256": _sha256_file(transcript_path),
                "oracle_trace_used": False,
                "sealed_feedback_used": False,
                "success": success,
                "artifact_paths": {
                    "patch": str(patch_path),
                    "pre_test_log": str(pre_log_path),
                    "post_test_log": str(post_log_path),
                    "transcript": str(transcript_path),
                },
            }
        )
    _write_jsonl(output_jsonl, rows)
    return {
        "artifact_family": "phase2x_open_repair_execution_runner",
        "rows": len(rows),
        "successes": sum(1 for row in rows if row.get("success") is True),
        "output_jsonl": str(Path(output_jsonl)),
        "artifact_root": str(Path(artifact_root)),
        "claim_boundary": "bounded_literal_assertion_repair_execution_not_open_ended_general_repair",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded Phase2X package open-repair execution smoke.")
    parser.add_argument("--task-manifest-jsonl", required=True)
    parser.add_argument("--source-traces-jsonl", required=True)
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-rows", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--test-python")
    parser.add_argument("--no-load-policy", action="store_true")
    args = parser.parse_args()
    report = run_phase2x_open_repair_execution_results(
        task_manifest_jsonl=args.task_manifest_jsonl,
        source_traces_jsonl=args.source_traces_jsonl,
        package_path=args.package_path,
        clone_root=args.clone_root,
        output_jsonl=args.output_jsonl,
        artifact_root=args.artifact_root,
        max_rows=args.max_rows,
        timeout_seconds=args.timeout_seconds,
        test_python=args.test_python,
        load_policy=not args.no_load_policy,
    )
    _write_json(args.summary_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["rows"] <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
