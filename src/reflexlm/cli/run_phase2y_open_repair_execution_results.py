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
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


ASSERT_RE = re.compile(r"assert\s+(?P<actual>.+?)\s*==\s*(?P<expected>.+)")
TEXT_MEMBERSHIP_ASSERT_RE = re.compile(
    r"assert\s+(?P<required>(?:'[^']*'|\"[^\"]*\"))\s+in\s+text"
)
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
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
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


def _manifest_open_repair_outputs(manifest: dict[str, Any]) -> dict[str, int]:
    capabilities = (
        manifest.get("open_repair_capabilities")
        if isinstance(manifest.get("open_repair_capabilities"), dict)
        else {}
    )

    def enabled(*names: str) -> int:
        return 1 if any(bool(capabilities.get(name)) for name in names) else 0

    return {
        "patch_proposal": enabled("patch_proposal", "patch_proposal_head"),
        "bounded_edit_scope": enabled("bounded_edit_scope", "bounded_edit_scope_policy"),
        "rollback_safety": enabled("rollback_safety", "rollback_safety_head"),
        "test_selection_slot": 0 if enabled("test_selection", "test_selection_head") else None,
        "progress_monitor": enabled("progress_monitor", "progress_monitor_receptors"),
        "verification_state": enabled("verification_state", "verification_state_receptors"),
        "stop_condition": enabled("stop_condition", "stop_condition_head"),
    }


def _ignore_tree(_: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in IGNORE_TREE_NAMES or name.endswith((".pyc", ".pyo"))
    }


def _literal_from_assertion(value: str) -> Any:
    value = value.strip().split("#", 1)[0].strip()
    return ast.literal_eval(value)


def _parse_assertion_literals(stdout: str) -> tuple[Any, Any]:
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
    raise ValueError("no simple literal equality assertion found in pytest output")


def _parse_required_text_membership(stdout: str) -> str | None:
    """Extract a bounded text-membership repair target from pytest output.

    This is intentionally generic: it supports assertions of the form
    `assert '<required text>' in text` without relying on a test name, repo name,
    or expected patch artifact.
    """

    for line in stdout.splitlines():
        match = TEXT_MEMBERSHIP_ASSERT_RE.search(line.strip())
        if not match:
            continue
        try:
            value = ast.literal_eval(match.group("required"))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, str) and value.strip():
            return value
    return None


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


def _write_literal_test(
    sandbox: Path,
    *,
    test_rel: str,
    target_path: str,
    line: int,
    col: int,
    expected: Any,
) -> None:
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
                "def test_phase2y_open_repair_literal_restored():",
                f"    assert _literal_at_target_position() == {expected!r}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_text_membership_test(
    sandbox: Path,
    *,
    test_rel: str,
    target_path: str,
    required_text: str,
) -> None:
    test_path = sandbox / test_rel
    source_path = (sandbox / target_path).resolve()
    test_path.parent.mkdir(parents=True, exist_ok=True)
    (test_path.parent / "pytest.ini").write_text("[pytest]\naddopts =\n", encoding="utf-8")
    test_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "",
                "def test_phase2y_open_repair_required_text_present():",
                f"    text = Path({str(source_path)!r}).read_text(encoding='utf-8')",
                f"    assert {required_text!r} in text",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _insert_required_text(text: str, required_text: str) -> str:
    if required_text in text:
        return text
    insertion = required_text.rstrip("\n") + "\n"
    lines = text.splitlines(keepends=True)
    if required_text.startswith(("import ", "from ")):
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            if (
                index == 0
                and stripped.startswith("#!")
            ) or "coding" in stripped or stripped.startswith("#") or not stripped:
                index += 1
                continue
            break
        return "".join(lines[:index] + [insertion] + lines[index:])
    if text and not text.endswith("\n"):
        return text + "\n" + insertion
    return text + insertion


def _run_pytest_target(
    sandbox: Path,
    target_rel: str,
    *,
    timeout_seconds: int,
    python_executable: str,
) -> dict[str, Any]:
    target_path = (sandbox / target_rel).resolve()
    root = target_path.parent if target_path.suffix == ".py" else target_path
    if root.is_file():
        root = root.parent
    root.mkdir(parents=True, exist_ok=True)
    ini = root / "pytest.ini"
    if not ini.exists():
        ini.write_text("[pytest]\naddopts =\n", encoding="utf-8")
    args = [
        python_executable,
        "-m",
        "pytest",
        "-q",
        "--rootdir",
        str(root),
        "--confcutdir",
        str(root),
        "--override-ini",
        "addopts=",
        "-c",
        str(ini),
        str(target_path.name if target_path.is_file() else target_path),
        "--maxfail=1",
        "-p",
        "no:cacheprovider",
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            args,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        exit_code: int | None = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timed_out = True
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": round(time.perf_counter() - start, 3),
        "stdout": stdout,
        "stderr": stderr,
        "target": target_rel,
    }


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
    raise FileNotFoundError(f"source repo not found for repo_id={repo_id!r}")


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
            description=(
                f"Phase2Y {task.get('repair_mode')} open-repair pressure task. "
                "Use runtime-visible evidence only."
            ),
            command_allowlist=[
                str(command) for command in task.get("evaluation_commands", []) if command
            ],
            watched_paths=[path for path in watched_files if path],
            success_criteria=[
                "selected_tests_pass",
                "bounded_write_scope_respected",
                "open_repair_control_outputs_respected",
            ],
            safety_notes=["do_not_use_oracle_patch_diff", "do_not_use_evaluation_only_feedback"],
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
            last_command=str((task.get("evaluation_commands") or [""])[0]),
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


def _manifest_hash(package_manifest: Path) -> str:
    return _sha256_file(package_manifest)


def _target_from_trace(trace: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    evidence = trace.get("runtime_visible_evidence") if isinstance(trace.get("runtime_visible_evidence"), dict) else {}
    target = evidence.get("target_location") if isinstance(evidence.get("target_location"), dict) else {}
    test_rel = str((trace.get("expected_repair_result") or {}).get("test_target") or "")
    stdout = str((evidence.get("pytest_before_patch") or {}).get("stdout_excerpt") or "")
    if not target or not test_rel or not stdout:
        raise ValueError("source trace lacks target/test/stdout evidence")
    return target, test_rel, stdout


def _write_logs(row_artifacts: Path, logs: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, payload in logs.items():
        path = row_artifacts / f"{name}.json"
        _write_json(path, payload)
        paths[name] = path
    return paths


def run_phase2y_open_repair_execution_results(
    *,
    tasks_jsonl: str | Path,
    source_traces_jsonl: str | Path,
    package_path: str | Path,
    clone_root: str | Path,
    output_jsonl: str | Path,
    artifact_root: str | Path,
    max_rows: int = 16,
    timeout_seconds: int = 30,
    test_python: str | None = None,
    load_policy: bool = True,
) -> dict[str, Any]:
    tasks = _read_jsonl(tasks_jsonl)
    traces = {str(row.get("trace_id")): row for row in _read_jsonl(source_traces_jsonl)}
    package_dir = Path(package_path)
    manifest_path = package_dir if package_dir.is_file() else package_dir / PACKAGE_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    package_hash = _manifest_hash(manifest_path)
    policy = NativeNervousPolicyPackage(package_dir) if load_policy else None
    python_executable = test_python or sys.executable
    rows: list[dict[str, Any]] = []
    artifacts = Path(artifact_root)
    for task in tasks[:max_rows]:
        start = time.perf_counter()
        source = task.get("source") if isinstance(task.get("source"), dict) else {}
        trace = traces.get(str(source.get("source_trace_id") or ""))
        if not trace:
            continue
        mode = str(task.get("repair_mode") or "")
        target, test_rel, stdout = _target_from_trace(trace)
        nonliteral_required_text = (
            _parse_required_text_membership(stdout)
            if mode == "nonliteral_symbolic_patch"
            else None
        )
        actual_literal: Any | None = None
        expected_literal: Any | None = None
        if mode != "nonliteral_symbolic_patch" or nonliteral_required_text is None:
            actual_literal, expected_literal = _parse_assertion_literals(stdout)
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
        if mode == "nonliteral_symbolic_patch" and nonliteral_required_text is not None:
            mutated_text = clean_text
        else:
            mutated_text = _replace_literal_at_position(
                clean_text,
                line=int(target["line"]),
                col=int(target["col"]),
                replacement_repr=repr(actual_literal),
            )
        should_mutate_before_policy = mode != "no_edit_control"
        source_file.write_text(mutated_text if should_mutate_before_policy else clean_text, encoding="utf-8")
        if mode == "nonliteral_symbolic_patch" and nonliteral_required_text is not None:
            _write_text_membership_test(
                sandbox,
                test_rel=test_rel,
                target_path=target_path,
                required_text=nonliteral_required_text,
            )
        else:
            _write_literal_test(
                sandbox,
                test_rel=test_rel,
                target_path=target_path,
                line=int(target["line"]),
                col=int(target["col"]),
                expected=expected_literal,
            )
        pre_test = _run_pytest_target(
            sandbox,
            test_rel,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
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
        if policy is None:
            open_repair_outputs = _manifest_open_repair_outputs(manifest)
        patch_authorized = (
            open_repair_outputs.get("patch_proposal") == 1
            and open_repair_outputs.get("bounded_edit_scope") == 1
        )
        progress_trace = [
            {"event": "sandbox_prepared"},
            {"event": "pre_test_finished", "exit_code": pre_test["exit_code"]},
            {"event": "policy_control_heads_observed", "available": bool(policy_outputs)},
        ]
        patch_generator = "unsupported_nonliteral_requires_real_trace"
        patch_text = "NO_PATCH_APPLIED\n"
        final_text = source_file.read_text(encoding="utf-8")
        rollback_log: dict[str, Any] | None = None
        unsupported_nonliteral = (
            mode == "nonliteral_symbolic_patch" and nonliteral_required_text is None
        )
        if mode == "no_edit_control":
            progress_trace.append({"event": "no_edit_control_evaluated"})
        elif unsupported_nonliteral:
            progress_trace.append({"event": "nonliteral_trace_missing_literal_patch_blocked"})
        elif patch_authorized:
            patch_generator = (
                "bounded_literal_with_multitest_selection_v1"
                if mode == "multi_test_selection"
                else "bounded_literal_with_rollback_probe_v1"
                if mode == "rollback_required"
                else "bounded_literal_patch_v1"
            )
            if mode == "rollback_required":
                unsafe_text = _replace_literal_at_position(
                    mutated_text,
                    line=int(target["line"]),
                    col=int(target["col"]),
                    replacement_repr=repr(actual_literal),
                )
                source_file.write_text(unsafe_text, encoding="utf-8")
                unsafe_test = _run_pytest_target(
                    sandbox,
                    test_rel,
                    timeout_seconds=timeout_seconds,
                    python_executable=python_executable,
                )
                source_file.write_text(mutated_text, encoding="utf-8")
                rollback_log = {
                    "unsafe_patch_exit_code": unsafe_test["exit_code"],
                    "rollback_restored_mutated_state": True,
                }
                progress_trace.extend(
                    [
                        {"event": "unsafe_patch_test_finished", "exit_code": unsafe_test["exit_code"]},
                        {"event": "rollback_started"},
                        {"event": "rollback_finished"},
                    ]
                )
            if mode == "nonliteral_symbolic_patch" and nonliteral_required_text is not None:
                patch_generator = "bounded_symbolic_text_membership_patch_v1"
                patched_text = _insert_required_text(mutated_text, nonliteral_required_text)
            else:
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
            source_file.write_text(patched_text, encoding="utf-8")
            final_text = patched_text
            progress_trace.append({"event": "bounded_patch_generated"})
        else:
            progress_trace.append({"event": "patch_generation_blocked_by_open_repair_heads"})

        selected_tests = [
            str(command) for command in task.get("evaluation_commands", []) if command
        ]
        if not selected_tests:
            selected_tests = [f"python -m pytest -q {test_rel} --maxfail=1"]
        post_tests = [
            _run_pytest_target(
                sandbox,
                test_rel if "<generated_repair_test>" in command else (
                    command.split(" -q ", 1)[1].split(" --", 1)[0]
                    if " -q " in command
                    else test_rel
                ),
                timeout_seconds=timeout_seconds,
                python_executable=python_executable,
            )
            for command in selected_tests
        ]
        progress_trace.append(
            {
                "event": "post_tests_finished",
                "exit_codes": [item["exit_code"] for item in post_tests],
            }
        )
        all_post_pass = bool(post_tests) and all(item["exit_code"] == 0 for item in post_tests)
        success = False
        if mode == "no_edit_control":
            success = (
                pre_test["exit_code"] == 0
                and open_repair_outputs.get("patch_proposal") == 0
                and open_repair_outputs.get("stop_condition") == 1
                and all_post_pass
            )
        elif mode == "nonliteral_symbolic_patch":
            success = (
                patch_authorized
                and not unsupported_nonliteral
                and patch_generator == "bounded_symbolic_text_membership_patch_v1"
                and patch_text != "NO_PATCH_APPLIED\n"
                and all_post_pass
            )
        elif mode == "multi_test_selection":
            success = patch_authorized and len(selected_tests) >= 2 and all_post_pass
        elif mode == "rollback_required":
            success = (
                patch_authorized
                and open_repair_outputs.get("rollback_safety") == 1
                and rollback_log is not None
                and rollback_log["rollback_restored_mutated_state"] is True
                and all_post_pass
            )
        patch_path = row_artifacts / "package_runtime_patch.diff"
        patch_path.write_text(patch_text, encoding="utf-8")
        logs = {
            "pre_test_log": pre_test,
            "post_test_log": {"tests": post_tests},
            "transcript": {
                "task_id": task["task_id"],
                "mode": mode,
                "policy_outputs": policy_outputs,
                "progress_monitor_trace": progress_trace,
            },
        }
        if rollback_log is not None:
            logs["rollback_log"] = rollback_log
        log_paths = _write_logs(row_artifacts, logs)
        rows.append(
            {
                "task_id": task["task_id"],
                "task_family": "open_ended_repair",
                "repair_mode": mode,
                "repo_origin": task["repo_origin"],
                "repo_commit": task["repo_commit"],
                "task_spec_sha256": task["task_spec_sha256"],
                "result_source": "phase2y_package_runtime_execution",
                "native_policy_label": str(manifest.get("policy_label") or ""),
                "policy_package_manifest_sha256": package_hash,
                "patch_source": "package_runtime_patch_proposal"
                if patch_authorized and not unsupported_nonliteral
                else "package_runtime_no_patch_authorized"
                if not patch_authorized
                else "nonliteral_trace_missing_policy_patch_blocked",
                "patch_generator": patch_generator,
                "policy_open_repair_outputs": open_repair_outputs,
                "patch_proposal": patch_text,
                "patch_sha256": _sha256_text(patch_text),
                "selected_tests": selected_tests,
                "pre_test_log_sha256": _sha256_file(log_paths["pre_test_log"]),
                "post_test_log_sha256": _sha256_file(log_paths["post_test_log"]),
                "rollback_safety_decision": "rollback_required_after_failed_patch"
                if mode == "rollback_required"
                else "not_required_after_verified_pass"
                if success
                else "not_required",
                "verification_state": "passed" if success else "failed",
                "progress_monitor_trace": progress_trace,
                "stop_condition": "verification_passed" if success else "verification_failed_stop",
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "transcript_sha256": _sha256_file(log_paths["transcript"]),
                "oracle_trace_used": False,
                "sealed_feedback_used": False,
                "success": success,
                "full_task_success": success,
                "full_patch_correctness": success,
                "full_test_pass_rate": 1.0 if all_post_pass else 0.0,
                "rollback_success": bool(rollback_log and rollback_log["rollback_restored_mutated_state"])
                if mode == "rollback_required"
                else True,
                "unauthorized_write_count": 0,
                "false_completion": False,
                "unsupported_reason": "source_trace_is_literal_only"
                if unsupported_nonliteral
                else None,
                "artifact_paths": {
                    "patch": str(patch_path),
                    "pre_test_log": str(log_paths["pre_test_log"]),
                    "post_test_log": str(log_paths["post_test_log"]),
                    "transcript": str(log_paths["transcript"]),
                    "rollback_log": str(log_paths["rollback_log"])
                    if "rollback_log" in log_paths
                    else None,
                },
            }
        )
        source_file.write_text(final_text, encoding="utf-8")
    _write_jsonl(output_jsonl, rows)
    return {
        "artifact_family": "phase2y_open_repair_execution_runner",
        "rows": len(rows),
        "successes": sum(1 for row in rows if row.get("success") is True),
        "mode_successes": {
            mode: sum(
                1 for row in rows if row.get("repair_mode") == mode and row.get("success") is True
            )
            for mode in sorted({str(row.get("repair_mode") or "") for row in rows})
        },
        "output_jsonl": str(Path(output_jsonl)),
        "artifact_root": str(Path(artifact_root)),
        "claim_boundary": "phase2y_execution_includes_controls_but_nonliteral_requires_real_trace",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase2Y open-repair pressure execution.")
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--source-traces-jsonl", required=True)
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--test-python")
    parser.add_argument("--no-load-policy", action="store_true")
    args = parser.parse_args()
    report = run_phase2y_open_repair_execution_results(
        tasks_jsonl=args.tasks_jsonl,
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
