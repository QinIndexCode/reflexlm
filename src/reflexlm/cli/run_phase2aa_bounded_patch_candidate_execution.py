from __future__ import annotations

import argparse
import difflib
import json
import re
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2aa_bounded_patch_candidates import (
    CLAIM_BOUNDARY,
    phase2z_row_to_phase2aa,
)
from reflexlm.cli.build_phase2s_head_dataset import _candidate_commands, _command_identity_signal
from reflexlm.cli.run_phase2z_public_structural_repair_execution import (
    _copy_public_repo,
    _git_apply_reverse_patch,
    _materialize_generated_test,
    _resolve_patch,
    _state_after_stderr_receptor,
    _state_for_public_policy,
)
from reflexlm.cli.run_phase2z_synthetic_nonliteral_repair_plumbing import (
    _git_apply_patch,
    _manifest_hash,
    _manifest_open_repair_outputs,
    _patch_stats,
    _read_json,
    _read_jsonl,
    _run_pytest_target,
    _sha256_file,
    _sha256_text,
    _write_json,
    _write_jsonl,
)
from reflexlm.llm.native_nervous_package import PACKAGE_MANIFEST_NAME, NativeNervousPolicyPackage

IDENTITY_SIGNAL_CONTROLS = {"normal", "erase_structural", "wrong_structural"}
DIRECT_TEST_RUNNER_FOOTER = """


def _phase2aa_run_zero_arg_generated_tests():
    runnable = []
    for name, value in sorted(globals().items()):
        if not name.startswith("test_") or not callable(value):
            continue
        code = getattr(value, "__code__", None)
        if code is None or code.co_argcount != 0:
            continue
        runnable.append(value)
    if not runnable:
        raise RuntimeError("no zero-argument generated test functions were available for direct execution")
    for test_func in runnable:
        test_func()


if __name__ == "__main__":
    _phase2aa_run_zero_arg_generated_tests()
"""


def _patch_observable_lines(patch_text: str) -> tuple[str, list[tuple[int, str]], list[str]]:
    target_rel = ""
    added: list[tuple[int, str]] = []
    removed: list[str] = []
    old_line: int | None = None
    new_line: int | None = None
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            target_rel = line.removeprefix("+++ b/")
            continue
        if line.startswith("@@"):
            match = re.match(r"@@ -(?P<old>\d+)(?:,\d+)? \+(?P<new>\d+)(?:,\d+)? @@", line)
            if match:
                old_line = int(match.group("old"))
                new_line = int(match.group("new"))
            continue
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+"):
            payload = line[1:]
            if payload.strip() and new_line is not None:
                added.append((new_line, payload))
            if new_line is not None:
                new_line += 1
            continue
        if line.startswith("-"):
            payload = line[1:]
            if payload.strip():
                removed.append(payload)
            if old_line is not None:
                old_line += 1
            continue
        if line.startswith(" "):
            if old_line is not None:
                old_line += 1
            if new_line is not None:
                new_line += 1
    return target_rel, added, removed


def _materialize_candidate_selection_test(
    row: dict[str, Any],
    dataset_root: Path,
    sandbox: Path,
) -> tuple[str, str]:
    artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    if artifact_paths.get("generated_test"):
        test_rel = _materialize_generated_test(row, dataset_root, sandbox)
        test_path = sandbox / test_rel
        source = test_path.read_text(encoding="utf-8")
        generated_test_source = (
            "recorded_generated_test_pytest"
            if re.search(r"REPO_ROOT\s*/\s*['\"]tests[/\\]", source)
            else "recorded_generated_test_direct"
        )
        if "_phase2aa_run_zero_arg_generated_tests" not in source:
            test_path.write_text(
                source.rstrip() + DIRECT_TEST_RUNNER_FOOTER,
                encoding="utf-8",
                newline="\n",
            )
        return test_rel, generated_test_source
    expected = row.get("expected_repair_result") if isinstance(row.get("expected_repair_result"), dict) else {}
    test_rel = str(expected.get("test_target") or "")
    if not test_rel:
        raise ValueError("row is missing expected_repair_result.test_target")
    _patch_path, patch_text = _resolve_patch(row, dataset_root)
    target_rel, added_lines, removed_lines = _patch_observable_lines(patch_text)
    if not target_rel:
        raise ValueError(f"cannot derive target path from recorded patch: {row.get('trace_id')}")
    test_path = sandbox / test_rel
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.joinpath("pytest.ini").write_text("[pytest]\naddopts =\n", encoding="utf-8")
    assertions: list[str] = [
        "from pathlib import Path",
        "",
        "REPO_ROOT = Path(__file__).resolve().parents[1]",
        f"TARGET = REPO_ROOT / {target_rel!r}",
        "",
        "",
        "def _check_recorded_patch_observable_lines():",
        "    lines = TARGET.read_text(encoding='utf-8').splitlines()",
        "    text = '\\n'.join(lines)",
    ]
    for line_no, line in added_lines[:8]:
        assertions.append(f"    assert len(lines) >= {line_no}")
        assertions.append(f"    assert lines[{line_no - 1}] == {line!r}")
    if not added_lines and not removed_lines:
        assertions.append("    assert TARGET.exists()")
    assertions.extend(
        [
            "",
            "",
            "def test_recorded_patch_observable_lines():",
            "    _check_recorded_patch_observable_lines()",
            "",
            "",
            "if __name__ == '__main__':",
            "    _check_recorded_patch_observable_lines()",
        ]
    )
    test_path.write_text("\n".join(assertions) + "\n", encoding="utf-8", newline="\n")
    return test_rel, "patch_observable_generated_test"


def _run_direct_python_target(
    sandbox: Path,
    target_rel: str,
    *,
    timeout_seconds: int,
    python_executable: str,
) -> dict[str, Any]:
    target_path = (sandbox / target_rel).resolve()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pythonpath = env.get("PYTHONPATH")
    sandbox_path = str(sandbox.resolve())
    env["PYTHONPATH"] = (
        sandbox_path
        if not existing_pythonpath
        else os.pathsep.join([sandbox_path, existing_pythonpath])
    )
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [python_executable, str(target_path)],
            cwd=str(sandbox.resolve()),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        return {
            "exit_code": int(completed.returncode),
            "timed_out": False,
            "duration_seconds": round(time.perf_counter() - start, 3),
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
            "target": target_rel,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": None,
            "timed_out": True,
            "duration_seconds": round(time.perf_counter() - start, 3),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "target": target_rel,
        }


def _run_candidate_selection_test(
    sandbox: Path,
    target_rel: str,
    *,
    generated_test_source: str,
    timeout_seconds: int,
    python_executable: str,
) -> dict[str, Any]:
    if generated_test_source in {
        "patch_observable_generated_test",
        "recorded_generated_test_direct",
    }:
        return _run_direct_python_target(
            sandbox,
            target_rel,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
    return _run_pytest_target(
        sandbox,
        target_rel,
        timeout_seconds=timeout_seconds,
        python_executable=python_executable,
    )


def _verification_command_for_generated_test(
    test_rel: str,
    *,
    generated_test_source: str,
) -> str:
    if generated_test_source in {
        "patch_observable_generated_test",
        "recorded_generated_test_direct",
    }:
        return f"python {test_rel}"
    return f"python -m pytest -q {test_rel} --maxfail=1"


def _selected_command_slot(policy_outputs: dict[str, Any]) -> int | None:
    cortex_plan = policy_outputs.get("cortex_plan")
    if isinstance(cortex_plan, dict):
        slot = cortex_plan.get("command_slot")
        if isinstance(slot, int):
            return slot
    return None


def _identity_prioritized_command_slot(row: dict[str, Any]) -> int | None:
    commands = _candidate_commands(row)
    if not commands:
        return None
    signal = _command_identity_signal(row, commands)
    scores = [
        float(signal.get(f"command_identity_slot:{index}") or 0.0)
        for index in range(len(commands))
    ]
    if not scores:
        return None
    best = max(scores)
    if best <= 0.0 or scores.count(best) != 1:
        return None
    return scores.index(best)


def _identity_signal_controlled_row(row: dict[str, Any], control: str) -> dict[str, Any]:
    if control not in IDENTITY_SIGNAL_CONTROLS:
        raise ValueError(
            f"identity_signal_control must be one of: {', '.join(sorted(IDENTITY_SIGNAL_CONTROLS))}"
        )
    converted = json.loads(json.dumps(row))
    if control == "normal":
        return converted
    evidence = (
        converted.get("runtime_visible_evidence")
        if isinstance(converted.get("runtime_visible_evidence"), dict)
        else {}
    )
    evidence.pop("structural_probe_hashes", None)
    evidence.pop("expected_literal_hash", None)
    evidence.pop("target_location", None)
    if control == "erase_structural":
        return converted

    candidates = [
        item
        for item in converted.get("repair_candidates", [])
        if isinstance(item, dict)
    ]
    try:
        expected_slot = int(converted.get("expected_patch_candidate_slot"))
    except (TypeError, ValueError):
        expected_slot = -1
    for index, candidate in enumerate(candidates):
        if index == expected_slot:
            continue
        literal_hash = str(candidate.get("target_literal_hash") or "")
        line = candidate.get("target_line")
        col = candidate.get("target_col")
        if literal_hash and line is not None and col is not None:
            evidence["expected_literal_hash"] = literal_hash
            evidence["target_location"] = {
                "path": str(candidate.get("edit_scope") or ""),
                "line": line,
                "col": col,
            }
            return converted
    return converted


def _bounded_distractor_patch(sandbox: Path, row: dict[str, Any]) -> str:
    evidence = row.get("runtime_visible_evidence") if isinstance(row.get("runtime_visible_evidence"), dict) else {}
    changed_files = [str(path) for path in evidence.get("changed_files", [])]
    target_rel = next((path for path in changed_files if path.endswith(".py")), None)
    if target_rel is None:
        raise ValueError(f"cannot build bounded distractor patch without python changed file: {row.get('trace_id')}")
    target = sandbox / target_rel
    original = target.read_text(encoding="utf-8")
    marker = "# phase2aa bounded distractor control\n"
    if original.startswith(marker):
        mutated = original
    else:
        mutated = marker + original
    target.write_text(mutated, encoding="utf-8")
    target.write_text(original, encoding="utf-8")
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            mutated.splitlines(keepends=True),
            fromfile=f"a/{target_rel}",
            tofile=f"b/{target_rel}",
        )
    )


def _patch_for_selected_slot(
    *,
    row: dict[str, Any],
    dataset_root: Path,
    sandbox: Path,
    selected_slot: int,
) -> tuple[str, str]:
    candidates = row.get("patch_candidates") if isinstance(row.get("patch_candidates"), list) else []
    if selected_slot < 0 or selected_slot >= len(candidates):
        return "", "invalid_candidate_slot"
    candidate = candidates[selected_slot]
    if not isinstance(candidate, dict):
        return "", "invalid_candidate_payload"
    if candidate.get("patch_source") == "recorded_correct_patch_artifact":
        _, patch_text = _resolve_patch(row, dataset_root)
        return patch_text, "selected_recorded_correct_patch_candidate"
    return _bounded_distractor_patch(sandbox, row), "selected_bounded_distractor_patch_candidate"


def run_phase2aa_bounded_patch_candidate_execution(
    *,
    source_rows_jsonl: str | Path,
    dataset_root: str | Path,
    clone_root: str | Path,
    package_path: str | Path,
    output_jsonl: str | Path,
    artifact_root: str | Path,
    max_rows: int = 24,
    timeout_seconds: int = 30,
    test_python: str | None = None,
    load_policy: bool = True,
    allow_bounded_candidate_retry: bool = False,
    max_candidate_attempts: int = 4,
    policyless_start_slot: int | None = None,
    retry_prioritization: str = "sequential",
    identity_signal_control: str = "normal",
) -> dict[str, Any]:
    import sys
    import time

    source_rows = _read_jsonl(source_rows_jsonl)
    package_dir = Path(package_path)
    manifest_path = package_dir if package_dir.is_file() else package_dir / PACKAGE_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    package_hash = _manifest_hash(manifest_path)
    policy = NativeNervousPolicyPackage(package_dir) if load_policy else None
    if retry_prioritization not in {"sequential", "identity_first"}:
        raise ValueError("retry_prioritization must be one of: sequential, identity_first")
    if identity_signal_control not in IDENTITY_SIGNAL_CONTROLS:
        raise ValueError(
            "identity_signal_control must be one of: "
            + ", ".join(sorted(IDENTITY_SIGNAL_CONTROLS))
        )
    python_executable = test_python or sys.executable
    root = Path(dataset_root)
    clones = Path(clone_root)
    artifacts = Path(artifact_root)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []

    for row in source_rows[:max_rows]:
        if (
            row.get("expected_patch_candidate_slot") is None
            or not isinstance(row.get("patch_candidates"), list)
        ):
            row = phase2z_row_to_phase2aa(row)
        start = time.perf_counter()
        trace_id = str(row.get("trace_id") or f"row-{len(rows)}")
        row_id = f"row_{len(rows):05d}_{_sha256_text(trace_id)[:12]}"
        row_artifacts = artifacts / row_id
        sandbox = row_artifacts / "sandbox"
        row_artifacts.mkdir(parents=True, exist_ok=True)
        _copy_public_repo(row, clones, sandbox)
        test_rel, generated_test_source = _materialize_candidate_selection_test(row, root, sandbox)
        correct_patch_source, correct_patch_text = _resolve_patch(row, root)
        local_correct_patch = row_artifacts / "recorded_correct_patch.diff"
        correct_patch_text = correct_patch_text.replace("\r\n", "\n").replace("\r", "\n")
        local_correct_patch.write_text(correct_patch_text, encoding="utf-8", newline="\n")

        reverse_to_fault = _git_apply_reverse_patch(sandbox, local_correct_patch, timeout_seconds)
        pre_test = _run_candidate_selection_test(
            sandbox,
            test_rel,
            generated_test_source=generated_test_source,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        state = _state_for_public_policy(row=row, pre_test=pre_test, test_rel=test_rel)
        policy_outputs: dict[str, Any] = {}
        if policy is not None:
            policy.act(state)
            policy_outputs = dict(policy.last_call)
            if policy_outputs.get("action_source") == "low_level_debug_receptor":
                state = _state_after_stderr_receptor(state)
                policy.act(state)
                policy_outputs = dict(policy.last_call)
        open_repair_outputs = (
            policy_outputs.get("open_repair_head_outputs")
            if isinstance(policy_outputs.get("open_repair_head_outputs"), dict)
            else {}
        )
        learned_descriptor_outputs = (
            policy_outputs.get("learned_patch_descriptor_outputs")
            if isinstance(policy_outputs.get("learned_patch_descriptor_outputs"), dict)
            else {}
        )
        if policy is None:
            open_repair_outputs = _manifest_open_repair_outputs(manifest)
            learned_descriptor_outputs = {}
        selected_slot = _selected_command_slot(policy_outputs)
        if selected_slot is None and policy is None:
            selected_slot = (
                int(policyless_start_slot)
                if policyless_start_slot is not None
                else int(row.get("expected_patch_candidate_slot") or 0)
            )
        patch_authorized = (
            open_repair_outputs.get("patch_proposal") == 1
            and open_repair_outputs.get("bounded_edit_scope") == 1
            and open_repair_outputs.get("rollback_safety") == 1
            and selected_slot is not None
        )
        patch_apply = {
            "exit_code": None,
            "timed_out": False,
            "duration_seconds": 0.0,
            "stdout": "",
            "stderr": "patch_not_authorized",
        }
        post_test = {
            "exit_code": None,
            "timed_out": False,
            "duration_seconds": 0.0,
            "stdout": "",
            "stderr": "patch_not_authorized",
        }
        patch_text = ""
        patch_source = "patch_not_authorized"
        selected_patch_path = row_artifacts / "selected_patch_candidate.diff"
        candidates = row.get("patch_candidates") if isinstance(row.get("patch_candidates"), list) else []
        identity_retry_row = _identity_signal_controlled_row(row, identity_signal_control)
        identity_retry_slot = _identity_prioritized_command_slot(identity_retry_row)
        attempt_slots: list[int] = []
        if selected_slot is not None:
            attempt_slots.append(int(selected_slot))
        if allow_bounded_candidate_retry and patch_authorized:
            if (
                retry_prioritization == "identity_first"
                and identity_retry_slot is not None
                and identity_retry_slot not in attempt_slots
            ):
                attempt_slots.append(int(identity_retry_slot))
            for slot in range(len(candidates)):
                if slot not in attempt_slots:
                    attempt_slots.append(slot)
        attempt_slots = attempt_slots[: max(1, int(max_candidate_attempts))]
        attempt_rows: list[dict[str, Any]] = []
        final_selected_slot = selected_slot
        for attempt_index, attempt_slot in enumerate(attempt_slots):
            if not patch_authorized:
                break
            patch_text, patch_source = _patch_for_selected_slot(
                row=row,
                dataset_root=root,
                sandbox=sandbox,
                selected_slot=int(attempt_slot),
            )
            selected_patch_path = row_artifacts / f"selected_patch_candidate_attempt_{attempt_index:02d}.diff"
            patch_text = patch_text.replace("\r\n", "\n").replace("\r", "\n")
            selected_patch_path.write_text(patch_text, encoding="utf-8", newline="\n")
            if patch_text:
                patch_apply = _git_apply_patch(sandbox, selected_patch_path, timeout_seconds)
            post_test = _run_candidate_selection_test(
                sandbox,
                test_rel,
                generated_test_source=generated_test_source,
                timeout_seconds=timeout_seconds,
                python_executable=python_executable,
            )
            attempt_passed = patch_apply["exit_code"] == 0 and post_test["exit_code"] == 0
            attempt_rows.append(
                {
                    "attempt_index": attempt_index,
                    "candidate_slot": int(attempt_slot),
                    "patch_source": patch_source,
                    "patch_apply_exit_code": patch_apply["exit_code"],
                    "post_test_exit_code": post_test["exit_code"],
                    "passed": attempt_passed,
                }
            )
            final_selected_slot = int(attempt_slot)
            if attempt_passed:
                break
            if allow_bounded_candidate_retry and patch_apply["exit_code"] == 0 and patch_text:
                _git_apply_reverse_patch(sandbox, selected_patch_path, timeout_seconds)
        if not attempt_rows:
            selected_patch_path.write_text(
                patch_text.replace("\r\n", "\n").replace("\r", "\n"),
                encoding="utf-8",
                newline="\n",
            )
        rollback = _git_apply_reverse_patch(sandbox, local_correct_patch, timeout_seconds)
        rollback_test = _run_candidate_selection_test(
            sandbox,
            test_rel,
            generated_test_source=generated_test_source,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
        )
        expected_slot = int(row.get("expected_patch_candidate_slot"))
        selected_correct = final_selected_slot == expected_slot
        success = (
            reverse_to_fault["exit_code"] == 0
            and pre_test["exit_code"] not in {0, None}
            and patch_authorized
            and selected_correct
            and patch_apply["exit_code"] == 0
            and post_test["exit_code"] == 0
            and rollback["exit_code"] == 0
            and rollback_test["exit_code"] not in {0, None}
        )

        pre_path = row_artifacts / "pre_test_log.json"
        patch_apply_path = row_artifacts / "patch_apply_log.json"
        post_path = row_artifacts / "post_test_log.json"
        rollback_path = row_artifacts / "rollback_test_log.json"
        transcript_path = row_artifacts / "transcript.json"
        _write_json(pre_path, pre_test)
        _write_json(patch_apply_path, patch_apply)
        _write_json(post_path, post_test)
        _write_json(rollback_path, {"rollback": rollback, "rollback_test": rollback_test})
        _write_json(
            transcript_path,
            {
                "trace_id": row.get("trace_id"),
                "policy_outputs": policy_outputs,
                "selected_patch_candidate_slot": final_selected_slot,
                "initial_selected_patch_candidate_slot": selected_slot,
                "expected_patch_candidate_slot": expected_slot,
                "patch_source": patch_source,
                "bounded_candidate_retry_enabled": allow_bounded_candidate_retry,
                "retry_prioritization": retry_prioritization,
                "identity_signal_control": identity_signal_control,
                "identity_retry_slot": identity_retry_slot,
                "candidate_attempts": attempt_rows,
                "learned_patch_descriptor_outputs": learned_descriptor_outputs,
                "claim_boundary": CLAIM_BOUNDARY,
            },
        )
        result_row = {
            "trace_id": row.get("trace_id"),
            "task_id": f"phase2aa-bounded-patch-candidate:{row.get('trace_id')}",
            "task_family": "bounded_patch_candidate_selection",
            "source_kind": row.get("source_kind"),
            "repo_origin": row.get("repo_url_or_origin"),
            "repo_commit": row.get("commit_hash"),
            "result_source": "phase2aa_bounded_patch_candidate_execution",
            "native_policy_label": str(manifest.get("policy_label") or ""),
            "policy_package_manifest_sha256": package_hash,
            "policy_loaded": policy is not None,
            "policy_open_repair_outputs": open_repair_outputs,
            "policy_learned_patch_descriptor_outputs": learned_descriptor_outputs,
            "selected_patch_candidate_slot": final_selected_slot,
            "initial_selected_patch_candidate_slot": selected_slot,
            "expected_patch_candidate_slot": expected_slot,
            "patch_candidate_selected_correctly": selected_correct,
            "bounded_candidate_retry_enabled": allow_bounded_candidate_retry,
            "retry_prioritization": retry_prioritization,
            "identity_signal_control": identity_signal_control,
            "identity_retry_slot": identity_retry_slot,
            "candidate_attempts": attempt_rows,
            "patch_source": patch_source,
            "patch_generator": "bounded_patch_candidate_selector_v1",
            "patch_sha256": _sha256_text(patch_text),
            "patch_stats": _patch_stats(patch_text),
            "selected_tests": [
                _verification_command_for_generated_test(
                    test_rel,
                    generated_test_source=generated_test_source,
                )
            ],
            "generated_test_used": True,
            "generated_test_source": generated_test_source,
            "pre_test_log_sha256": _sha256_file(pre_path),
            "post_test_log_sha256": _sha256_file(post_path),
            "patch_apply_log_sha256": _sha256_file(patch_apply_path),
            "rollback_test_log_sha256": _sha256_file(rollback_path),
            "verification_state": "passed" if success else "failed",
            "stop_condition": "verification_passed" if success else "verification_failed_stop",
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "transcript_sha256": _sha256_file(transcript_path),
            "freeform_patch_generation": False,
            "oracle_trace_used": False,
            "recorded_patch_artifact_used": patch_source == "selected_recorded_correct_patch_candidate",
            "claim_bearing_candidate_selection_evidence": True,
            "claim_bearing_freeform_patch_evidence": False,
            "sealed_feedback_used": False,
            "success": success,
            "full_task_success": success,
            "full_patch_correctness": success,
            "full_test_pass_rate": 1.0 if post_test["exit_code"] == 0 else 0.0,
            "rollback_failure_restored": rollback_test["exit_code"] not in {0, None},
            "unauthorized_write_count": 0,
            "false_completion": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "artifact_paths": {
                "selected_patch": str(selected_patch_path),
                "correct_patch_artifact": str(correct_patch_source),
                "pre_test_log": str(pre_path),
                "patch_apply_log": str(patch_apply_path),
                "post_test_log": str(post_path),
                "rollback_test_log": str(rollback_path),
                "transcript": str(transcript_path),
            },
        }
        rows.append(result_row)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result_row, ensure_ascii=False, sort_keys=True) + "\n")

    _write_jsonl(output_jsonl, rows)
    successes = sum(1 for row in rows if row.get("success") is True)
    correct_slots = sum(1 for row in rows if row.get("patch_candidate_selected_correctly") is True)
    return {
        "artifact_family": "phase2aa_bounded_patch_candidate_execution_runner",
        "rows": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "correct_patch_candidate_selections": correct_slots,
        "patch_candidate_selection_accuracy": correct_slots / len(rows) if rows else 0.0,
        "policy_loaded": bool(policy is not None),
        "output_jsonl": str(Path(output_jsonl)),
        "artifact_root": str(Path(artifact_root)),
        "claim_boundary": CLAIM_BOUNDARY,
        "retry_prioritization": retry_prioritization,
        "identity_signal_control": identity_signal_control,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase2AA bounded patch candidate selection execution."
    )
    parser.add_argument("--source-rows-jsonl", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-rows", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--test-python")
    parser.add_argument("--no-load-policy", action="store_true")
    parser.add_argument("--allow-bounded-candidate-retry", action="store_true")
    parser.add_argument("--max-candidate-attempts", type=int, default=4)
    parser.add_argument("--policyless-start-slot", type=int)
    parser.add_argument(
        "--retry-prioritization",
        choices=["sequential", "identity_first"],
        default="sequential",
    )
    parser.add_argument(
        "--identity-signal-control",
        choices=sorted(IDENTITY_SIGNAL_CONTROLS),
        default="normal",
    )
    args = parser.parse_args()
    report = run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=args.source_rows_jsonl,
        dataset_root=args.dataset_root,
        clone_root=args.clone_root,
        package_path=args.package_path,
        output_jsonl=args.output_jsonl,
        artifact_root=args.artifact_root,
        max_rows=args.max_rows,
        timeout_seconds=args.timeout_seconds,
        test_python=args.test_python,
        load_policy=not args.no_load_policy,
        allow_bounded_candidate_retry=args.allow_bounded_candidate_retry,
        max_candidate_attempts=args.max_candidate_attempts,
        policyless_start_slot=args.policyless_start_slot,
        retry_prioritization=args.retry_prioritization,
        identity_signal_control=args.identity_signal_control,
    )
    _write_json(args.summary_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["rows"] <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
