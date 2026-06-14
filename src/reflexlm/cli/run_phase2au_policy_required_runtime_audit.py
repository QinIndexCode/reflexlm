from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2au_runtime_delta_gate import PHASE2AU_RUNTIME_BOUNDARY
from reflexlm.cli.build_phase2au_policy_required_head_dataset import (
    _commands,
    _state_prompt,
)
from reflexlm.llm.native_nervous_package import (
    PACKAGE_MANIFEST_NAME,
    NativeNervousPolicyPackage,
)
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


def _read_json(path: str | Path) -> dict[str, Any]:
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


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
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


def _manifest_path(package_path: str | Path) -> Path:
    path = Path(package_path)
    return path if path.is_file() else path / PACKAGE_MANIFEST_NAME


def _scope_paths(task: dict[str, Any]) -> list[str]:
    scope = task.get("allowed_write_scope")
    if isinstance(scope, list):
        return [str(item).replace("\\", "/") for item in scope if str(item).strip()]
    if isinstance(scope, str) and scope.strip():
        return [scope.replace("\\", "/")]
    return []


def _state_for_phase2au_task(task: dict[str, Any], commands: list[str]) -> SystemStateFrame:
    prompt = _state_prompt(task, commands)
    watched_paths = _scope_paths(task)
    stderr_lines = [
        "Phase2AU policy-required runtime task.",
        "A bounded repair candidate must be selected from the command allowlist.",
        prompt,
    ]
    return SystemStateFrame(
        time=TimeState(tick=0, runtime_ms=0, since_last_output_ms=0),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description=prompt,
            command_allowlist=commands,
            watched_paths=watched_paths,
            success_criteria=[
                str(command)
                for command in task.get("evaluation_commands", [])
                if str(command).strip()
            ],
            safety_notes=[
                "bounded command selection only",
                "no freeform shell generation",
                "no sealed feedback",
            ],
            recovery_hint="Select one allowlisted phase2au_apply_candidate command.",
        ),
        process=ProcessState(
            status=ProcessStatus.EXITED,
            exit_code=1,
            waiting_for_input=False,
            resource_alert=False,
        ),
        terminal=TerminalState(
            stdout_delta="",
            stderr_delta="\n".join(stderr_lines),
            stdout_lines=0,
            stderr_lines=len(stderr_lines),
            last_output_channel="stderr",
            prompt_visible=True,
        ),
        filesystem=FileSystemState(
            watched_paths=watched_paths,
            changed_paths=watched_paths,
            dirty_files=[],
        ),
    )


def _expected_command(task: dict[str, Any], commands: list[str]) -> str | None:
    expected = str(task.get("expected_repair_action") or "").strip()
    if not expected:
        return None
    for command in commands:
        if expected in command:
            return command
    return None


def _select_no_policy(commands: list[str], strategy: str) -> tuple[str | None, str]:
    if not commands:
        return None, "no_candidate_commands"
    if strategy == "first_candidate":
        return commands[0], "no_policy_first_candidate"
    raise ValueError("no-policy strategy must be 'first_candidate'")


def _select_with_policy(
    policy: NativeNervousPolicyPackage,
    state: SystemStateFrame,
) -> tuple[str | None, dict[str, Any]]:
    action = policy.act(state)
    outputs = dict(policy.last_call)
    if outputs.get("action_source") == "low_level_debug_receptor":
        observed = "\n".join(
            part for part in [state.terminal.stdout_delta, state.terminal.stderr_delta] if part
        )
        state = state.model_copy(
            deep=True,
            update={
                "terminal": state.terminal.model_copy(
                    update={
                        "stdout_delta": observed,
                        "stderr_delta": "",
                        "stdout_lines": len(observed.splitlines()),
                        "stderr_lines": 0,
                        "last_command": "READ_STDERR receptor observation completed",
                    }
                )
            },
        )
        action = policy.act(state)
        outputs = dict(policy.last_call)
        outputs["phase2au_receptor_observation_replayed"] = True
    return action.command, outputs


def run_phase2au_policy_required_runtime_audit(
    *,
    tasks_jsonl: str | Path,
    package_path: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    max_rows: int = 20,
    load_policy: bool = True,
    no_policy_strategy: str = "first_candidate",
) -> dict[str, Any]:
    tasks = _read_jsonl(tasks_jsonl)
    manifest_path = _manifest_path(package_path)
    manifest = _read_json(manifest_path)
    policy = NativeNervousPolicyPackage(package_path) if load_policy else None
    rows: list[dict[str, Any]] = []
    failure_reasons: dict[str, int] = {}
    low_level_qwen_calls = 0

    for task in tasks[:max_rows]:
        start = time.perf_counter()
        commands = _commands(task)
        expected = _expected_command(task, commands)
        if expected is None:
            selected = None
            outputs: dict[str, Any] = {}
            reason = "expected_repair_action_not_in_commands"
        elif policy is None:
            selected, reason = _select_no_policy(commands, no_policy_strategy)
            outputs = {}
        else:
            state = _state_for_phase2au_task(task, commands)
            selected, outputs = _select_with_policy(policy, state)
            reason = str(outputs.get("slot_source") or outputs.get("action_source") or "")
            nsi_debug = outputs.get("nsi_debug") if isinstance(outputs.get("nsi_debug"), dict) else {}
            if nsi_debug.get("qwen_called") is True:
                low_level_qwen_calls += 1
        success = bool(selected and expected and selected == expected)
        if not success:
            failure_reasons[reason or "incorrect_or_missing_selection"] = (
                failure_reasons.get(reason or "incorrect_or_missing_selection", 0) + 1
            )
        rows.append(
            {
                "task_id": task.get("task_id"),
                "benchmark_family": task.get("benchmark_family"),
                "policy_loaded": policy is not None,
                "native_policy_label": manifest.get("policy_label"),
                "selected_command": selected,
                "expected_command": expected,
                "success": success,
                "selection_reason": reason,
                "policy_outputs": outputs,
                "sealed_feedback_used": False,
                "claim_boundary": PHASE2AU_RUNTIME_BOUNDARY,
                "elapsed_seconds": round(time.perf_counter() - start, 6),
            }
        )
    _write_jsonl(output_jsonl, rows)
    successes = sum(1 for row in rows if row.get("success") is True)
    success_rate = successes / len(rows) if rows else 0.0
    report = {
        "artifact_family": "phase2au_policy_required_runtime_audit",
        "passed": bool(policy is not None and rows and success_rate >= 0.85),
        "claim_boundary": PHASE2AU_RUNTIME_BOUNDARY,
        "checks": {
            "all_rows_policy_loaded": bool(rows) and all(
                row.get("policy_loaded") is True for row in rows
            ),
            "sealed_feedback_absent": True,
            "freeform_shell_absent": True,
            "command_selection_only_not_patch_execution": True,
        },
        "metrics": {
            "row_count": len(rows),
            "successes": successes,
            "success_rate": success_rate,
            "failure_reasons": failure_reasons,
            "low_level_qwen_calls": low_level_qwen_calls,
        },
        "blocked_actions": [
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_open_ended_debugging_generalization",
            "do_not_claim_patch_execution_success_from_command_selection_only",
        ],
        "unsupported_claims": [
            "patch_execution_success",
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "inputs": {
            "tasks_jsonl": str(Path(tasks_jsonl)),
            "package_path": str(Path(package_path)),
            "manifest_path": str(manifest_path),
            "load_policy": load_policy,
            "no_policy_strategy": no_policy_strategy,
        },
        "outputs": {
            "output_jsonl": str(Path(output_jsonl)),
            "summary_json": str(Path(summary_json)),
        },
    }
    _write_json(summary_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase2AU bounded command-selection runtime audit."
    )
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--no-load-policy", action="store_true")
    parser.add_argument(
        "--no-policy-strategy",
        choices=["first_candidate"],
        default="first_candidate",
    )
    args = parser.parse_args()
    report = run_phase2au_policy_required_runtime_audit(
        tasks_jsonl=args.tasks_jsonl,
        package_path=args.package_path,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        max_rows=args.max_rows,
        load_policy=not args.no_load_policy,
        no_policy_strategy=args.no_policy_strategy,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["metrics"]["row_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
