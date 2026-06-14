from __future__ import annotations

import argparse
import copy
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from reflexlm.cli.build_phase2ax_head_dataset import _command_identity_reference
from reflexlm.cli.run_phase2z_public_structural_repair_execution import (
    _state_after_stderr_receptor,
    run_phase2z_public_structural_repair_execution,
)
from reflexlm.cli.run_phase2z_synthetic_nonliteral_repair_plumbing import (
    _read_jsonl,
    _sha256_text,
    _write_json,
    _write_jsonl,
)
from reflexlm.llm.native_nervous_package import NativeNervousPolicyPackage
from reflexlm.runtime.plasticity import PLASTICITY_CONTROLS
from reflexlm.schema import (
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    RuntimeEvidenceState,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


CLAIM_BOUNDARY = (
    "phase2ay_slot_conditioned_runtime_execution_smoke_not_phase2ax_package_or_epoch_claim"
)
SUPPORTED_SELECTION_POLICIES = {
    "prior_runtime_resolver",
    "current_only_slot0",
    "wrong_cache",
    "expected_oracle",
    "model_prediction_records",
    "package_loaded_native_head",
}
SUPPORTED_NSI_REFERENCE_MODES = {
    "runtime_visible_override",
    "low_level_only",
}
SUPPORTED_RUNTIME_EVIDENCE_LABELS = {
    "Prior runtime evidence",
    "Runtime-visible repair evidence",
}
SUPPORTED_RUNTIME_EVIDENCE_CHANNELS = {
    "prompt_text",
    "structured_receptor",
    "dual",
}
SUPPORTED_RUNTIME_EVIDENCE_CONTROLS = {
    "normal",
    "erased",
    "identity_erased",
    "wrong",
}
_RUNTIME_EVIDENCE_SECTION_RE = re.compile(
    r"(?P<prefix>\A|\n\n)(?:Prior runtime evidence|Runtime-visible repair evidence):\n"
    r"\{.*?\}(?=\n\n[A-Z][^\n]*:|\Z)",
    re.DOTALL,
)
_STRUCTURAL_REPAIR_ID_RE = re.compile(r"structural_repair_([a-zA-Z0-9]+)")


def _repo_id_from_origin(origin: str) -> str:
    parsed = urlparse(origin)
    path = parsed.path if parsed.scheme else origin
    return path.strip("/").removesuffix(".git").replace("/", "_").replace("-", "_").lower()


def _test_python_for_row(
    row: dict[str, Any],
    *,
    default_test_python: str | None,
    test_python_map: dict[str, Any] | None,
) -> tuple[str | None, str]:
    mapping = test_python_map if isinstance(test_python_map, dict) else {}
    repos = mapping.get("repos") if isinstance(mapping.get("repos"), dict) else {}
    origin = str(row.get("repo_origin") or "")
    repo_id = _repo_id_from_origin(origin)
    for key in (origin, repo_id):
        executable = repos.get(key)
        if isinstance(executable, str) and executable.strip():
            return executable, f"repo_override:{key}"
    if default_test_python:
        return default_test_python, "cli_default"
    mapped_default = mapping.get("default")
    if isinstance(mapped_default, str) and mapped_default.strip():
        return mapped_default, "map_default"
    return None, "runtime_default"


def _candidate_actions(row: dict[str, Any]) -> list[str]:
    candidates = row.get("repair_candidates") if isinstance(row.get("repair_candidates"), list) else []
    return [
        str(candidate.get("repair_action") or "")
        for candidate in candidates
        if isinstance(candidate, dict)
    ]


def _expected_slot(row: dict[str, Any]) -> int | None:
    expected = str(row.get("expected_repair_action") or "")
    for index, action in enumerate(_candidate_actions(row)):
        if action == expected:
            return index
    return None


def _prior_runtime_slot(row: dict[str, Any]) -> int | None:
    prior = (
        row.get("phase2ax_prior_runtime_evidence")
        if isinstance(row.get("phase2ax_prior_runtime_evidence"), dict)
        else {}
    )
    probes = [str(value) for value in prior.get("structural_probe_hashes", []) if value]
    actions = _candidate_actions(row)
    for index, action in enumerate(actions):
        if any(probe[:12] in action or probe in action for probe in probes):
            return index
    return None


def _prediction_key(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("episode_id") or "")


def _prediction_records_by_episode(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        episode_id = str(record.get("episode_id") or "")
        if episode_id:
            indexed[episode_id] = record
        example_id = str(record.get("example_id") or "")
        if example_id:
            indexed[example_id] = record
            if ":val:" in example_id:
                indexed[example_id.split(":val:", 1)[0]] = record
    return indexed


def _head_records_by_episode(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        episode_id = str(record.get("episode_id") or "")
        if episode_id:
            indexed[episode_id] = record
        example_id = str(record.get("example_id") or "")
        if example_id:
            indexed[example_id] = record
            if ":val:" in example_id:
                indexed[example_id.split(":val:", 1)[0]] = record
    return indexed


def _phase2ax_candidate_commands(
    row: dict[str, Any],
    head_record: dict[str, Any] | None,
) -> list[str]:
    if isinstance(head_record, dict) and isinstance(head_record.get("candidate_commands"), list):
        commands = [str(command) for command in head_record["candidate_commands"] if command]
        if commands:
            return commands
    commands = row.get("candidate_policy_commands")
    if isinstance(commands, list):
        normalized = [str(command) for command in commands if command]
        if normalized:
            return normalized
    return _candidate_actions(row)


def _phase2ax_visible_state_prompt(
    row: dict[str, Any],
    head_record: dict[str, Any] | None,
    runtime_evidence_label: str = "Prior runtime evidence",
    runtime_evidence_payload: dict[str, Any] | None = None,
    include_runtime_evidence: bool = True,
) -> str:
    if runtime_evidence_label not in SUPPORTED_RUNTIME_EVIDENCE_LABELS:
        raise ValueError(
            "runtime_evidence_label must be one of "
            f"{sorted(SUPPORTED_RUNTIME_EVIDENCE_LABELS)}"
        )
    if isinstance(head_record, dict) and isinstance(head_record.get("state_prompt"), str):
        prompt = str(head_record["state_prompt"])
        def replace_runtime_evidence(match: re.Match[str]) -> str:
            prefix = match.group("prefix")
            if not include_runtime_evidence:
                return prefix
            if runtime_evidence_payload is None:
                section = match.group(0)
                for label in SUPPORTED_RUNTIME_EVIDENCE_LABELS:
                    section = section.replace(
                        f"{label}:",
                        f"{runtime_evidence_label}:",
                    )
                return section
            payload = (
                runtime_evidence_payload
                if runtime_evidence_payload is not None
                else {}
            )
            return (
                f"{prefix}{runtime_evidence_label}:\n"
                f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
            )

        prompt = _RUNTIME_EVIDENCE_SECTION_RE.sub(replace_runtime_evidence, prompt)
        return prompt
    lines = [
            "Phase2AX package-loaded counterfactual repair native-head input.",
            "Use only masked current repair surface plus prior runtime evidence.",
            "The current repair surface is identical within each counterfactual pair.",
            "Do not use sealed feedback, gold labels, candidate slot markers, pre-test stdout, or freeform patch text.",
            "",
            "task_family=phase2ax_package_loaded_counterfactual_repair",
            f"repo_origin={row.get('repo_origin')}",
            f"repo_commit={row.get('repo_commit')}",
            f"pair_id={row.get('phase2ax_pair_id')}",
            "",
            "Masked current repair surface:",
            json.dumps(
                row.get("phase2ax_current_repair_surface") or {},
                ensure_ascii=False,
                sort_keys=True,
            ),
        ]
    if include_runtime_evidence:
        lines.extend(
            [
                "",
                f"{runtime_evidence_label}:",
                json.dumps(
                    runtime_evidence_payload
                    if runtime_evidence_payload is not None
                    else row.get("phase2ax_prior_runtime_evidence") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "Runtime-visible contract:",
            json.dumps(
                row.get("runtime_visible_contract") or {},
                ensure_ascii=False,
                sort_keys=True,
            ),
            "",
            "Candidate commands:",
            *[f"- {command}" for command in _phase2ax_candidate_commands(row, head_record)],
        ]
    )
    return "\n".join(lines)


def _controlled_runtime_evidence(
    row: dict[str, Any],
    commands: list[str],
    control: str,
) -> dict[str, Any]:
    if control not in SUPPORTED_RUNTIME_EVIDENCE_CONTROLS:
        raise ValueError(
            "runtime evidence control must be one of "
            f"{sorted(SUPPORTED_RUNTIME_EVIDENCE_CONTROLS)}"
        )
    evidence = copy.deepcopy(
        row.get("phase2ax_prior_runtime_evidence")
        if isinstance(row.get("phase2ax_prior_runtime_evidence"), dict)
        else {}
    )
    if control == "normal":
        return evidence
    if control == "erased":
        return {}
    if control == "identity_erased":
        for key in (
            "structural_probe_hashes",
            "expected_literal_hash",
            "target_path_hash",
            "target_symbol_hash",
            "target_location",
            "target_path",
            "target_line",
            "target_col",
        ):
            evidence.pop(key, None)
        return evidence
    probes = [str(value).lower() for value in evidence.get("structural_probe_hashes", []) if value]
    matched_index = next(
        (
            index
            for index, command in enumerate(commands)
            if any(probe[:12] in command.lower() or probe in command.lower() for probe in probes)
        ),
        None,
    )
    if matched_index is None or len(commands) <= 1:
        evidence["structural_probe_hashes"] = ["wrong_runtime_evidence"]
        return evidence
    wrong_command = commands[(matched_index + 1) % len(commands)]
    match = _STRUCTURAL_REPAIR_ID_RE.search(wrong_command)
    evidence["structural_probe_hashes"] = [
        f"{match.group(1)}ffff" if match else "wrong_runtime_evidence"
    ]
    return evidence


def _runtime_evidence_state(payload: dict[str, Any]) -> RuntimeEvidenceState:
    location = payload.get("target_location") if isinstance(payload.get("target_location"), dict) else {}
    return RuntimeEvidenceState(
        source="phase2ax_prior_runtime_receptor" if payload else None,
        version=str(payload.get("version")) if payload.get("version") else None,
        changed_files=[str(value) for value in payload.get("changed_files", []) if value],
        watched_files=[str(value) for value in payload.get("watched_files", []) if value],
        structural_probe_hashes=[
            str(value) for value in payload.get("structural_probe_hashes", []) if value
        ],
        traceback_symbols=[str(value) for value in payload.get("traceback_symbols", []) if value],
        repair_modes=[str(value) for value in payload.get("repair_modes", []) if value],
        descriptor_operation=payload.get("descriptor_operation"),
        descriptor_template=payload.get("descriptor_template"),
        expected_literal_hash=payload.get("expected_literal_hash"),
        target_path_hash=payload.get("target_path_hash"),
        target_symbol_hash=payload.get("target_symbol_hash"),
        target_path=payload.get("target_path") or location.get("path"),
        target_line=payload.get("target_line") or location.get("line"),
        target_col=payload.get("target_col") or location.get("col"),
    )


def _phase2ax_nsi_reference(row: dict[str, Any], commands: list[str]) -> dict[str, Any]:
    return {
        "reflex_action": "RUN_COMMAND",
        "route_name": "debug_cortex",
        "confidence": 0.9,
        "risk": 0.35,
        "salience": 0.85,
        "prediction_error": 0.2,
        **_command_identity_reference(row, commands),
    }


def _select_with_package_native_head_state(
    *,
    policy: Any,
    row: dict[str, Any],
    head_record: dict[str, Any] | None,
    row_index: int,
    nsi_reference_mode: str = "runtime_visible_override",
    runtime_evidence_label: str = "Prior runtime evidence",
    runtime_evidence_channel: str = "prompt_text",
    runtime_evidence_control: str = "normal",
) -> dict[str, Any]:
    if nsi_reference_mode not in SUPPORTED_NSI_REFERENCE_MODES:
        raise ValueError(
            f"nsi_reference_mode must be one of {sorted(SUPPORTED_NSI_REFERENCE_MODES)}"
        )
    commands = _phase2ax_candidate_commands(row, head_record)
    if runtime_evidence_channel not in SUPPORTED_RUNTIME_EVIDENCE_CHANNELS:
        raise ValueError(
            "runtime_evidence_channel must be one of "
            f"{sorted(SUPPORTED_RUNTIME_EVIDENCE_CHANNELS)}"
        )
    evidence = _controlled_runtime_evidence(row, commands, runtime_evidence_control)
    visible_prompt = _phase2ax_visible_state_prompt(
        row,
        head_record,
        runtime_evidence_label=runtime_evidence_label,
        runtime_evidence_payload=evidence,
        include_runtime_evidence=runtime_evidence_channel in {"prompt_text", "dual"},
    )
    prior = (
        row.get("phase2ax_prior_runtime_evidence")
        if isinstance(row.get("phase2ax_prior_runtime_evidence"), dict)
        else {}
    )
    changed_files = [
        str(path)
        for path in prior.get("changed_files", [])
        if isinstance(prior.get("changed_files"), list) and path
    ]
    watched_paths = [
        str(path)
        for path in [
            *(prior.get("watched_files", []) if isinstance(prior.get("watched_files"), list) else []),
            *changed_files,
        ]
        if path
    ]
    state = SystemStateFrame(
        time=TimeState(tick=row_index + 1, runtime_ms=0),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="Phase2AX package-loaded counterfactual repair selection.",
            command_allowlist=commands,
            watched_paths=watched_paths,
            success_criteria=[
                "select_bounded_counterfactual_repair_candidate",
                "use_prior_runtime_evidence",
                "do_not_use_sealed_feedback",
            ],
            safety_notes=[
                "fixed_candidate_command_space",
                "no_freeform_shell_generation",
                "no_freeform_patch_generation",
            ],
        ),
        process=ProcessState(status=ProcessStatus.EXITED, exit_code=1, runtime_ms=0),
        terminal=TerminalState(
            stdout_delta=visible_prompt,
            stderr_delta="",
            stdout_lines=len(visible_prompt.splitlines()),
            stderr_lines=0,
            last_command="PHASE2AX_PRIOR_RUNTIME_RECEPTOR",
        ),
        filesystem=FileSystemState(
            watched_paths=watched_paths,
            changed_paths=changed_files,
            dirty_files=changed_files,
        ),
        runtime_evidence=_runtime_evidence_state(evidence)
        if runtime_evidence_channel in {"structured_receptor", "dual"}
        else RuntimeEvidenceState(),
    )
    if hasattr(policy, "reset"):
        policy.reset()
    nsi_reference = (
        _phase2ax_nsi_reference(row, commands)
        if nsi_reference_mode == "runtime_visible_override"
        else None
    )
    policy.act(state, nsi_reference_override=nsi_reference)
    policy_outputs = dict(getattr(policy, "last_call", {}) or {})
    receptor_observed = False
    if policy_outputs.get("action_source") == "low_level_debug_receptor":
        receptor_observed = True
        state = _state_after_stderr_receptor(state)
        policy.act(state, nsi_reference_override=nsi_reference)
        policy_outputs = dict(getattr(policy, "last_call", {}) or {})
    cortex_plan = policy_outputs.get("cortex_plan")
    selected_slot = (
        int(cortex_plan["command_slot"])
        if isinstance(cortex_plan, dict) and isinstance(cortex_plan.get("command_slot"), int)
        else None
    )
    open_repair_outputs = (
        policy_outputs.get("open_repair_head_outputs")
        if isinstance(policy_outputs.get("open_repair_head_outputs"), dict)
        else {}
    )
    return {
        "selected_slot": selected_slot,
        "policy_outputs": policy_outputs,
        "low_level_debug_receptor_observed": receptor_observed,
        "qwen_called": bool(policy_outputs.get("qwen_called")),
        "open_repair_authorized": open_repair_outputs.get("patch_proposal") == 1
        and open_repair_outputs.get("bounded_edit_scope") == 1
        and open_repair_outputs.get("rollback_safety") == 1,
        "visible_state_source": "phase2ax_head_record"
        if isinstance(head_record, dict)
        else "phase2ax_task_record",
        "nsi_reference_override": nsi_reference,
        "nsi_reference_mode": nsi_reference_mode,
        "runtime_evidence_label": runtime_evidence_label,
        "runtime_evidence_channel": runtime_evidence_channel,
        "runtime_evidence_control": runtime_evidence_control,
        "runtime_evidence_prompt_present": any(
            f"{label}:" in visible_prompt for label in SUPPORTED_RUNTIME_EVIDENCE_LABELS
        ),
        "structured_runtime_evidence": state.runtime_evidence.model_dump(
            mode="json", exclude_none=True
        ),
    }


def _selected_slot(
    row: dict[str, Any],
    policy: str,
    prediction_records: dict[str, dict[str, Any]] | None = None,
) -> int | None:
    if policy == "prior_runtime_resolver":
        return _prior_runtime_slot(row)
    if policy == "current_only_slot0":
        return 0
    if policy == "wrong_cache":
        expected = _expected_slot(row)
        if expected is None:
            return None
        actions = _candidate_actions(row)
        if len(actions) <= 1:
            return None
        return (expected + 1) % len(actions)
    if policy == "expected_oracle":
        return _expected_slot(row)
    if policy == "model_prediction_records":
        record = (prediction_records or {}).get(_prediction_key(row))
        if not isinstance(record, dict):
            return None
        try:
            return int(record.get("command_slot_prediction"))
        except (TypeError, ValueError):
            return None
    if policy == "package_loaded_native_head":
        return None
    raise ValueError(f"unsupported selection policy: {policy}")


def _runner_row(row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    generated_tests = (
        converted.get("artifact_paths", {}).get("generated_tests")
        if isinstance(converted.get("artifact_paths"), dict)
        else None
    )
    generated_test = str(generated_tests[0]) if isinstance(generated_tests, list) and generated_tests else ""
    if not generated_test:
        raise ValueError(f"row is missing artifact_paths.generated_tests: {row.get('task_id')}")
    prior = (
        row.get("phase2ax_prior_runtime_evidence")
        if isinstance(row.get("phase2ax_prior_runtime_evidence"), dict)
        else {}
    )
    watched = [str(path) for path in prior.get("watched_files", []) if path]
    test_target = watched[0] if watched else str(Path(generated_test).name)
    artifact_paths = dict(converted.get("artifact_paths") or {})
    artifact_paths["generated_test"] = generated_test
    artifact_paths["patch_diff"] = str(Path(generated_test).parent / "patch.diff").replace("\\", "/")
    runtime_evidence = {
        "changed_files": [str(path) for path in prior.get("changed_files", []) if path],
        "watched_files": watched,
        "structural_probe_hashes": [
            str(value) for value in prior.get("structural_probe_hashes", []) if value
        ],
        "repair_modes": [str(value) for value in prior.get("repair_modes", []) if value],
    }
    converted.update(
        {
            "repo_id": _repo_id_from_origin(str(row.get("repo_origin") or "")),
            "repo_url_or_origin": row.get("repo_origin"),
            "commit_hash": row.get("repo_commit"),
            "artifact_paths": artifact_paths,
            "runtime_visible_evidence": runtime_evidence,
            "expected_repair_result": {"test_target": test_target},
        }
    )
    return converted


def _contract_allows_execution(row: dict[str, Any]) -> bool:
    contract = row.get("runtime_visible_contract")
    return (
        isinstance(contract, dict)
        and contract.get("no_freeform_patch_generation") is True
        and contract.get("no_sealed_feedback") is True
        and contract.get("no_gold_hint") is True
        and row.get("sealed_feedback_used") is False
    )


def _execute_selected_row(
    *,
    row: dict[str, Any],
    dataset_root: Path,
    clone_root: Path,
    package_path: Path,
    artifact_root: Path,
    timeout_seconds: int,
    test_python: str | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="phase2ay_exec_") as tmp:
        source_rows = Path(tmp) / "row.jsonl"
        execution_rows = Path(tmp) / "execution.jsonl"
        _write_jsonl(source_rows, [_runner_row(row)])
        summary = run_phase2z_public_structural_repair_execution(
            source_rows_jsonl=source_rows,
            dataset_root=dataset_root,
            clone_root=clone_root,
            package_path=package_path,
            output_jsonl=execution_rows,
            artifact_root=artifact_root,
            max_rows=1,
            timeout_seconds=timeout_seconds,
            test_python=test_python,
            load_policy=False,
            patch_mode="runtime_symbolic_structural",
        )
        rows = _read_jsonl(execution_rows)
        return {"summary": summary, "row": rows[0] if rows else {}}


def run_phase2ay_counterfactual_slot_execution(
    *,
    phase2ax_tasks_jsonl: str | Path,
    full_postflight_json: str | Path,
    dataset_root: str | Path,
    clone_root: str | Path,
    package_path: str | Path,
    output_jsonl: str | Path,
    artifact_root: str | Path,
    selection_policy: str = "prior_runtime_resolver",
    prediction_records_json: str | Path | None = None,
    head_records_jsonl: str | Path | None = None,
    max_rows: int = 8,
    timeout_seconds: int = 30,
    test_python: str | None = None,
    test_python_map_json: str | Path | None = None,
    package_nsi_reference_mode: str = "runtime_visible_override",
    package_runtime_evidence_label: str = "Prior runtime evidence",
    package_runtime_evidence_channel: str = "prompt_text",
    package_runtime_evidence_control: str = "normal",
    package_plasticity_memory_in: str | Path | None = None,
    package_plasticity_memory_out: str | Path | None = None,
    package_plasticity_feedback_enabled: bool = False,
    package_plasticity_control: str = "normal",
) -> dict[str, Any]:
    if selection_policy not in SUPPORTED_SELECTION_POLICIES:
        raise ValueError(f"selection_policy must be one of {sorted(SUPPORTED_SELECTION_POLICIES)}")
    if package_nsi_reference_mode not in SUPPORTED_NSI_REFERENCE_MODES:
        raise ValueError(
            "package_nsi_reference_mode must be one of "
            f"{sorted(SUPPORTED_NSI_REFERENCE_MODES)}"
        )
    if package_runtime_evidence_label not in SUPPORTED_RUNTIME_EVIDENCE_LABELS:
        raise ValueError(
            "package_runtime_evidence_label must be one of "
            f"{sorted(SUPPORTED_RUNTIME_EVIDENCE_LABELS)}"
        )
    if package_runtime_evidence_channel not in SUPPORTED_RUNTIME_EVIDENCE_CHANNELS:
        raise ValueError(
            "package_runtime_evidence_channel must be one of "
            f"{sorted(SUPPORTED_RUNTIME_EVIDENCE_CHANNELS)}"
        )
    if package_runtime_evidence_control not in SUPPORTED_RUNTIME_EVIDENCE_CONTROLS:
        raise ValueError(
            "package_runtime_evidence_control must be one of "
            f"{sorted(SUPPORTED_RUNTIME_EVIDENCE_CONTROLS)}"
        )
    if package_plasticity_control not in PLASTICITY_CONTROLS:
        raise ValueError(
            f"package_plasticity_control must be one of {sorted(PLASTICITY_CONTROLS)}"
        )
    full_postflight = json.loads(Path(full_postflight_json).read_text(encoding="utf-8-sig"))
    prediction_records: dict[str, dict[str, Any]] = {}
    if prediction_records_json is not None:
        prediction_payload = json.loads(Path(prediction_records_json).read_text(encoding="utf-8-sig"))
        records = (
            prediction_payload.get("prediction_records")
            if isinstance(prediction_payload, dict)
            else prediction_payload
        )
        prediction_records = _prediction_records_by_episode(records if isinstance(records, list) else [])
    head_records: dict[str, dict[str, Any]] = {}
    if head_records_jsonl is not None:
        head_records = _head_records_by_episode(_read_jsonl(head_records_jsonl))
    test_python_map: dict[str, Any] = {}
    if test_python_map_json is not None:
        payload = json.loads(Path(test_python_map_json).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("test_python_map_json must contain a JSON object")
        test_python_map = payload
    all_tasks = _read_jsonl(phase2ax_tasks_jsonl)
    if selection_policy == "model_prediction_records":
        all_tasks = [row for row in all_tasks if _prediction_key(row) in prediction_records]
    tasks = all_tasks[:max_rows]
    rows: list[dict[str, Any]] = []
    root = Path(dataset_root)
    clones = Path(clone_root)
    package = Path(package_path)
    artifacts = Path(artifact_root)
    artifacts.mkdir(parents=True, exist_ok=True)
    package_policy = (
        NativeNervousPolicyPackage(package)
        if selection_policy == "package_loaded_native_head"
        else None
    )
    package_policy_metadata = (
        dict(package_policy.metadata()) if package_policy is not None else {}
    )
    if package_policy is not None and package_plasticity_memory_in is not None:
        package_policy.load_plasticity_memory(package_plasticity_memory_in)
        package_policy_metadata = dict(package_policy.metadata())
    if package_policy is not None and hasattr(package_policy, "set_plasticity_control"):
        package_policy.set_plasticity_control(package_plasticity_control)
    plasticity_feedback_rows = 0
    plasticity_feedback_accepted_rows = 0

    for index, row in enumerate(tasks):
        started = time.perf_counter()
        expected = _expected_slot(row)
        prediction_record = prediction_records.get(_prediction_key(row))
        actions = _candidate_actions(row)
        trace_id = str(row.get("task_id") or f"row-{index}")
        row_artifacts = artifacts / f"row_{index:05d}_{_sha256_text(trace_id)[:12]}"
        package_selection: dict[str, Any] = {}
        if package_policy is not None:
            package_selection = _select_with_package_native_head_state(
                policy=package_policy,
                row=row,
                head_record=head_records.get(_prediction_key(row)),
                row_index=index,
                nsi_reference_mode=package_nsi_reference_mode,
                runtime_evidence_label=package_runtime_evidence_label,
                runtime_evidence_channel=package_runtime_evidence_channel,
                runtime_evidence_control=package_runtime_evidence_control,
            )
            selected = package_selection.get("selected_slot")
        else:
            selected = _selected_slot(row, selection_policy, prediction_records)
        selected_action = actions[selected] if isinstance(selected, int) and 0 <= selected < len(actions) else ""
        expected_action = actions[expected] if isinstance(expected, int) and 0 <= expected < len(actions) else ""
        row_test_python, row_test_python_source = _test_python_for_row(
            row,
            default_test_python=test_python,
            test_python_map=test_python_map,
        )
        slot_correct = selected == expected and selected is not None
        contract_allowed = _contract_allows_execution(row)
        execution_payload: dict[str, Any] = {}
        executed_row: dict[str, Any] = {}
        if full_postflight.get("passed") is True and slot_correct and contract_allowed:
            execution_payload = _execute_selected_row(
                row=row,
                dataset_root=root,
                clone_root=clones,
                package_path=package,
                artifact_root=row_artifacts / "execution",
                timeout_seconds=timeout_seconds,
                test_python=row_test_python,
            )
            executed_row = execution_payload.get("row") or {}
            success = executed_row.get("success") is True
            stop_condition = executed_row.get("stop_condition") or "execution_finished"
            verification_state = executed_row.get("verification_state")
            artifact_paths = executed_row.get("artifact_paths") or {}
            attempted_execution = True
        else:
            success = False
            attempted_execution = False
            artifact_paths = {}
            if full_postflight.get("passed") is not True:
                stop_condition = "phase2ax_full_postflight_required_before_execution"
            elif not contract_allowed:
                stop_condition = "runtime_contract_blocks_execution"
            else:
                stop_condition = "slot_selection_failed_before_execution"
            verification_state = "blocked"
        plasticity_feedback: dict[str, Any] = {}
        if (
            package_policy is not None
            and package_plasticity_feedback_enabled
            and attempted_execution
        ):
            plasticity_feedback_rows += 1
            plasticity_feedback = package_policy.observe_feedback(
                verified_success=success,
                verifier="post_execution_verifier",
            )
            if plasticity_feedback.get("accepted") is True:
                plasticity_feedback_accepted_rows += 1
        rows.append(
            {
                "task_id": row.get("task_id"),
                "phase2ax_pair_id": row.get("phase2ax_pair_id"),
                "phase2ax_pair_member": row.get("phase2ax_pair_member"),
                "task_family": "phase2ay_counterfactual_slot_conditioned_execution",
                "selection_policy": selection_policy,
                "selected_slot": selected,
                "expected_slot": expected,
                "selected_repair_action": selected_action,
                "expected_repair_action": expected_action,
                "model_prediction_record_present": isinstance(prediction_record, dict),
                "model_command_slot_prediction": prediction_record.get("command_slot_prediction")
                if isinstance(prediction_record, dict)
                else None,
                "model_command_slot_label": prediction_record.get("command_slot_label")
                if isinstance(prediction_record, dict)
                else None,
                "model_command_slot_correct": prediction_record.get("command_slot_correct")
                if isinstance(prediction_record, dict)
                else None,
                "package_policy_loaded": package_policy is not None,
                "package_policy_metadata": package_policy_metadata,
                "package_policy_outputs": package_selection.get("policy_outputs")
                if package_selection
                else {},
                "package_low_level_debug_receptor_observed": package_selection.get(
                    "low_level_debug_receptor_observed"
                )
                is True,
                "package_qwen_called": package_selection.get("qwen_called") is True,
                "package_open_repair_authorized": package_selection.get(
                    "open_repair_authorized"
                )
                is True,
                "package_visible_state_source": package_selection.get(
                    "visible_state_source"
                )
                if package_selection
                else None,
                "package_nsi_reference_override": package_selection.get(
                    "nsi_reference_override"
                )
                if package_selection
                else {},
                "package_nsi_reference_mode": package_selection.get(
                    "nsi_reference_mode"
                )
                if package_selection
                else None,
                "package_runtime_evidence_label": package_selection.get(
                    "runtime_evidence_label"
                )
                if package_selection
                else None,
                "package_runtime_evidence_channel": package_selection.get(
                    "runtime_evidence_channel"
                )
                if package_selection
                else None,
                "package_runtime_evidence_control": package_selection.get(
                    "runtime_evidence_control"
                )
                if package_selection
                else None,
                "package_runtime_evidence_prompt_present": package_selection.get(
                    "runtime_evidence_prompt_present"
                )
                is True,
                "package_structured_runtime_evidence": package_selection.get(
                    "structured_runtime_evidence"
                )
                if package_selection
                else {},
                "package_plasticity_prediction": (
                    package_selection.get("policy_outputs", {}).get(
                        "plasticity_prediction"
                    )
                    if package_selection
                    else {}
                ),
                "package_plasticity_feedback": plasticity_feedback,
                "slot_selection_correct": slot_correct,
                "runtime_contract_allows_execution": contract_allowed,
                "execution_attempted": attempted_execution,
                "test_python": row_test_python,
                "test_python_source": row_test_python_source,
                "phase2z_execution_summary": execution_payload.get("summary"),
                "phase2z_patch_source": executed_row.get("patch_source"),
                "phase2z_patch_generator": executed_row.get("patch_generator"),
                "phase2z_patch_authorized": executed_row.get("patch_authorized"),
                "phase2z_symbolic_patch_failure": executed_row.get(
                    "symbolic_patch_failure"
                ),
                "phase2z_symbolic_patch_kinds": executed_row.get(
                    "symbolic_patch_kinds"
                )
                or [],
                "success": success,
                "full_task_success": success,
                "full_patch_correctness": executed_row.get("full_patch_correctness") is True,
                "full_test_pass_rate": executed_row.get("full_test_pass_rate", 0.0),
                "rollback_failure_restored": executed_row.get("rollback_failure_restored") is True,
                "unauthorized_write_count": int(executed_row.get("unauthorized_write_count") or 0),
                "false_completion": executed_row.get("false_completion") is True,
                "recorded_patch_artifact_used": executed_row.get("recorded_patch_artifact_used") is True,
                "recorded_patch_artifact_used_for_fault_injection": executed_row.get(
                    "recorded_patch_artifact_used_for_fault_injection"
                )
                is True,
                "claim_bearing_execution_evidence": executed_row.get(
                    "claim_bearing_execution_evidence"
                )
                is True,
                "oracle_trace_used": executed_row.get("oracle_trace_used") is True,
                "freeform_patch_generation": False,
                "sealed_feedback_used": False,
                "verification_state": verification_state,
                "stop_condition": stop_condition,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "claim_boundary": CLAIM_BOUNDARY,
                "artifact_paths": artifact_paths,
            }
        )

    _write_jsonl(output_jsonl, rows)
    if package_policy is not None and package_plasticity_memory_out is not None:
        package_policy.save_plasticity_memory(package_plasticity_memory_out)
    selection_correct = sum(1 for row in rows if row.get("slot_selection_correct") is True)
    attempted = sum(1 for row in rows if row.get("execution_attempted") is True)
    successes = sum(1 for row in rows if row.get("success") is True)
    return {
        "artifact_family": "phase2ay_counterfactual_slot_execution_runner",
        "selection_policy": selection_policy,
        "rows": len(rows),
        "slot_selection_correct": selection_correct,
        "slot_selection_accuracy": selection_correct / len(rows) if rows else 0.0,
        "execution_attempts": attempted,
        "execution_attempt_rate": attempted / len(rows) if rows else 0.0,
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "attempt_success_rate": successes / attempted if attempted else 0.0,
        "recorded_patch_artifact_used_rows": sum(
            1 for row in rows if row.get("recorded_patch_artifact_used") is True
        ),
        "recorded_patch_artifact_used_for_fault_injection_rows": sum(
            1
            for row in rows
            if row.get("recorded_patch_artifact_used_for_fault_injection") is True
        ),
        "claim_bearing_execution_evidence_rows": sum(
            1 for row in rows if row.get("claim_bearing_execution_evidence") is True
        ),
        "freeform_patch_generation_rows": sum(
            1 for row in rows if row.get("freeform_patch_generation") is True
        ),
        "sealed_feedback_used_rows": sum(
            1 for row in rows if row.get("sealed_feedback_used") is True
        ),
        "model_prediction_records_present_rows": sum(
            1 for row in rows if row.get("model_prediction_record_present") is True
        ),
        "package_policy_loaded_rows": sum(
            1 for row in rows if row.get("package_policy_loaded") is True
        ),
        "package_low_level_debug_receptor_observed_rows": sum(
            1
            for row in rows
            if row.get("package_low_level_debug_receptor_observed") is True
        ),
        "package_qwen_called_rows": sum(
            1 for row in rows if row.get("package_qwen_called") is True
        ),
        "package_open_repair_authorized_rows": sum(
            1 for row in rows if row.get("package_open_repair_authorized") is True
        ),
        "package_head_record_visible_state_rows": sum(
            1
            for row in rows
            if row.get("package_visible_state_source") == "phase2ax_head_record"
        ),
        "package_model_load_strategy": package_policy_metadata.get(
            "model_load_strategy"
        ),
        "package_offload_state_dict": package_policy_metadata.get(
            "offload_state_dict"
        ),
        "package_nsi_reference_mode": (
            package_nsi_reference_mode if package_policy is not None else None
        ),
        "package_nsi_reference_override_rows": sum(
            1 for row in rows if row.get("package_nsi_reference_override")
        ),
        "package_runtime_evidence_label": (
            package_runtime_evidence_label if package_policy is not None else None
        ),
        "package_runtime_evidence_channel": (
            package_runtime_evidence_channel if package_policy is not None else None
        ),
        "package_runtime_evidence_control": (
            package_runtime_evidence_control if package_policy is not None else None
        ),
        "package_structured_runtime_evidence_rows": sum(
            1
            for row in rows
            if row.get("package_runtime_evidence_channel")
            in {"structured_receptor", "dual"}
        ),
        "package_runtime_evidence_prompt_present_rows": sum(
            1
            for row in rows
            if row.get("package_runtime_evidence_prompt_present") is True
        ),
        "package_structural_probe_receptor_rows": sum(
            1
            for row in rows
            if (
                isinstance(row.get("package_structured_runtime_evidence"), dict)
                and row["package_structured_runtime_evidence"].get(
                    "structural_probe_hashes"
                )
            )
        ),
        "package_plasticity_feedback_enabled": package_plasticity_feedback_enabled,
        "package_plasticity_control": package_plasticity_control,
        "package_plasticity_feedback_rows": plasticity_feedback_rows,
        "package_plasticity_feedback_accepted_rows": plasticity_feedback_accepted_rows,
        "package_plasticity_memory_in": (
            str(Path(package_plasticity_memory_in))
            if package_plasticity_memory_in is not None
            else None
        ),
        "package_plasticity_memory_out": (
            str(Path(package_plasticity_memory_out))
            if package_plasticity_memory_out is not None
            else None
        ),
        "package_plasticity_memory_hit_rows": sum(
            1
            for row in rows
            if isinstance(row.get("package_plasticity_prediction"), dict)
            and row["package_plasticity_prediction"].get("memory_hit") is True
        ),
        "test_python_profiles": {
            source: sum(1 for row in rows if row.get("test_python_source") == source)
            for source in sorted(
                {
                    str(row.get("test_python_source"))
                    for row in rows
                    if row.get("test_python_source")
                }
            )
        },
        "test_python_map_json": (
            str(Path(test_python_map_json)) if test_python_map_json is not None else None
        ),
        "output_jsonl": str(Path(output_jsonl)),
        "artifact_root": str(artifacts),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase2AY slot-conditioned counterfactual repair execution smoke."
    )
    parser.add_argument("--phase2ax-tasks-jsonl", required=True)
    parser.add_argument("--full-postflight-json", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--prediction-records-json")
    parser.add_argument("--head-records-jsonl")
    parser.add_argument(
        "--selection-policy",
        choices=sorted(SUPPORTED_SELECTION_POLICIES),
        default="prior_runtime_resolver",
    )
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--test-python")
    parser.add_argument("--test-python-map-json")
    parser.add_argument(
        "--package-nsi-reference-mode",
        choices=sorted(SUPPORTED_NSI_REFERENCE_MODES),
        default="runtime_visible_override",
    )
    parser.add_argument(
        "--package-runtime-evidence-label",
        choices=sorted(SUPPORTED_RUNTIME_EVIDENCE_LABELS),
        default="Prior runtime evidence",
    )
    parser.add_argument(
        "--package-runtime-evidence-channel",
        choices=sorted(SUPPORTED_RUNTIME_EVIDENCE_CHANNELS),
        default="prompt_text",
    )
    parser.add_argument(
        "--package-runtime-evidence-control",
        choices=sorted(SUPPORTED_RUNTIME_EVIDENCE_CONTROLS),
        default="normal",
    )
    parser.add_argument("--package-plasticity-memory-in")
    parser.add_argument("--package-plasticity-memory-out")
    parser.add_argument("--package-plasticity-feedback-enabled", action="store_true")
    parser.add_argument(
        "--package-plasticity-control",
        choices=sorted(PLASTICITY_CONTROLS),
        default="normal",
    )
    args = parser.parse_args()
    report = run_phase2ay_counterfactual_slot_execution(
        phase2ax_tasks_jsonl=args.phase2ax_tasks_jsonl,
        full_postflight_json=args.full_postflight_json,
        dataset_root=args.dataset_root,
        clone_root=args.clone_root,
        package_path=args.package_path,
        output_jsonl=args.output_jsonl,
        artifact_root=args.artifact_root,
        selection_policy=args.selection_policy,
        prediction_records_json=args.prediction_records_json,
        head_records_jsonl=args.head_records_jsonl,
        max_rows=args.max_rows,
        timeout_seconds=args.timeout_seconds,
        test_python=args.test_python,
        test_python_map_json=args.test_python_map_json,
        package_nsi_reference_mode=args.package_nsi_reference_mode,
        package_runtime_evidence_label=args.package_runtime_evidence_label,
        package_runtime_evidence_channel=args.package_runtime_evidence_channel,
        package_runtime_evidence_control=args.package_runtime_evidence_control,
        package_plasticity_memory_in=args.package_plasticity_memory_in,
        package_plasticity_memory_out=args.package_plasticity_memory_out,
        package_plasticity_feedback_enabled=args.package_plasticity_feedback_enabled,
        package_plasticity_control=args.package_plasticity_control,
    )
    _write_json(args.summary_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["rows"] <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
