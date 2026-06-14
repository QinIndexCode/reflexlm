import json
from pathlib import Path

import reflexlm.cli.run_phase2ay_counterfactual_slot_execution as runner
from reflexlm.cli.audit_phase2ay_runtime_execution_eval import (
    audit_phase2ay_runtime_execution_eval,
)
from reflexlm.cli.run_phase2ay_counterfactual_slot_execution import (
    _test_python_for_row,
    run_phase2ay_counterfactual_slot_execution,
)
from reflexlm.cli.run_phase2z_public_structural_repair_execution import (
    _infer_missing_known_import_texts,
)
from reflexlm.llm.receptor_latent import runtime_command_identity_signal


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_phase2ay_test_python_map_selects_repo_isolated_interpreter() -> None:
    executable, source = _test_python_for_row(
        {"repo_origin": "https://github.com/pydantic/pydantic.git"},
        default_test_python="python-default",
        test_python_map={
            "repos": {"pydantic_pydantic": "python-pydantic"},
        },
    )

    assert executable == "python-pydantic"
    assert source == "repo_override:pydantic_pydantic"


def test_phase2ay_test_python_map_falls_back_to_cli_default() -> None:
    executable, source = _test_python_for_row(
        {"repo_origin": "https://github.com/pytest-dev/pluggy.git"},
        default_test_python="python-default",
        test_python_map={
            "repos": {"pydantic_pydantic": "python-pydantic"},
        },
    )

    assert executable == "python-default"
    assert source == "cli_default"


def test_phase2ay_package_selection_can_use_low_level_nsi_only() -> None:
    class FakePolicy:
        last_call = {"cortex_plan": {"command_slot": 0}, "qwen_called": True}

        def reset(self) -> None:
            pass

        def act(self, _state, *, nsi_reference_override=None):
            assert nsi_reference_override is None
            return None

    result = runner._select_with_package_native_head_state(
        policy=FakePolicy(),
        row=_row("a"),
        head_record=None,
        row_index=0,
        nsi_reference_mode="low_level_only",
    )

    assert result["nsi_reference_override"] is None
    assert result["nsi_reference_mode"] == "low_level_only"


def test_phase2ay_visible_state_supports_runtime_evidence_label_shift() -> None:
    prompt = runner._phase2ax_visible_state_prompt(
        _row("a"),
        None,
        runtime_evidence_label="Runtime-visible repair evidence",
    )

    assert "Runtime-visible repair evidence:" in prompt
    assert "Prior runtime evidence:" not in prompt

    shifted_head_prompt = runner._phase2ax_visible_state_prompt(
        _row("a"),
        {"state_prompt": "Prior runtime evidence:\n{}"},
        runtime_evidence_label="Runtime-visible repair evidence",
    )
    assert shifted_head_prompt == "Runtime-visible repair evidence:\n{}"


def test_phase2ay_structured_receptor_removes_runtime_evidence_from_prompt() -> None:
    row = _row("a")
    prompt = runner._phase2ax_visible_state_prompt(
        row,
        {
            "state_prompt": "\n\n".join(
                [
                    "Visible state",
                    'Prior runtime evidence:\n{"structural_probe_hashes": ["aaaabbbbcccc9999"]}',
                    'Runtime-visible contract:\n{"no_gold_hint": true}',
                ]
            )
        },
        runtime_evidence_payload=row["phase2ax_prior_runtime_evidence"],
        include_runtime_evidence=False,
    )

    assert "Prior runtime evidence:" not in prompt
    assert "Runtime-visible repair evidence:" not in prompt
    assert "aaaabbbbcccc9999" not in prompt
    assert "Runtime-visible contract:" in prompt


def test_phase2ay_structured_receptor_drives_low_level_identity_without_prompt_probe() -> None:
    observed = {}

    class FakePolicy:
        last_call = {}

        def reset(self) -> None:
            pass

        def act(self, state, *, nsi_reference_override=None):
            assert nsi_reference_override is None
            observed["state"] = state
            signal = runtime_command_identity_signal(state)
            slot_scores = [
                signal[f"command_identity_slot:{index}"] for index in range(4)
            ]
            self.last_call = {
                "cortex_plan": {"command_slot": slot_scores.index(max(slot_scores))},
                "qwen_called": True,
            }
            return None

    result = runner._select_with_package_native_head_state(
        policy=FakePolicy(),
        row=_row("a"),
        head_record=None,
        row_index=0,
        nsi_reference_mode="low_level_only",
        runtime_evidence_channel="structured_receptor",
    )

    state = observed["state"]
    assert result["selected_slot"] == 0
    assert result["runtime_evidence_prompt_present"] is False
    assert "aaaabbbbcccc9999" not in state.terminal.stdout_delta
    assert state.runtime_evidence.structural_probe_hashes == ["aaaabbbbcccc9999"]


def test_phase2ay_wrong_structured_receptor_rotates_probe_to_other_candidate() -> None:
    evidence = runner._controlled_runtime_evidence(
        _row("a"),
        [
            "apply structural_repair_aaaabbbbcccc",
            "apply structural_repair_dddd11112222",
        ],
        "wrong",
    )

    assert evidence["structural_probe_hashes"] == ["dddd11112222ffff"]


def _row(member: str = "a", *, sealed: bool = False) -> dict:
    expected = "structural_repair_aaaabbbbcccc" if member == "a" else "structural_repair_dddd11112222"
    return {
        "task_id": f"phase2ax:pair_00000:{member}",
        "phase2ax_pair_id": "phase2ax_pair_00000",
        "phase2ax_pair_member": member,
        "repo_origin": "https://github.com/Example/Repo.git",
        "repo_commit": "abc123",
        "artifact_paths": {"generated_tests": [f"artifacts/val/example/row_{member}/generated_test.py"]},
        "expected_repair_action": expected,
        "repair_candidates": [
            {"repair_action": "structural_repair_aaaabbbbcccc"},
            {"repair_action": "structural_repair_dddd11112222"},
        ],
        "phase2ax_prior_runtime_evidence": {
            "changed_files": ["src/mod.py"],
            "structural_probe_hashes": ["aaaabbbbcccc9999"]
            if member == "a"
            else ["dddd11112222eeee"],
            "repair_modes": ["mode"],
            "watched_files": ["phase2z_repair_tests/test_case.py"],
        },
        "runtime_visible_contract": {
            "no_freeform_patch_generation": True,
            "no_sealed_feedback": True,
            "no_gold_hint": True,
        },
        "sealed_feedback_used": sealed,
    }


def _postflight() -> dict:
    return {"passed": True, "ready_for_phase2ay_runtime_execution_eval": True}


def _fake_execution(**_kwargs) -> dict:
    return {
        "summary": {"rows": 1, "successes": 1},
        "row": {
            "success": True,
            "full_patch_correctness": True,
            "full_test_pass_rate": 1.0,
            "rollback_failure_restored": True,
            "unauthorized_write_count": 0,
            "false_completion": False,
            "recorded_patch_artifact_used": False,
            "recorded_patch_artifact_used_for_fault_injection": True,
            "claim_bearing_execution_evidence": True,
            "oracle_trace_used": False,
            "verification_state": "passed",
            "stop_condition": "verification_passed",
            "patch_source": "package_runtime_symbolic_structural_patch_proposal",
            "patch_generator": "bounded_symbolic_structural_patch_v1",
            "patch_authorized": True,
            "symbolic_patch_failure": None,
            "symbolic_patch_kinds": ["import_restoration"],
            "artifact_paths": {"patch": "patch.diff"},
        },
    }


def test_phase2ay_runner_executes_only_prior_runtime_selected_slots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_execute_selected_row", _fake_execution)

    report = run_phase2ay_counterfactual_slot_execution(
        phase2ax_tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", [_row("a"), _row("b")]),
        full_postflight_json=_write_json(tmp_path / "postflight.json", _postflight()),
        dataset_root=tmp_path / "dataset",
        clone_root=tmp_path / "clones",
        package_path=tmp_path / "pkg",
        output_jsonl=tmp_path / "out.jsonl",
        artifact_root=tmp_path / "artifacts",
        selection_policy="prior_runtime_resolver",
        max_rows=2,
    )

    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert report["slot_selection_accuracy"] == 1.0
    assert report["execution_attempts"] == 2
    assert report["attempt_success_rate"] == 1.0
    assert report["recorded_patch_artifact_used_rows"] == 0
    assert report["recorded_patch_artifact_used_for_fault_injection_rows"] == 2
    assert all(row["execution_attempted"] is True for row in rows)
    assert all(row["freeform_patch_generation"] is False for row in rows)
    assert all(
        row["phase2z_patch_source"]
        == "package_runtime_symbolic_structural_patch_proposal"
        for row in rows
    )
    assert all(row["phase2z_patch_authorized"] is True for row in rows)
    assert all(row["phase2z_symbolic_patch_kinds"] == ["import_restoration"] for row in rows)


def test_phase2ay_runner_wrong_cache_stops_before_execution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner,
        "_execute_selected_row",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not execute")),
    )

    report = run_phase2ay_counterfactual_slot_execution(
        phase2ax_tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", [_row("a"), _row("b")]),
        full_postflight_json=_write_json(tmp_path / "postflight.json", _postflight()),
        dataset_root=tmp_path / "dataset",
        clone_root=tmp_path / "clones",
        package_path=tmp_path / "pkg",
        output_jsonl=tmp_path / "out.jsonl",
        artifact_root=tmp_path / "artifacts",
        selection_policy="wrong_cache",
        max_rows=2,
    )

    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert report["slot_selection_accuracy"] == 0.0
    assert report["execution_attempts"] == 0
    assert report["success_rate"] == 0.0
    assert all(row["stop_condition"] == "slot_selection_failed_before_execution" for row in rows)


def test_phase2ay_runner_executes_model_prediction_records(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_execute_selected_row", _fake_execution)
    prediction_report = {
        "prediction_records": [
            {
                "episode_id": "phase2ax:pair_00000:a",
                "example_id": "phase2ax:pair_00000:a:val:phase2ax_counterfactual_repair",
                "command_slot_label": 0,
                "command_slot_prediction": 0,
                "command_slot_correct": True,
            },
            {
                "episode_id": "phase2ax:pair_00000:b",
                "example_id": "phase2ax:pair_00000:b:val:phase2ax_counterfactual_repair",
                "command_slot_label": 1,
                "command_slot_prediction": 1,
                "command_slot_correct": True,
            },
        ]
    }

    report = run_phase2ay_counterfactual_slot_execution(
        phase2ax_tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", [_row("a"), _row("b")]),
        full_postflight_json=_write_json(tmp_path / "postflight.json", _postflight()),
        dataset_root=tmp_path / "dataset",
        clone_root=tmp_path / "clones",
        package_path=tmp_path / "pkg",
        output_jsonl=tmp_path / "out.jsonl",
        artifact_root=tmp_path / "artifacts",
        selection_policy="model_prediction_records",
        prediction_records_json=_write_json(tmp_path / "predictions.json", prediction_report),
        max_rows=2,
    )

    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert report["selection_policy"] == "model_prediction_records"
    assert report["slot_selection_accuracy"] == 1.0
    assert report["model_prediction_records_present_rows"] == 2
    assert report["execution_attempts"] == 2
    assert [row["model_command_slot_prediction"] for row in rows] == [0, 1]


def test_phase2ay_runner_executes_package_loaded_native_head_selection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    loaded_paths: list[Path] = []

    class FakePackage:
        def __init__(self, package_path: Path) -> None:
            loaded_paths.append(Path(package_path))

        def metadata(self) -> dict:
            return {
                "model_load_strategy": "single_device",
                "offload_state_dict": True,
            }

    def fake_select_with_package_native_head_state(**kwargs) -> dict:
        row = kwargs["row"]
        assert kwargs["head_record"]["state_prompt"].startswith("visible")
        expected = 0 if str(row.get("task_id", "")).endswith(":a") else 1
        return {
            "selected_slot": expected,
            "policy_outputs": {
                "cortex_plan": {"command_slot": expected},
                "open_repair_head_outputs": {
                    "patch_proposal": 1,
                    "bounded_edit_scope": 1,
                    "rollback_safety": 1,
                },
            },
            "low_level_debug_receptor_observed": True,
            "qwen_called": True,
            "open_repair_authorized": True,
            "visible_state_source": "phase2ax_head_record",
        }

    monkeypatch.setattr(runner, "NativeNervousPolicyPackage", FakePackage)
    monkeypatch.setattr(
        runner,
        "_select_with_package_native_head_state",
        fake_select_with_package_native_head_state,
    )
    monkeypatch.setattr(runner, "_execute_selected_row", _fake_execution)
    head_rows = [
        {
            "episode_id": "phase2ax:pair_00000:a",
            "state_prompt": "visible a",
            "candidate_commands": ["cmd0", "cmd1"],
        },
        {
            "episode_id": "phase2ax:pair_00000:b",
            "state_prompt": "visible b",
            "candidate_commands": ["cmd0", "cmd1"],
        },
    ]

    package_path = tmp_path / "pkg"
    report = run_phase2ay_counterfactual_slot_execution(
        phase2ax_tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", [_row("a"), _row("b")]),
        full_postflight_json=_write_json(tmp_path / "postflight.json", _postflight()),
        dataset_root=tmp_path / "dataset",
        clone_root=tmp_path / "clones",
        package_path=package_path,
        output_jsonl=tmp_path / "out.jsonl",
        artifact_root=tmp_path / "artifacts",
        selection_policy="package_loaded_native_head",
        head_records_jsonl=_write_jsonl(tmp_path / "head.jsonl", head_rows),
        max_rows=2,
    )

    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert loaded_paths == [package_path]
    assert report["selection_policy"] == "package_loaded_native_head"
    assert report["slot_selection_accuracy"] == 1.0
    assert report["execution_attempts"] == 2
    assert report["package_policy_loaded_rows"] == 2
    assert report["package_qwen_called_rows"] == 2
    assert report["package_low_level_debug_receptor_observed_rows"] == 2
    assert report["package_head_record_visible_state_rows"] == 2
    assert report["package_model_load_strategy"] == "single_device"
    assert report["package_offload_state_dict"] is True
    assert all(row["package_policy_loaded"] is True for row in rows)
    assert all(
        row["package_policy_metadata"]["model_load_strategy"] == "single_device"
        for row in rows
    )
    assert [row["selected_slot"] for row in rows] == [0, 1]


def test_phase2ay_runner_blocks_sealed_feedback_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner,
        "_execute_selected_row",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not execute")),
    )

    report = run_phase2ay_counterfactual_slot_execution(
        phase2ax_tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", [_row("a", sealed=True)]),
        full_postflight_json=_write_json(tmp_path / "postflight.json", _postflight()),
        dataset_root=tmp_path / "dataset",
        clone_root=tmp_path / "clones",
        package_path=tmp_path / "pkg",
        output_jsonl=tmp_path / "out.jsonl",
        artifact_root=tmp_path / "artifacts",
        selection_policy="prior_runtime_resolver",
        max_rows=1,
    )

    row = json.loads((tmp_path / "out.jsonl").read_text())
    assert report["execution_attempts"] == 0
    assert row["slot_selection_correct"] is True
    assert row["runtime_contract_allows_execution"] is False
    assert row["stop_condition"] == "runtime_contract_blocks_execution"


def _summary(policy: str, *, prior_ok: bool = True) -> dict:
    if policy == "prior_runtime_resolver":
        return {
            "selection_policy": policy,
            "rows": 2,
            "slot_selection_accuracy": 1.0 if prior_ok else 0.5,
            "execution_attempts": 2 if prior_ok else 1,
            "success_rate": 1.0 if prior_ok else 0.5,
            "attempt_success_rate": 1.0 if prior_ok else 0.5,
            "recorded_patch_artifact_used_rows": 0,
            "recorded_patch_artifact_used_for_fault_injection_rows": 2 if prior_ok else 1,
            "claim_bearing_execution_evidence_rows": 2 if prior_ok else 1,
            "freeform_patch_generation_rows": 0,
            "sealed_feedback_used_rows": 0,
        }
    return {
        "selection_policy": policy,
        "rows": 2,
        "slot_selection_accuracy": 0.0,
        "execution_attempts": 0,
        "success_rate": 0.0,
        "attempt_success_rate": 0.0,
        "recorded_patch_artifact_used_rows": 0,
        "recorded_patch_artifact_used_for_fault_injection_rows": 0,
        "claim_bearing_execution_evidence_rows": 0,
        "freeform_patch_generation_rows": 0,
        "sealed_feedback_used_rows": 0,
    }


def test_phase2ay_audit_accepts_slot_conditioned_execution_but_blocks_claim_upgrade(
    tmp_path: Path,
) -> None:
    report = audit_phase2ay_runtime_execution_eval(
        prior_summary_json=_write_json(tmp_path / "prior.json", _summary("prior_runtime_resolver")),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _summary("wrong_cache")),
        full_postflight_json=_write_json(tmp_path / "postflight.json", _postflight()),
    )

    assert report["passed"] is True
    assert report["ready_for_phase2ay_expanded_runtime_execution_eval"] is True
    assert report["ready_for_phase2ax_package"] is False
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert report["checks"]["wrong_cache_no_execution_attempts"] is True


def test_phase2ay_audit_accepts_model_prediction_execution_but_blocks_package(
    tmp_path: Path,
) -> None:
    prior = _summary("prior_runtime_resolver")
    prior["selection_policy"] = "model_prediction_records"
    prior["model_prediction_records_present_rows"] = 2
    report = audit_phase2ay_runtime_execution_eval(
        prior_summary_json=_write_json(tmp_path / "prior.json", prior),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _summary("wrong_cache")),
        full_postflight_json=_write_json(tmp_path / "postflight.json", _postflight()),
        expected_prior_selection_policy="model_prediction_records",
    )

    assert report["passed"] is True
    assert report["ready_for_phase2ay_model_prediction_execution_eval"] is True
    assert report["ready_for_phase2ax_package"] is False
    assert report["checks"]["model_prediction_records_present_when_required"] is True


def test_phase2ay_audit_rejects_weak_prior_execution_delta(tmp_path: Path) -> None:
    report = audit_phase2ay_runtime_execution_eval(
        prior_summary_json=_write_json(
            tmp_path / "prior.json", _summary("prior_runtime_resolver", prior_ok=False)
        ),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _summary("wrong_cache")),
        full_postflight_json=_write_json(tmp_path / "postflight.json", _postflight()),
    )

    assert report["passed"] is False
    assert report["checks"]["prior_slot_accuracy_gate"] is False
    assert report["checks"]["prior_attempt_success_rate_gate"] is False


def test_phase2ay_symbolic_import_inference_covers_logging_runtime_nameerror() -> None:
    imports = _infer_missing_known_import_texts("LOG = logging.getLogger(__name__)\n")

    assert "import logging" in imports
