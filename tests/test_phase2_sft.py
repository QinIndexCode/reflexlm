import json
from pathlib import Path

import pytest

from reflexlm.data.jsonl import read_jsonl
from reflexlm.data.tasks import materialize_phase1_dataset
from reflexlm.baselines.text_policies import HuggingFaceJSONPolicy
from reflexlm.cli.qwen_tiny_overfit import _payload_uses_allowlisted_slots
from reflexlm.llm.prompts import SynapseSummary, build_phase2_user_prompt
from reflexlm.llm.sft import materialize_sft_corpus
from reflexlm.llm.qlora import _tokenize_example
from reflexlm.schema import ActionDecision, ActionType


def test_phase2_sft_manifest_and_prompt_safety(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=5)
    output_dir = tmp_path / "phase2_sft"
    manifest = materialize_sft_corpus(
        train_jsonl=dataset_dir / "train.jsonl",
        val_jsonl=dataset_dir / "val.jsonl",
        output_dir=output_dir,
        prompt_styles=["prompt_only"],
    )
    shared_train = output_dir / "prompt_only" / "shared" / "train.jsonl"
    assert shared_train.exists()
    assert manifest["styles"]["prompt_only"]["shared"]["train"] > 0
    first_row = json.loads(shared_train.read_text(encoding="utf-8").splitlines()[0])
    assert "recovery_hint=" not in first_row["user_prompt"]
    assert "Return only JSON." in first_row["user_prompt"]
    assert "target_text" in first_row


def test_nsi_state_v2_prompt_excludes_hidden_fields_and_exposes_motor_schema(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=11)
    record = next(row for row in read_jsonl(dataset_dir / "train.jsonl") if row.goal.recovery_hint)
    prompt = build_phase2_user_prompt(
        record.state,
        prompt_style="nsi_state_v2",
        synapse_summary=SynapseSummary(
            route_name="terminal_cortex",
            salience=0.7,
            risk=0.2,
            prediction_error=0.5,
            confidence=0.9,
            reflex_action="READ_STDERR",
        ),
    )
    assert "recovery_hint" not in prompt
    assert str(record.goal.recovery_hint) not in prompt
    assert "task=" not in prompt
    assert "Legal action mask:" in prompt
    assert "reflex_action=READ_STDERR" in prompt
    assert "receptor -> synaptic state -> reflex layer" in prompt


def test_nsi_state_v2_sft_requires_synapse_checkpoint(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=13)
    with pytest.raises(ValueError, match="nsi_state_v2"):
        materialize_sft_corpus(
            train_jsonl=dataset_dir / "train.jsonl",
            val_jsonl=dataset_dir / "val.jsonl",
            output_dir=tmp_path / "phase2_sft",
            prompt_styles=["nsi_state_v2"],
        )


def test_llm_policy_falls_back_on_non_candidate_command(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=17)
    record = next(row for row in read_jsonl(dataset_dir / "train.jsonl") if row.state.goal.command_allowlist)
    policy = HuggingFaceJSONPolicy.__new__(HuggingFaceJSONPolicy)
    invalid = ActionDecision(type=ActionType.RUN_COMMAND, command="python made_up.py")
    resolved = policy._validate_or_fallback_action(invalid, record.state)
    assert resolved.type == ActionType.WAIT
    assert resolved.reason == "llm_invalid_command_candidate"


def test_llm_policy_normalizes_general_react_action_aliases() -> None:
    policy = HuggingFaceJSONPolicy.__new__(HuggingFaceJSONPolicy)

    rerun = policy._parse_action(
        '{"action":"rerun_selected_test","command":"python -m pytest -q tests/test_x.py::test_y","file_target":"tests/test_x.py"}'
    )
    react_with_command = policy._parse_action(
        '{"action":"react","command":"python -m pytest -q tests/test_x.py::test_y","file_target":"tests/test_x.py"}'
    )
    inspect_error = policy._parse_action(
        '{"action":"react_to_failure","command":null,"file_target":null}'
    )
    inspect_file = policy._parse_action(
        '{"action":"inspect_source_file","command":null,"file_target":"src/example.py"}'
    )

    assert rerun.type == ActionType.RUN_COMMAND
    assert react_with_command.type == ActionType.RUN_COMMAND
    assert inspect_error.type == ActionType.READ_STDERR
    assert inspect_file.type == ActionType.READ_FILE


def test_llm_policy_bounded_history_keeps_only_recent_steps() -> None:
    policy = HuggingFaceJSONPolicy.__new__(HuggingFaceJSONPolicy)
    policy.maintain_history = True
    policy.max_history_steps = 2
    policy.history = []

    for index in range(3):
        policy._append_history(
            state_text=f"state-{index}",
            response=f"response-{index}",
            action=ActionDecision(type=ActionType.WAIT),
        )

    assert [row["state"] for row in policy.history] == ["state-1", "state-2"]
    assert "state-0" not in policy._history_text()


def test_tiny_overfit_probe_checks_allowlisted_slots(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=19)
    record = next(row for row in read_jsonl(dataset_dir / "train.jsonl") if row.state.goal.command_allowlist)
    prompt = build_phase2_user_prompt(
        record.state,
        prompt_style="nsi_state_v2",
        synapse_summary=SynapseSummary(
            route_name="debug_cortex",
            salience=0.9,
            risk=0.1,
            prediction_error=0.2,
            confidence=0.8,
        ),
    )
    row = {"user_prompt": prompt}
    assert _payload_uses_allowlisted_slots(
        row,
        {"action": "RUN_COMMAND", "command": record.state.goal.command_allowlist[0], "file_target": None},
    )
    assert not _payload_uses_allowlisted_slots(
        row,
        {"action": "RUN_COMMAND", "command": "python made_up.py", "file_target": None},
    )


def test_qlora_tokenizer_reserves_target_tokens() -> None:
    class FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "\n".join(message["content"] for message in messages)

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(1, len(text.split()) + 1))}

    row = {
        "system_prompt": "system " * 10,
        "user_prompt": "user " * 200,
        "target_text": '{"action":"WAIT","command":null,"file_target":null}',
    }
    encoded = _tokenize_example(FakeTokenizer(), row, max_length=32)
    labels = encoded["labels"].tolist()
    assert any(token != -100 for token in labels)
