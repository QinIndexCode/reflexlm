import json
from pathlib import Path

from reflexlm.cli.generate_debug_cortex_challenge import build_debug_cortex_challenge
from reflexlm.data.tasks import materialize_phase1_dataset
from reflexlm.llm.head_dataset import materialize_phase2c_head_corpus
from reflexlm.llm.receptor_latent import COMMAND_IDENTITY_LATENT_FIELDS, DEBUG_ACTION_STAGE_ORDER
from reflexlm.schema import ActionType, InternalTarget, RouteName, TaskType


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_phase2c_head_dataset_has_no_json_target_or_hidden_hint_prompt(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=31)
    output_dir = tmp_path / "phase2c_heads"

    manifest = materialize_phase2c_head_corpus(
        train_jsonl=dataset_dir / "train.jsonl",
        val_jsonl=dataset_dir / "val.jsonl",
        test_jsonl=dataset_dir / "test.jsonl",
        output_dir=output_dir,
    )

    assert manifest["json_text_target"] is False
    assert manifest["leakage_audit"]["passed"] is True
    rows = _read_rows(output_dir / "train.jsonl")
    first = rows[0]
    assert "target_text" not in first
    assert "state_prompt" in first
    assert "Return only JSON" not in first["state_prompt"]
    assert "recovery_hint=" not in first["state_prompt"]
    assert "\ntask=" not in first["state_prompt"]
    assert first["prompt_style"] == "phase2c_head_state_v1"


def test_phase2c_head_dataset_records_nonlabel_command_identity_latent(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    extra_train_dir = tmp_path / "phase2i_semantic_train"
    materialize_phase1_dataset(dataset_dir, seed=32)
    build_debug_cortex_challenge(
        extra_train_dir,
        profile="phase2i_semantic_train",
        episodes_per_scenario=1,
    )
    output_dir = tmp_path / "phase2c_heads"

    materialize_phase2c_head_corpus(
        train_jsonl=dataset_dir / "train.jsonl",
        val_jsonl=dataset_dir / "val.jsonl",
        extra_train_jsonls=[extra_train_dir / "challenge.jsonl"],
        output_dir=output_dir,
    )

    rows = _read_rows(output_dir / "train.jsonl")
    command_rows = [
        row
        for row in rows
        if row["action_type"] == ActionType.RUN_COMMAND.value
        and row["command_slot"] != -100
    ]

    assert command_rows
    reference = command_rows[0]["nsi_reference"]
    assert reference["receptor_failure_signal"]
    assert reference["debug_action_stage"] in DEBUG_ACTION_STAGE_ORDER
    assert set(COMMAND_IDENTITY_LATENT_FIELDS).issubset(reference)
    assert "debug_action_stage=" in command_rows[0]["state_prompt"]


def test_phase2c_test_failure_routes_to_debug_cortex_target(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=37)
    output_dir = tmp_path / "phase2c_heads"
    materialize_phase2c_head_corpus(
        train_jsonl=dataset_dir / "train.jsonl",
        val_jsonl=dataset_dir / "val.jsonl",
        output_dir=output_dir,
    )

    rows = _read_rows(output_dir / "train.jsonl") + _read_rows(output_dir / "val.jsonl")
    debug_rows = [row for row in rows if row["task_type"] == TaskType.TEST_FAILURE.value]
    assert debug_rows
    assert {row["internal_target"] for row in debug_rows} == {
        InternalTarget.ESCALATE_TO_DEBUG_CORTEX.value
    }
    assert {row["head_scope"] for row in debug_rows} == {"debug_cortex"}
    assert {row["route_name"] for row in debug_rows} == {RouteName.DEBUG.value}
    assert any(row["action_type"] == ActionType.RUN_COMMAND.value for row in debug_rows)


def test_phase2c_external_file_change_uses_refresh_receptor_labels(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=41)
    output_dir = tmp_path / "phase2c_heads"
    materialize_phase2c_head_corpus(
        train_jsonl=dataset_dir / "train.jsonl",
        val_jsonl=dataset_dir / "val.jsonl",
        output_dir=output_dir,
    )

    rows = _read_rows(output_dir / "train.jsonl") + _read_rows(output_dir / "val.jsonl")
    refresh_rows = [
        row
        for row in rows
        if "stale_state_refresh_receptor" in row["runtime_overrides"]
    ]
    assert refresh_rows
    assert {row["task_type"] for row in refresh_rows} == {TaskType.FILE_CHANGE.value}
    assert {row["internal_target"] for row in refresh_rows} == {InternalTarget.REFLEX_MOTOR.value}
    assert {row["route_name"] for row in refresh_rows} == {RouteName.FILE.value}
    assert {row["action_type"] for row in refresh_rows} == {ActionType.REFRESH_STATE.value}


def test_phase2c_dangerous_action_uses_inhibition_labels(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=43)
    output_dir = tmp_path / "phase2c_heads"
    materialize_phase2c_head_corpus(
        train_jsonl=dataset_dir / "train.jsonl",
        val_jsonl=dataset_dir / "val.jsonl",
        output_dir=output_dir,
    )

    rows = _read_rows(output_dir / "train.jsonl") + _read_rows(output_dir / "val.jsonl")
    safety_rows = [row for row in rows if row["task_type"] == TaskType.DANGEROUS_ACTION.value]
    assert safety_rows
    assert {row["internal_target"] for row in safety_rows} == {InternalTarget.INHIBIT.value}
    assert {row["head_scope"] for row in safety_rows} == {"inhibition"}
    assert {row["route_name"] for row in safety_rows} == {RouteName.SAFETY.value}
    assert {row["action_type"] for row in safety_rows} == {ActionType.BLOCK.value}
    assert all(row["inhibition_target"] == 1.0 for row in safety_rows)


def test_phase2c_head_dataset_accepts_disjoint_transition_extras(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=47)
    extra_train_dir = tmp_path / "debug_transition_train"
    extra_val_dir = tmp_path / "debug_transition_val"
    build_debug_cortex_challenge(
        extra_train_dir,
        profile="debug_transition_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        extra_val_dir,
        profile="debug_transition_val",
        episodes_per_scenario=1,
    )
    output_dir = tmp_path / "phase2c_heads"

    manifest = materialize_phase2c_head_corpus(
        train_jsonl=dataset_dir / "train.jsonl",
        val_jsonl=dataset_dir / "val.jsonl",
        test_jsonl=dataset_dir / "test.jsonl",
        extra_train_jsonls=[extra_train_dir / "challenge.jsonl"],
        extra_val_jsonls=[extra_val_dir / "challenge.jsonl"],
        output_dir=output_dir,
    )

    assert manifest["splits"]["train"]["extra_source_jsonls"]
    assert manifest["splits"]["val"]["extra_source_jsonls"]
    assert not manifest["splits"]["test"]["extra_source_jsonls"]
    train_rows = _read_rows(output_dir / "train.jsonl")
    val_rows = _read_rows(output_dir / "val.jsonl")
    test_rows = _read_rows(output_dir / "test.jsonl")
    assert any(
        row["episode_id"].startswith("extra_train_0_debug_transition_train__")
        for row in train_rows
    )
    assert any(
        row["episode_id"].startswith("extra_val_0_debug_transition_val__")
        for row in val_rows
    )
    assert all(not row["episode_id"].startswith("extra_") for row in test_rows)
    assert any(
        row["head_scope"] == "debug_cortex"
        and row["action_type"] == ActionType.RUN_COMMAND.value
        and row["command_intent"] == "test_rerun"
        for row in train_rows
    )
    assert any(
        row["head_scope"] == "debug_cortex"
        and row["action_type"] == ActionType.RUN_COMMAND.value
        and row["command_intent"] == "test_rerun"
        for row in val_rows
    )


def test_phase2f_latent_profiles_compress_cortex_failure_text(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "phase1"
    materialize_phase1_dataset(dataset_dir, seed=53)
    extra_train_dir = tmp_path / "phase2f_latent_train"
    build_debug_cortex_challenge(
        extra_train_dir,
        profile="phase2f_latent_train",
        episodes_per_scenario=1,
    )
    output_dir = tmp_path / "phase2f_heads"

    materialize_phase2c_head_corpus(
        train_jsonl=dataset_dir / "train.jsonl",
        val_jsonl=dataset_dir / "val.jsonl",
        extra_train_jsonls=[extra_train_dir / "challenge.jsonl"],
        output_dir=output_dir,
    )

    latent_rows = [
        row
        for row in _read_rows(output_dir / "train.jsonl")
        if row["episode_id"].startswith("extra_train_0_phase2f_latent_train__")
    ]
    assert latent_rows
    assert any("failure_signal=latent_required" in row["state_prompt"] for row in latent_rows)
    compressed_rows = [
        row
        for row in latent_rows
        if "stdout_delta=<compressed_failure_signal>" in row["state_prompt"]
        or "stderr_delta=<compressed_failure_signal>" in row["state_prompt"]
    ]
    assert compressed_rows
    compressed_text = "\n".join(row["state_prompt"] for row in compressed_rows).lower()
    assert "modulenotfounderror" not in compressed_text
    assert "snapshot mismatch" not in compressed_text
    assert "assertionerror" not in compressed_text
