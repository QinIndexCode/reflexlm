import json
from pathlib import Path

from reflexlm.cli.audit_phase2i_data_health import _canonical_rows_sha256
from reflexlm.cli.audit_phase2j_data_health import build_phase2j_data_health_audit
from reflexlm.cli.generate_debug_cortex_challenge import build_debug_cortex_challenge
from reflexlm.llm.native_head_training import _balance_debug_command_intent_rows


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows),
        encoding="utf-8",
    )
    return path


def _head_row(split: str, slot: int) -> dict:
    commands = [
        f"python -m pytest -q tests/phase2j_{split}/test_slot_{slot}_{index}.py::test_case_{index}"
        for index in range(4)
    ]
    return {
        "example_id": f"{split}-{slot}",
        "episode_id": f"{split}-{slot}",
        "task_type": "test_failure_reflex",
        "head_scope": "debug_cortex",
        "action_type": "RUN_COMMAND",
        "command": commands[slot],
        "command_slot": slot,
        "file_slot": -100,
        "command_intent": "test_rerun",
        "candidate_commands": commands,
        "candidate_files": [],
        "state_prompt": f"visible Phase2J receptor source evidence for slot {slot}",
        "nsi_reference": {
            f"command_identity_slot:{index}": 1.0 if index == slot else 0.0
            for index in range(4)
        },
    }


def _hard_head_row(
    split: str,
    slot: int,
    *,
    source_overlap_easy: bool = False,
    candidate_count: int = 4,
) -> dict:
    suffixes = ["cache_refresh", "target_resolution", "payload_sidecar", "delta_gate"]
    commands = [
        (
            "python -m pytest -q "
            f"tests/phase2j_hard_{split}/test_command_identity.py::test_{suffix}"
        )
        for suffix in suffixes
    ]
    state_prompt = (
        "stdout_delta=Source inspected: semantic disambiguation required. "
        "command_identity_tokens=<redacted>."
    )
    if source_overlap_easy:
        state_prompt = (
            f"stdout_delta=Selected visible terms: {suffixes[slot].replace('_', ' ')}.\n"
            "command_identity_tokens=<redacted>."
        )
    commands = commands[:candidate_count]
    return {
        "example_id": f"{split}-hard-{slot}",
        "episode_id": f"{split}-hard-{slot}",
        "task_type": "test_failure_reflex",
        "head_scope": "debug_cortex",
        "action_type": "RUN_COMMAND",
        "command": commands[slot],
        "command_slot": slot,
        "file_slot": -100,
        "command_intent": "test_rerun",
        "candidate_commands": commands,
        "candidate_files": [],
        "state_prompt": state_prompt,
        "nsi_reference": {
            f"command_identity_slot:{index}": 1.0 if index == slot else 0.0
            for index in range(4)
        },
    }


def test_phase2j_data_health_accepts_disjoint_balanced_nonsealed_sets(tmp_path: Path) -> None:
    train_dir = tmp_path / "phase2j_train"
    val_dir = tmp_path / "phase2j_val"
    build_debug_cortex_challenge(train_dir, profile="phase2j_semantic_train", episodes_per_scenario=1)
    build_debug_cortex_challenge(val_dir, profile="phase2j_semantic_val", episodes_per_scenario=1)
    head_train = _write_jsonl(
        tmp_path / "head_train.jsonl",
        [_head_row("train", slot) for slot in range(4)],
    )
    head_val = _write_jsonl(
        tmp_path / "head_val.jsonl",
        [_head_row("val", slot) for slot in range(4)],
    )

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_semantic_train": train_dir / "challenge.jsonl",
            "phase2j_semantic_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_semantic_nn=1.01,
    )

    assert report["passed"] is True
    assert report["sealed_usage"]["sealed_splits_used_for_training"] is False
    assert report["checks"]["phase2j_effective_split_hashes_present"] is True
    assert report["checks"]["phase2j_train_val_command_intent_coverage"] is True
    assert len(report["effective_split_hashes"]["phase2j_head_train"]) == 64
    assert report["head_splits"]["phase2j_head_train"]["command_slot_max_share"] == 0.25


def test_phase2j_data_health_rejects_missing_train_intent(tmp_path: Path) -> None:
    train_dir = tmp_path / "phase2j_train"
    val_dir = tmp_path / "phase2j_val"
    build_debug_cortex_challenge(train_dir, profile="phase2j_semantic_train", episodes_per_scenario=1)
    build_debug_cortex_challenge(val_dir, profile="phase2j_semantic_val", episodes_per_scenario=1)
    train_row = _head_row("train", 0)
    train_row["command_intent"] = "dependency_install"
    train_row["command"] = "python -m pip install -r requirements.txt"
    train_row["candidate_commands"] = ["python -m pip install -r requirements.txt"]
    train_row["command_slot"] = 0
    val_row = _head_row("val", 0)
    head_train = _write_jsonl(tmp_path / "head_train.jsonl", [train_row])
    head_val = _write_jsonl(tmp_path / "head_val.jsonl", [val_row])

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_semantic_train": train_dir / "challenge.jsonl",
            "phase2j_semantic_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=1,
        max_semantic_nn=1.01,
    )

    assert report["passed"] is False
    assert report["checks"]["phase2j_train_val_command_intent_coverage"] is False
    assert report["train_val_command_intent_gap"]["missing_train_command_intents"] == [
        "test_rerun"
    ]


def test_phase2j_source_overlap_hard_data_health_requires_low_baseline_and_identity_signal(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "phase2j_hard_train"
    val_dir = tmp_path / "phase2j_hard_val"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2j_source_overlap_hard_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2j_source_overlap_hard_val",
        episodes_per_scenario=1,
    )
    head_train = _write_jsonl(
        tmp_path / "head_train.jsonl",
        [_hard_head_row("train", slot) for slot in range(4)],
    )
    head_val = _write_jsonl(
        tmp_path / "head_val.jsonl",
        [_hard_head_row("val", slot) for slot in range(4)],
    )

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_source_overlap_hard_train": train_dir / "challenge.jsonl",
            "phase2j_source_overlap_hard_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_semantic_nn=1.01,
        source_overlap_hard=True,
        max_source_overlap_val_accuracy=0.30,
    )

    assert report["passed"] is True
    assert report["checks"]["phase2j_source_overlap_hard_val_baseline_below_threshold"] is True
    assert report["checks"]["phase2j_source_overlap_hard_identity_signal_present"] is True
    assert report["checks"]["phase2j_source_overlap_hard_prompt_redacts_identity_sidecar"] is True
    assert (
        report["source_overlap_hard"]["phase2j_head_val"]["source_overlap_accuracy"]
        == 0.25
    )
    assert report["source_overlap_hard"]["phase2j_head_val"]["identity_accuracy"] == 1.0


def test_phase2j_data_health_rejects_single_candidate_count_band_when_required(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "phase2j_hard_train"
    val_dir = tmp_path / "phase2j_hard_val"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2j_source_overlap_hard_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2j_source_overlap_hard_val",
        episodes_per_scenario=1,
    )
    head_train = _write_jsonl(
        tmp_path / "head_train.jsonl",
        [_hard_head_row("train", slot) for slot in range(4)],
    )
    head_val = _write_jsonl(
        tmp_path / "head_val.jsonl",
        [_hard_head_row("val", slot) for slot in range(4)],
    )

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_source_overlap_hard_train": train_dir / "challenge.jsonl",
            "phase2j_source_overlap_hard_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_semantic_nn=1.01,
        source_overlap_hard=True,
        max_source_overlap_val_accuracy=0.30,
        min_command_candidate_count_bands=3,
        required_command_candidate_counts=[2, 3, 4],
    )

    assert report["passed"] is False
    assert report["checks"]["phase2j_head_train_command_candidate_count_band_coverage"] is False
    assert report["checks"]["phase2j_head_val_command_candidate_count_band_coverage"] is False
    assert report["checks"]["phase2j_head_train_required_command_candidate_counts_present"] is False
    assert report["thresholds"]["required_command_candidate_counts"] == ["2", "3", "4"]


def test_phase2j_data_health_accepts_actiongate_candidate_count_pressure(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "phase2j_actiongate_train"
    val_dir = tmp_path / "phase2j_actiongate_val"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2j_source_overlap_hard_actiongate_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2j_source_overlap_hard_actiongate_val",
        episodes_per_scenario=1,
    )
    rows = [
        (1, 2),
        (1, 2),
        (0, 2),
        (2, 3),
        (2, 3),
        (1, 3),
        (3, 4),
        (2, 4),
        (1, 4),
        (0, 4),
    ]
    head_train = _write_jsonl(
        tmp_path / "head_train.jsonl",
        [_hard_head_row("train", slot, candidate_count=count) for slot, count in rows],
    )
    head_val = _write_jsonl(
        tmp_path / "head_val.jsonl",
        [_hard_head_row("val", slot, candidate_count=count) for slot, count in rows],
    )

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_source_overlap_hard_actiongate_train": train_dir / "challenge.jsonl",
            "phase2j_source_overlap_hard_actiongate_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_semantic_nn=1.01,
        source_overlap_hard=True,
        max_source_overlap_val_accuracy=0.30,
        min_command_candidate_count_bands=3,
        required_command_candidate_counts=[2, 3, 4],
    )

    assert report["passed"] is True
    assert report["checks"]["phase2j_head_train_command_candidate_count_band_coverage"] is True
    assert report["checks"]["phase2j_head_val_command_candidate_count_band_coverage"] is True
    assert report["checks"]["phase2j_head_train_required_command_candidate_counts_present"] is True


def test_phase2j_data_health_balances_train_only_to_match_training_hashes(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "phase2j_train"
    val_dir = tmp_path / "phase2j_val"
    build_debug_cortex_challenge(train_dir, profile="phase2j_semantic_train", episodes_per_scenario=1)
    build_debug_cortex_challenge(val_dir, profile="phase2j_semantic_val", episodes_per_scenario=1)
    rows = [
        _head_row("hash", 0),
        _head_row("hash", 1),
        _head_row("hash", 2),
    ]
    rows[0]["command"] = "python -m pip install -r requirements.txt"
    rows[0]["candidate_commands"][0] = rows[0]["command"]
    head_train = _write_jsonl(tmp_path / "head_train.jsonl", rows)
    head_val = _write_jsonl(tmp_path / "head_val.jsonl", rows)

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_semantic_train": train_dir / "challenge.jsonl",
            "phase2j_semantic_val": val_dir / "challenge.jsonl",
        },
        balance_debug_command_intents=True,
        min_val_target_commands=1,
        max_command_slot_share=1.0,
        max_semantic_nn=1.01,
    )

    assert report["effective_split_hashes"]["phase2j_head_train"] == _canonical_rows_sha256(
        _balance_debug_command_intent_rows(rows)
    )
    assert report["effective_split_hashes"]["phase2j_head_val"] == _canonical_rows_sha256(rows)


def test_phase2j_data_health_requires_synapse_reference_when_requested(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "phase2j_train"
    val_dir = tmp_path / "phase2j_val"
    build_debug_cortex_challenge(train_dir, profile="phase2j_semantic_train", episodes_per_scenario=1)
    build_debug_cortex_challenge(val_dir, profile="phase2j_semantic_val", episodes_per_scenario=1)
    rows = [_head_row("synapse", slot) for slot in range(4)]
    for row in rows:
        row["nsi_reference"].update(
            {
                "reflex_action": "READ_STDERR",
                "salience": 1.0,
                "risk": 0.2,
                "prediction_error": 0.1,
                "confidence": 0.8,
            }
        )
    head_train = _write_jsonl(tmp_path / "head_train.jsonl", rows)
    head_val = _write_jsonl(tmp_path / "head_val.jsonl", rows)

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_semantic_train": train_dir / "challenge.jsonl",
            "phase2j_semantic_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_train_val_target_overlap=1.0,
        max_semantic_nn=1.01,
        require_synapse_reference=True,
    )

    assert report["passed"] is True
    assert report["checks"]["phase2j_head_train_synapse_reference_present"] is True
    assert report["checks"]["phase2j_head_val_synapse_reference_present"] is True
    assert report["synapse_reference"]["phase2j_head_val"]["coverage"] == 1.0


def test_phase2j_data_health_rejects_missing_synapse_reference_when_requested(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "phase2j_train"
    val_dir = tmp_path / "phase2j_val"
    build_debug_cortex_challenge(train_dir, profile="phase2j_semantic_train", episodes_per_scenario=1)
    build_debug_cortex_challenge(val_dir, profile="phase2j_semantic_val", episodes_per_scenario=1)
    head_train = _write_jsonl(
        tmp_path / "head_train.jsonl",
        [_head_row("train", slot) for slot in range(4)],
    )
    head_val = _write_jsonl(
        tmp_path / "head_val.jsonl",
        [_head_row("val", slot) for slot in range(4)],
    )

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_semantic_train": train_dir / "challenge.jsonl",
            "phase2j_semantic_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_train_val_target_overlap=1.0,
        max_semantic_nn=1.01,
        require_synapse_reference=True,
    )

    assert report["passed"] is False
    assert report["checks"]["phase2j_head_train_synapse_reference_present"] is False
    assert report["checks"]["phase2j_head_val_synapse_reference_present"] is False


def test_phase2j_data_health_requires_debug_action_stage_when_requested(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "phase2j_train"
    val_dir = tmp_path / "phase2j_val"
    build_debug_cortex_challenge(train_dir, profile="phase2j_semantic_train", episodes_per_scenario=1)
    build_debug_cortex_challenge(val_dir, profile="phase2j_semantic_val", episodes_per_scenario=1)
    stages = [
        "raw_failure_output",
        "parsed_failure_summary",
        "source_inspected",
        "source_inspected",
    ]
    rows = [_head_row("stage", slot) for slot in range(4)]
    for row, stage in zip(rows, stages):
        row["nsi_reference"]["debug_action_stage"] = stage
    head_train = _write_jsonl(tmp_path / "head_train.jsonl", rows)
    head_val = _write_jsonl(tmp_path / "head_val.jsonl", rows)

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_semantic_train": train_dir / "challenge.jsonl",
            "phase2j_semantic_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_train_val_target_overlap=1.0,
        max_semantic_nn=1.01,
        require_debug_action_stage=True,
    )

    assert report["passed"] is True
    assert report["checks"]["phase2j_head_train_debug_action_stage_present"] is True
    assert report["checks"]["phase2j_head_val_debug_action_stage_coverage"] is True
    assert report["debug_action_stage"]["phase2j_head_val"]["stage_counts"][
        "raw_failure_output"
    ] == 1


def test_phase2j_data_health_rejects_missing_debug_action_stage_when_requested(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "phase2j_train"
    val_dir = tmp_path / "phase2j_val"
    build_debug_cortex_challenge(train_dir, profile="phase2j_semantic_train", episodes_per_scenario=1)
    build_debug_cortex_challenge(val_dir, profile="phase2j_semantic_val", episodes_per_scenario=1)
    rows = [_head_row("stage", slot) for slot in range(4)]
    rows[0]["nsi_reference"]["debug_action_stage"] = "raw_failure_output"
    rows[1]["nsi_reference"]["debug_action_stage"] = "parsed_failure_summary"
    head_train = _write_jsonl(tmp_path / "head_train.jsonl", rows)
    head_val = _write_jsonl(tmp_path / "head_val.jsonl", rows)

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_semantic_train": train_dir / "challenge.jsonl",
            "phase2j_semantic_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_train_val_target_overlap=1.0,
        max_semantic_nn=1.01,
        require_debug_action_stage=True,
    )

    assert report["passed"] is False
    assert report["checks"]["phase2j_head_train_debug_action_stage_present"] is False
    assert report["checks"]["phase2j_head_val_debug_action_stage_coverage"] is False


def test_phase2j_source_overlap_hard_data_health_rejects_easy_source_overlap_val(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "phase2j_hard_train"
    val_dir = tmp_path / "phase2j_hard_val"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2j_source_overlap_hard_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2j_source_overlap_hard_val",
        episodes_per_scenario=1,
    )
    head_train = _write_jsonl(
        tmp_path / "head_train.jsonl",
        [_hard_head_row("train", slot) for slot in range(4)],
    )
    head_val = _write_jsonl(
        tmp_path / "head_val.jsonl",
        [_hard_head_row("val", slot, source_overlap_easy=True) for slot in range(4)],
    )

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_source_overlap_hard_train": train_dir / "challenge.jsonl",
            "phase2j_source_overlap_hard_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_semantic_nn=1.01,
        source_overlap_hard=True,
        max_source_overlap_val_accuracy=0.30,
    )

    assert report["passed"] is False
    assert report["checks"]["phase2j_source_overlap_hard_val_baseline_below_threshold"] is False


def test_phase2j_source_overlap_hard_data_health_rejects_incomplete_identity_signal(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "phase2j_hard_train"
    val_dir = tmp_path / "phase2j_hard_val"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2j_source_overlap_hard_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2j_source_overlap_hard_val",
        episodes_per_scenario=1,
    )
    head_train = _write_jsonl(
        tmp_path / "head_train.jsonl",
        [_hard_head_row("train", slot) for slot in range(4)],
    )
    val_rows = [_hard_head_row("val", slot) for slot in range(4)]
    val_rows[2]["nsi_reference"] = {f"command_identity_slot:{index}": 0.0 for index in range(4)}
    head_val = _write_jsonl(tmp_path / "head_val.jsonl", val_rows)

    report = build_phase2j_data_health_audit(
        head_splits={"phase2j_head_train": head_train, "phase2j_head_val": head_val},
        challenge_splits={
            "phase2j_source_overlap_hard_train": train_dir / "challenge.jsonl",
            "phase2j_source_overlap_hard_val": val_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_semantic_nn=1.01,
        source_overlap_hard=True,
        max_source_overlap_val_accuracy=0.30,
    )

    assert report["passed"] is False
    assert report["checks"]["phase2j_source_overlap_hard_identity_signal_present"] is False
    assert report["source_overlap_hard"]["phase2j_head_val"]["identity_accuracy"] == 0.75
