from pathlib import Path

from reflexlm.data.jsonl import read_jsonl
from reflexlm.cli.generate_debug_cortex_challenge import build_debug_cortex_challenge
from reflexlm.cli.generate_external_trace_set import (
    build_semantic_necessity_audit,
    seal_external_trace_set,
)
from reflexlm.cli.audit_phase2i_data_health import build_phase2i_data_health_audit
from reflexlm.data.tasks import TaskType, materialize_phase1_dataset, scenario_templates_for
from reflexlm.eval import RuleOraclePolicyAdapter, evaluate_policy
from reflexlm.models.features import candidate_commands, serialize_state_as_text


def test_materialized_dataset_respects_episode_splits(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    manifest = materialize_phase1_dataset(output_dir, seed=7)
    assert manifest["total_episodes"] > 0
    train = read_jsonl(output_dir / "train.jsonl")
    val = read_jsonl(output_dir / "val.jsonl")
    test = read_jsonl(output_dir / "test.jsonl")
    train_ids = {record.episode_id for record in train}
    val_ids = {record.episode_id for record in val}
    test_ids = {record.episode_id for record in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_scenario_holdout_uses_sidecar_metadata_without_split_overlap(tmp_path: Path) -> None:
    output_dir = tmp_path / "wide_dataset"
    manifest = materialize_phase1_dataset(
        output_dir,
        seed=17,
        profile="wide_ood",
        split_strategy="scenario_holdout",
    )
    assert manifest["metadata_path"] == "episode_metadata.json"
    metadata_rows = __import__("json").loads(
        (output_dir / "episode_metadata.json").read_text(encoding="utf-8")
    )
    metadata = {row["episode_id"]: row for row in metadata_rows}
    train = read_jsonl(output_dir / "train.jsonl")
    test = read_jsonl(output_dir / "test.jsonl")
    assert metadata
    assert {record.episode_id for record in train + test}.issubset(metadata)
    train_scenarios = {
        (metadata[record.episode_id]["task_type"], metadata[record.episode_id]["scenario_template"])
        for record in train
    }
    test_scenarios = {
        (metadata[record.episode_id]["task_type"], metadata[record.episode_id]["scenario_template"])
        for record in test
    }
    assert train_scenarios.isdisjoint(test_scenarios)


def test_model_serialization_excludes_hidden_hint_and_scenario_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "wide_dataset"
    materialize_phase1_dataset(
        output_dir,
        seed=19,
        profile="wide_ood",
        split_strategy="scenario_holdout",
    )
    record = read_jsonl(output_dir / "train.jsonl")[0]
    serialized = serialize_state_as_text(record.state)
    assert "recovery_hint=" not in serialized
    assert "scenario_template" not in serialized
    assert "profile_seed" not in serialized


def test_debug_ood_profile_does_not_change_wide_ood_scenario_templates() -> None:
    wide_templates = scenario_templates_for(TaskType.TEST_FAILURE, "wide_ood")
    debug_templates = scenario_templates_for(TaskType.TEST_FAILURE, "debug_ood")
    debug_v2_templates = scenario_templates_for(TaskType.TEST_FAILURE, "debug_ood_v2")
    quasi_templates = scenario_templates_for(TaskType.TEST_FAILURE, "quasi_real_terminal")
    latent_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2f_latent_sensitive")
    latent_train_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2f_latent_train")
    latent_val_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2f_latent_val")
    external_templates = scenario_templates_for(TaskType.TEST_FAILURE, "external_trace_v1")
    semantic_templates = scenario_templates_for(
        TaskType.TEST_FAILURE,
        "external_trace_v2_semantic_required",
    )
    semantic_train_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2g_semantic_train")
    semantic_val_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2g_semantic_val")
    phase2h_train_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2h_semantic_train")
    phase2h_val_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2h_semantic_val")
    phase2i_train_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2i_semantic_train")
    phase2i_val_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2i_semantic_val")
    phase2j_train_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2j_semantic_train")
    phase2j_val_templates = scenario_templates_for(TaskType.TEST_FAILURE, "phase2j_semantic_val")
    phase2j_hard_train_templates = scenario_templates_for(
        TaskType.TEST_FAILURE,
        "phase2j_source_overlap_hard_train",
    )
    phase2j_hard_val_templates = scenario_templates_for(
        TaskType.TEST_FAILURE,
        "phase2j_source_overlap_hard_val",
    )
    phase2j_actiongate_train_templates = scenario_templates_for(
        TaskType.TEST_FAILURE,
        "phase2j_source_overlap_hard_actiongate_train",
    )
    phase2j_actiongate_val_templates = scenario_templates_for(
        TaskType.TEST_FAILURE,
        "phase2j_source_overlap_hard_actiongate_val",
    )
    phase2k_train_templates = scenario_templates_for(
        TaskType.TEST_FAILURE,
        "phase2k_continuation_pressure_train",
    )
    phase2k_val_templates = scenario_templates_for(
        TaskType.TEST_FAILURE,
        "phase2k_continuation_pressure_val",
    )
    semantic_v3_templates = scenario_templates_for(
        TaskType.TEST_FAILURE,
        "external_trace_v3_semantic_required",
    )

    assert "pytest_snapshot_update" in wide_templates
    assert "plugin_contract_snapshot_update" not in wide_templates
    assert "plugin_contract_snapshot_update" in debug_templates
    assert len(debug_templates) >= 6
    assert "plugin_contract_snapshot_update_v2" in debug_v2_templates
    assert "local_phase2c_gate_snapshot" in quasi_templates
    assert "latent_assertion_or_cached_rerun" in latent_templates
    assert "latent_train_policy_assertion" in latent_train_templates
    assert "latent_val_acl_assertion" in latent_val_templates
    assert "external_archive_manifest_snapshot" in external_templates
    assert "external_semantic_archive_manifest" in semantic_templates
    assert "semantic_train_manifest_schema" in semantic_train_templates
    assert "semantic_val_gate_delta" in semantic_val_templates
    assert "semantic_train_artifact_hash_registry" in phase2h_train_templates
    assert "semantic_val_candidate_feature_overlap" in phase2h_val_templates
    assert "semantic_train_pair_prompt_visibility" in phase2i_train_templates
    assert "semantic_val_pairwise_collate" in phase2i_val_templates
    assert "phase2j_train_runtime_identity_signal" in phase2j_train_templates
    assert "phase2j_val_pairwise_policy_mask" in phase2j_val_templates
    assert "phase2j_hard_train_router_identity" in phase2j_hard_train_templates
    assert "phase2j_hard_val_sidecar_redaction" in phase2j_hard_val_templates
    assert "phase2j_actiongate_train_2cand_router_resolution" in phase2j_actiongate_train_templates
    assert "phase2j_actiongate_val_2cand_router_resolution" in phase2j_actiongate_val_templates
    assert "phase2k_continuation_pressure_train_2cand_low_same_intent" in phase2k_train_templates
    assert "phase2k_continuation_pressure_val_2cand_low_same_intent" in phase2k_val_templates
    assert "external_v3_phase2b_complete_gate" in semantic_v3_templates
    assert set(latent_templates).isdisjoint(latent_train_templates)
    assert set(latent_templates).isdisjoint(latent_val_templates)
    assert set(external_templates).isdisjoint(debug_v2_templates)
    assert set(external_templates).isdisjoint(quasi_templates)
    assert set(external_templates).isdisjoint(latent_templates)
    assert set(semantic_templates).isdisjoint(external_templates)
    assert set(semantic_templates).isdisjoint(semantic_train_templates)
    assert set(semantic_templates).isdisjoint(semantic_val_templates)
    assert set(semantic_templates).isdisjoint(phase2h_train_templates)
    assert set(semantic_templates).isdisjoint(phase2h_val_templates)
    assert set(semantic_v3_templates).isdisjoint(semantic_templates)
    assert set(semantic_v3_templates).isdisjoint(phase2i_train_templates)
    assert set(semantic_v3_templates).isdisjoint(phase2i_val_templates)
    assert set(semantic_v3_templates).isdisjoint(phase2j_train_templates)
    assert set(semantic_v3_templates).isdisjoint(phase2j_val_templates)
    assert set(semantic_v3_templates).isdisjoint(phase2j_hard_train_templates)
    assert set(semantic_v3_templates).isdisjoint(phase2j_hard_val_templates)
    assert set(semantic_v3_templates).isdisjoint(phase2j_actiongate_train_templates)
    assert set(semantic_v3_templates).isdisjoint(phase2j_actiongate_val_templates)
    assert set(semantic_v3_templates).isdisjoint(phase2k_train_templates)
    assert set(semantic_v3_templates).isdisjoint(phase2k_val_templates)
    assert set(phase2j_train_templates).isdisjoint(phase2i_train_templates)
    assert set(phase2j_val_templates).isdisjoint(phase2i_val_templates)
    assert set(phase2j_hard_train_templates).isdisjoint(phase2i_train_templates)
    assert set(phase2j_hard_val_templates).isdisjoint(phase2i_val_templates)
    assert set(phase2j_actiongate_train_templates).isdisjoint(phase2i_train_templates)
    assert set(phase2j_actiongate_val_templates).isdisjoint(phase2i_val_templates)
    assert set(phase2k_train_templates).isdisjoint(phase2i_train_templates)
    assert set(phase2k_val_templates).isdisjoint(phase2i_val_templates)


def test_debug_cortex_challenge_has_coverage_without_hidden_hint_leaks(tmp_path: Path) -> None:
    output_dir = tmp_path / "debug_challenge"
    manifest = build_debug_cortex_challenge(output_dir, episodes_per_scenario=2)

    assert manifest["passed"] is True
    assert manifest["episode_count"] == 12
    assert set(manifest["variant_counts"]) == {"snapshot", "dependency", "assertion"}
    assert len(manifest["command_targets"]) >= 2
    records = read_jsonl(output_dir / "challenge.jsonl")
    serialized = "\n".join(serialize_state_as_text(record.state) for record in records)
    assert "recovery_hint=" not in serialized
    assert "scenario_template" not in serialized


def test_rule_oracle_uses_debug_ood_allowlist_commands(tmp_path: Path) -> None:
    output_dir = tmp_path / "debug_challenge"
    build_debug_cortex_challenge(output_dir, episodes_per_scenario=1)

    summary = evaluate_policy(
        RuleOraclePolicyAdapter(),
        dataset_path=output_dir / "challenge.jsonl",
        task_filter={TaskType.TEST_FAILURE},
        env_profile="debug_ood",
    )

    assert summary.aggregate["task_completion_rate"]["mean"] == 1.0


def test_debug_ood_v2_and_quasi_real_challenges_are_allowlist_closed(tmp_path: Path) -> None:
    for profile in [
        "debug_ood_v2",
        "debug_transition_train",
        "debug_transition_val",
        "quasi_real_terminal",
        "phase2f_latent_sensitive",
        "phase2f_latent_train",
        "phase2f_latent_val",
        "external_trace_v1",
        "external_trace_v2_semantic_required",
        "phase2g_semantic_train",
        "phase2g_semantic_val",
        "phase2h_semantic_train",
        "phase2h_semantic_val",
        "phase2i_semantic_train",
        "phase2i_semantic_val",
        "phase2j_semantic_train",
        "phase2j_semantic_val",
        "phase2j_source_overlap_hard_train",
        "phase2j_source_overlap_hard_val",
        "phase2j_source_overlap_hard_actiongate_train",
        "phase2j_source_overlap_hard_actiongate_val",
        "phase2j_pressure_val",
        "phase2k_continuation_pressure_train",
        "phase2k_continuation_pressure_val",
        "external_trace_v3_semantic_required",
    ]:
        output_dir = tmp_path / profile
        manifest = build_debug_cortex_challenge(
            output_dir,
            profile=profile,
            episodes_per_scenario=1,
        )
        assert manifest["passed"] is True
        assert manifest["completion_definition"]
        assert manifest["baseline_input_fairness"]
        if profile.startswith("debug_transition"):
            assert manifest["command_intents"]["test_rerun"] > 0
        if profile == "quasi_real_terminal":
            assert "python -m pytest -q paper_draft.md --snapshot-update" not in manifest[
                "command_targets"
            ]
            assert (
                "python -m pytest -q tests/docs/test_paper_evidence_snapshot.py --snapshot-update"
                in manifest["command_targets"]
            )
        if profile.startswith("phase2f_latent"):
            assert manifest["profile"] == profile
            assert manifest["command_intents"]["test_rerun"] > 0
        if profile == "external_trace_v1":
            assert manifest["profile"] == profile
            assert "python -m pip install -e .[dev]" in manifest["command_targets"]
        if profile in {
            "external_trace_v2_semantic_required",
            "external_trace_v3_semantic_required",
        }:
            assert manifest["profile"] == profile
            assert manifest["variant_counts"] == {"assertion": len(manifest["scenario_counts"])}
            assert manifest["command_intents"]["test_rerun"] == len(manifest["scenario_counts"])
            audit = build_semantic_necessity_audit(output_dir / "challenge.jsonl")
            assert audit["passed"] is True
        if (
            profile.startswith("phase2h_semantic")
            or profile.startswith("phase2i_semantic")
            or profile.startswith("phase2j_semantic")
            or profile.startswith("phase2j_source_overlap_hard")
            or profile.startswith("phase2j_pressure")
            or profile.startswith("phase2k_continuation_pressure")
        ):
            assert manifest["profile"] == profile
            assert manifest["variant_counts"] == {"assertion": len(manifest["scenario_counts"])}
        summary = evaluate_policy(
            RuleOraclePolicyAdapter(),
            dataset_path=output_dir / "challenge.jsonl",
            task_filter={TaskType.TEST_FAILURE},
            env_profile=profile,
        )
        if (
            profile.startswith("phase2j_source_overlap_hard")
            or profile.startswith("phase2j_pressure")
            or profile.startswith("phase2k_continuation_pressure")
        ):
            assert summary.aggregate["task_completion_rate"]["mean"] < 1.0
        else:
            assert summary.aggregate["task_completion_rate"]["mean"] == 1.0
        assert summary.aggregate["state_hallucination_rate"]["mean"] == 0.0


def test_external_trace_generation_seals_and_refuses_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "external"
    control_dir = tmp_path / "control"
    manifest = seal_external_trace_set(
        output=output_dir,
        control_dir=control_dir,
        episodes_per_scenario=1,
        reference_paths=[],
    )

    assert manifest["sealed"] is True
    assert manifest["profile"] == "external_trace_v1"
    assert (output_dir / "leakage_audit.json").exists()
    assert (output_dir / "semantic_nn_audit.json").exists()
    assert (output_dir / "command_slot_overlap_audit.json").exists()
    assert (control_dir / "external_trace_v1.sealed").exists()
    leakage = __import__("json").loads((output_dir / "leakage_audit.json").read_text())
    semantic = __import__("json").loads(
        (output_dir / "semantic_necessity_audit.json").read_text()
    )
    assert leakage["passed"] is True
    assert semantic["passed"] is False

    try:
        seal_external_trace_set(
            output=output_dir,
            control_dir=control_dir,
            episodes_per_scenario=1,
            reference_paths=[],
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("sealed external trace generation should refuse overwrite")


def test_semantic_required_external_trace_has_ambiguous_same_intent_commands(tmp_path: Path) -> None:
    output_dir = tmp_path / "external_semantic"
    control_dir = tmp_path / "control"
    manifest = seal_external_trace_set(
        output=output_dir,
        control_dir=control_dir,
        version_name="external_trace_v2_semantic_required",
        profile="external_trace_v2_semantic_required",
        episodes_per_scenario=1,
        reference_paths=[],
    )

    assert manifest["sealed"] is True
    assert manifest["profile"] == "external_trace_v2_semantic_required"
    assert (control_dir / "external_trace_v2_semantic_required.sealed").exists()
    semantic = __import__("json").loads(
        (output_dir / "semantic_necessity_audit.json").read_text()
    )
    assert semantic["passed"] is True
    records = read_jsonl(output_dir / "challenge.jsonl")
    serialized = "\n".join(serialize_state_as_text(record.state) for record in records)
    assert "correct_command" not in serialized
    assert "scenario_template" not in serialized
    assert "recovery_hint=" not in serialized


def test_phase2i_semantic_profiles_balance_correct_command_slots(tmp_path: Path) -> None:
    for profile in [
        "phase2i_semantic_train",
        "phase2i_semantic_val",
        "phase2j_semantic_train",
        "phase2j_semantic_val",
        "phase2j_source_overlap_hard_train",
        "phase2j_source_overlap_hard_val",
        "external_trace_v3_semantic_required",
    ]:
        output_dir = tmp_path / profile
        build_debug_cortex_challenge(
            output_dir,
            profile=profile,
            episodes_per_scenario=1,
        )
        records = read_jsonl(output_dir / "challenge.jsonl")
        slots = []
        for record in records:
            if record.action and record.action.command:
                commands = candidate_commands(record.state)
                slots.append(commands.index(record.action.command))

        assert sorted(slots) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_phase2j_actiongate_profiles_cover_candidate_count_pressure(tmp_path: Path) -> None:
    for profile in [
        "phase2j_source_overlap_hard_actiongate_train",
        "phase2j_source_overlap_hard_actiongate_val",
    ]:
        output_dir = tmp_path / profile
        build_debug_cortex_challenge(
            output_dir,
            profile=profile,
            episodes_per_scenario=1,
        )
        records = read_jsonl(output_dir / "challenge.jsonl")
        run_records = [
            record
            for record in records
            if record.action and record.action.command
        ]
        candidate_counts = {len(candidate_commands(record.state)) for record in run_records}
        slots = [
            candidate_commands(record.state).index(record.action.command)
            for record in run_records
        ]

        assert candidate_counts == {2, 3, 4}
        assert max(slots.count(slot) for slot in set(slots)) / len(slots) <= 0.45


def test_phase2k_continuation_pressure_profiles_record_graded_dimensions(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "phase2k_continuation_pressure_val"
    manifest = build_debug_cortex_challenge(
        output_dir,
        profile="phase2k_continuation_pressure_val",
        episodes_per_scenario=1,
    )
    metadata = __import__("json").loads((output_dir / "episode_metadata.json").read_text())
    records = read_jsonl(output_dir / "challenge.jsonl")
    command_records = [
        record for record in records if record.action and record.action.command
    ]

    assert manifest["profile"] == "phase2k_continuation_pressure_val"
    assert {row["phase2k_evidence_density"] for row in metadata} == {
        "low",
        "medium",
        "high",
    }
    assert {row["phase2k_candidate_count"] for row in metadata} == {2, 3, 4}
    assert {row["phase2k_continuation_depth"] for row in metadata} == {
        "one_step",
        "two_step",
        "stale_state_refresh",
    }
    assert {row["phase2k_ambiguity_class"] for row in metadata} == {
        "same_intent_command",
        "same_file_read",
        "stage_transition",
    }
    assert all(record.state.terminal.last_command == "" for record in command_records)


def test_phase2i_data_health_audit_accepts_disjoint_balanced_semantic_sets(tmp_path: Path) -> None:
    train_dir = tmp_path / "phase2i_train"
    val_dir = tmp_path / "phase2i_val"
    external_dir = tmp_path / "external_v3"
    control_dir = tmp_path / "control"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2i_semantic_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2i_semantic_val",
        episodes_per_scenario=1,
    )
    seal_external_trace_set(
        output=external_dir,
        control_dir=control_dir,
        version_name="external_trace_v3_semantic_required",
        profile="external_trace_v3_semantic_required",
        episodes_per_scenario=1,
        reference_paths=[train_dir / "challenge.jsonl", val_dir / "challenge.jsonl"],
    )
    head_train = tmp_path / "head_train.jsonl"
    head_val = tmp_path / "head_val.jsonl"
    head_train_rows = []
    head_val_rows = []
    for slot in range(4):
        commands = [f"python -m pytest -q tests/test_slot_{slot}_{index}.py" for index in range(4)]
        val_commands = [
            f"python -m pytest -q tests/test_val_slot_{slot}_{index}.py"
            for index in range(4)
        ]
        head_train_rows.append(
            {
                "example_id": f"train-{slot}",
                "episode_id": f"train-{slot}",
                "task_type": "test_failure_reflex",
                "head_scope": "debug_cortex",
                "action_type": "RUN_COMMAND",
                "command_slot": slot,
                "file_slot": -100,
                "command_intent": "test_rerun",
                "candidate_commands": commands,
                "candidate_files": [],
                "state_prompt": "visible receptor state only",
            }
        )
        head_val_rows.append(
            {
                **head_train_rows[-1],
                "example_id": f"val-{slot}",
                "episode_id": f"val-{slot}",
                "candidate_commands": val_commands,
            }
        )
    head_train.write_text(
        "\n".join(__import__("json").dumps(row) for row in head_train_rows),
        encoding="utf-8",
    )
    head_val.write_text(
        "\n".join(__import__("json").dumps(row) for row in head_val_rows),
        encoding="utf-8",
    )

    report = build_phase2i_data_health_audit(
        head_splits={"phase2i_head_train": head_train, "phase2i_head_val": head_val},
        challenge_splits={
            "phase2i_semantic_train": train_dir / "challenge.jsonl",
            "phase2i_semantic_val": val_dir / "challenge.jsonl",
            "external_trace_v3_semantic_required": external_dir / "challenge.jsonl",
        },
        min_val_target_commands=4,
        max_semantic_nn=1.01,
    )

    assert report["passed"] is True
    assert report["checks"]["external_v3_has_no_phase2i_command_overlap"] is True
    assert report["checks"]["phase2i_effective_split_hashes_present"] is True
    assert len(report["head_splits"]["phase2i_head_train"]["effective_split_sha256"]) == 64
    assert (
        report["effective_split_hashes"]["phase2i_head_train"]
        == report["head_splits"]["phase2i_head_train"]["effective_split_sha256"]
    )
    assert report["head_splits"]["phase2i_head_train"]["command_slot_max_share"] == 0.25


def test_phase2i_data_health_audit_rejects_missing_train_command_intent(tmp_path: Path) -> None:
    train_dir = tmp_path / "phase2i_train"
    val_dir = tmp_path / "phase2i_val"
    external_dir = tmp_path / "external_v3"
    control_dir = tmp_path / "control"
    build_debug_cortex_challenge(
        train_dir,
        profile="phase2i_semantic_train",
        episodes_per_scenario=1,
    )
    build_debug_cortex_challenge(
        val_dir,
        profile="phase2i_semantic_val",
        episodes_per_scenario=1,
    )
    seal_external_trace_set(
        output=external_dir,
        control_dir=control_dir,
        version_name="external_trace_v3_semantic_required",
        profile="external_trace_v3_semantic_required",
        episodes_per_scenario=1,
        reference_paths=[train_dir / "challenge.jsonl", val_dir / "challenge.jsonl"],
    )
    head_train = tmp_path / "head_train.jsonl"
    head_val = tmp_path / "head_val.jsonl"
    train_row = {
        "example_id": "train-dep",
        "episode_id": "train-dep",
        "task_type": "test_failure_reflex",
        "head_scope": "debug_cortex",
        "action_type": "RUN_COMMAND",
        "command_slot": 0,
        "file_slot": -100,
        "command_intent": "dependency_install",
        "candidate_commands": ["python -m pip install -r requirements.txt"],
        "candidate_files": [],
        "state_prompt": "visible receptor state only",
    }
    val_row = {
        **train_row,
        "example_id": "val-rerun",
        "episode_id": "val-rerun",
        "command_intent": "test_rerun",
        "candidate_commands": ["python -m pytest -q tests/test_api.py::test_contract"],
    }
    head_train.write_text(__import__("json").dumps(train_row), encoding="utf-8")
    head_val.write_text(__import__("json").dumps(val_row), encoding="utf-8")

    report = build_phase2i_data_health_audit(
        head_splits={"phase2i_head_train": head_train, "phase2i_head_val": head_val},
        challenge_splits={
            "phase2i_semantic_train": train_dir / "challenge.jsonl",
            "phase2i_semantic_val": val_dir / "challenge.jsonl",
            "external_trace_v3_semantic_required": external_dir / "challenge.jsonl",
        },
        min_val_target_commands=1,
        max_semantic_nn=1.01,
    )

    assert report["passed"] is False
    assert report["checks"]["phase2i_train_val_command_intent_coverage"] is False
    assert report["train_val_command_intent_gap"]["missing_train_command_intents"] == [
        "test_rerun"
    ]
