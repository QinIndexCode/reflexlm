from pathlib import Path

from reflexlm.baselines.text_policies import scaled_generation_budget
from reflexlm.data.tasks import (
    build_env,
    build_env_from_episode_id,
    materialize_phase1_dataset,
    parse_episode_id,
    rollout_env,
)
from reflexlm.eval import RuleOraclePolicyAdapter, SequenceModelPolicy, evaluate_policy
from reflexlm.reporting import bootstrap_paired_difference
from reflexlm.schema import ActionDecision, ActionType, TaskType
from reflexlm.train import (
    TrainerConfig,
    load_model_checkpoint,
    save_model_checkpoint,
    train_nsi_model,
)


def test_episode_id_round_trip() -> None:
    env = build_env(TaskType.TEST_FAILURE, 7)
    task_type, episode_index = parse_episode_id(env.episode_id)
    reconstructed = build_env_from_episode_id(env.episode_id)
    assert task_type == TaskType.TEST_FAILURE
    assert episode_index == 7
    assert reconstructed.episode_id == env.episode_id
    assert reconstructed.variant == env.variant


def test_scaled_generation_budget_grows_only_on_retry() -> None:
    first = scaled_generation_budget(
        max_new_tokens=96,
        max_time_s=20.0,
        attempt=0,
        parse_retry_growth=2.0,
    )
    retry = scaled_generation_budget(
        max_new_tokens=96,
        max_time_s=20.0,
        attempt=1,
        parse_retry_growth=2.0,
    )

    assert first == {"max_new_tokens": 96, "max_time": 20.0}
    assert retry == {"max_new_tokens": 192, "max_time": 40.0}


class PartialTestFailurePolicy:
    def __init__(self) -> None:
        self.stats = type("Stats", (), {"token_cost": 0, "model_calls": 0, "parse_failures": 0, "retries": 0})()
        self.last_call = {}

    def reset(self) -> None:
        self.last_call = {}

    def metadata(self) -> dict[str, str]:
        return {"policy_family": "test_policy", "policy_label": "partial_test_failure"}

    def act(self, state):
        if state.terminal.stderr_delta:
            return ActionDecision(type=ActionType.READ_STDERR, reason="inspect")
        if state.filesystem.dirty_files:
            return ActionDecision(
                type=ActionType.READ_FILE,
                file_target=state.filesystem.dirty_files[0],
                reason="inspect_source",
            )
        wrong_command = state.goal.command_allowlist[-1]
        return ActionDecision(type=ActionType.RUN_COMMAND, command=wrong_command, reason="wrong_rerun")


def test_evaluation_records_partial_credit_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "assertion_test_failure.jsonl"
    profile = "phase2j_source_overlap_hard_val"
    records = rollout_env(build_env(TaskType.TEST_FAILURE, 2, profile=profile))
    dataset_path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )

    summary = evaluate_policy(
        PartialTestFailurePolicy(),
        dataset_path=dataset_path,
        limit_episodes=1,
        task_filter={TaskType.TEST_FAILURE},
        env_profile=profile,
    )
    row = summary.per_episode[0]

    assert row["task_completion_rate"] == 0.0
    assert 0.0 < row["oracle_step_accuracy"] < 1.0
    assert row["read_file_decision_accuracy"] == 1.0
    assert row["command_decision_accuracy"] == 0.0
    assert 0.0 < row["positive_reward_credit"] < 1.0
    assert summary.aggregate["oracle_step_accuracy"]["mean"] == round(
        row["oracle_step_accuracy"],
        6,
    )
    assert summary.aggregate["command_decision_accuracy"]["mean"] == 0.0


def test_evaluation_and_checkpoint_round_trip(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    materialize_phase1_dataset(dataset_dir, seed=7)
    test_split = dataset_dir / "test.jsonl"
    train_split = dataset_dir / "train.jsonl"

    rule_summary = evaluate_policy(
        RuleOraclePolicyAdapter(),
        dataset_path=test_split,
        limit_episodes=12,
    )
    assert len(rule_summary.per_episode) == 12
    assert rule_summary.aggregate["task_completion_rate"]["mean"] == 1.0

    model, vectorizer, training_summary = train_nsi_model(
        train_split,
        trainer_config=TrainerConfig(
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            device="cpu",
            seed=7,
        ),
        smoke=True,
    )
    checkpoint_path = save_model_checkpoint(
        model,
        vectorizer,
        checkpoint_path=tmp_path / "smoke_nsi.pt",
        model_kind=training_summary["model_kind"],
        summary=training_summary,
    )
    loaded_model, loaded_vectorizer, _payload = load_model_checkpoint(checkpoint_path, device="cpu")
    sequence_summary = evaluate_policy(
        SequenceModelPolicy(loaded_model, loaded_vectorizer, policy_label="smoke_nsi"),
        dataset_path=test_split,
        limit_episodes=8,
    )
    assert len(sequence_summary.per_episode) == 8
    assert sequence_summary.aggregate["model_calls"]["mean"] >= 1.0

    diff = bootstrap_paired_difference(
        rule_summary.per_episode[:8],
        sequence_summary.per_episode,
        "task_completion_rate",
    )
    assert diff is not None
