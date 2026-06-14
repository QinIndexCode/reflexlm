from pathlib import Path

from reflexlm.data.jsonl import write_jsonl
from reflexlm.data.tasks import build_env, rollout_env
from reflexlm.runtime.oracle import RuleOracle
from reflexlm.schema import TaskType
from reflexlm.train import TrainerConfig, train_flat_text_baseline, train_nsi_model


def _small_records() -> list:
    oracle = RuleOracle()
    records = []
    for task_type in TaskType:
        env = build_env(task_type, 0)
        records.extend(rollout_env(env, policy=oracle))
    return records


def test_nsi_training_smoke(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    write_jsonl(dataset_path, _small_records())
    _, _, summary = train_nsi_model(
        dataset_path,
        trainer_config=TrainerConfig(epochs=1, batch_size=2, learning_rate=1e-3),
        smoke=True,
    )
    assert summary["parameter_count"] > 0
    assert summary["history"][0]["loss"] >= 0.0


def test_flat_text_training_smoke(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    write_jsonl(dataset_path, _small_records())
    _, _, summary = train_flat_text_baseline(
        dataset_path,
        trainer_config=TrainerConfig(epochs=1, batch_size=2, learning_rate=1e-3),
        smoke=True,
    )
    assert summary["parameter_count"] > 0
    assert summary["history"][0]["loss"] >= 0.0
