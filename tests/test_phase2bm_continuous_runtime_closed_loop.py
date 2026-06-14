from dataclasses import asdict
from pathlib import Path

from reflexlm.cli.audit_phase2bm_continuous_runtime_closed_loop import (
    audit_phase2bm_continuous_runtime_closed_loop,
)
from reflexlm.data.tasks import materialize_phase1_dataset
from reflexlm.models.features import StateVectorizer
from reflexlm.models.nsi_model import NSIModelConfig, NSIReflexModel
from reflexlm.train import save_model_checkpoint


def test_phase2bm_audit_rejects_non_runtime_noncontinuous_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    materialize_phase1_dataset(dataset_dir, seed=11)
    vectorizer = StateVectorizer()
    config = NSIModelConfig.smoke(vectorizer.vector_dim)
    model = NSIReflexModel(config)
    checkpoint = save_model_checkpoint(
        model,
        vectorizer,
        checkpoint_path=tmp_path / "random.pt",
        model_kind="nsi",
        summary={
            "model_kind": "nsi",
            "model_config": asdict(config),
            "vectorizer": asdict(vectorizer),
            "training_summary": {},
        },
    )

    report = audit_phase2bm_continuous_runtime_closed_loop(
        checkpoint_path=checkpoint,
        dataset_path=dataset_dir / "test.jsonl",
        min_rows=1,
        max_rows=20,
    )

    assert report["passed"] is False
    assert report["checks"]["all_rows_are_runtime_observations"] is False
    assert report["ready_for_bounded_continuous_runtime_policy_world_model_claim"] is False
