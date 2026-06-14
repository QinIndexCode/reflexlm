from dataclasses import asdict
from pathlib import Path

from reflexlm.cli.audit_phase2bj_action_conditioned_world_model import (
    audit_phase2bj_action_conditioned_world_model,
)
from reflexlm.data.tasks import materialize_phase1_dataset
from reflexlm.models.nsi_model import NSIModelConfig, NSIReflexModel
from reflexlm.train import save_model_checkpoint


def test_phase2bj_audit_rejects_unconditioned_legacy_world_model(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    materialize_phase1_dataset(dataset_dir, seed=11)
    from reflexlm.models.features import StateVectorizer

    vectorizer = StateVectorizer()
    config = NSIModelConfig.smoke(vectorizer.vector_dim)
    config.action_conditioned_world_model = False
    model = NSIReflexModel(config)
    summary = {
        "model_kind": "nsi",
        "model_config": asdict(config),
        "vectorizer": asdict(vectorizer),
        "training_summary": {},
    }
    checkpoint = save_model_checkpoint(
        model,
        vectorizer,
        checkpoint_path=tmp_path / "unconditioned.pt",
        model_kind="nsi",
        summary=summary,
    )

    report = audit_phase2bj_action_conditioned_world_model(
        checkpoint_path=checkpoint,
        dataset_path=dataset_dir / "test.jsonl",
        min_rows=1,
        max_rows=20,
    )

    assert report["passed"] is False
    assert report["checks"]["checkpoint_declares_action_conditioned_world_model"] is False
    assert report["checks"]["world_model_is_action_sensitive"] is False
    assert report["ready_for_bounded_heldout_action_conditioned_world_model_claim"] is False
    assert report["ready_for_bounded_real_runtime_action_conditioned_world_model_claim"] is False
    assert report["ready_for_bounded_continuous_runtime_recovery_world_model_claim"] is False
