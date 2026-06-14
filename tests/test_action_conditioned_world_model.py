from dataclasses import asdict
from pathlib import Path

import torch

from reflexlm.models.features import ACTION_ORDER, StateVectorizer
from reflexlm.models.nsi_model import NSIModelConfig, NSIReflexModel
from reflexlm.schema import ActionType
from reflexlm.train import load_model_checkpoint


def test_world_model_target_mask_excludes_stochastic_telemetry_and_text_hashes() -> None:
    vectorizer = StateVectorizer(hash_bins=32)

    mask = vectorizer.world_model_target_mask()

    assert mask.shape == (vectorizer.vector_dim,)
    assert mask[:6].tolist() == [0.0] * 6
    assert mask[6:13].tolist() == [1.0] * 7
    assert mask[13:15].tolist() == [0.0] * 2
    assert mask[15 : vectorizer.numeric_dim].tolist() == [1.0] * (
        vectorizer.numeric_dim - 15
    )
    assert mask[vectorizer.numeric_dim :].tolist() == [0.0] * vectorizer.hash_bins


def test_action_conditioned_world_model_changes_prediction_for_different_actions() -> None:
    torch.manual_seed(7)
    config = NSIModelConfig.smoke(input_dim=12)
    model = NSIReflexModel(config).eval()
    inputs = torch.zeros(1, 1, config.input_dim)
    wait = torch.tensor([[ACTION_ORDER.index(ActionType.WAIT)]])
    stop = torch.tensor([[ACTION_ORDER.index(ActionType.STOP_PROCESS)]])

    wait_prediction = model(inputs, action_indices=wait)["next_state"]
    stop_prediction = model(inputs, action_indices=stop)["next_state"]
    inferred_action_prediction = model(inputs)["next_state"]

    assert wait_prediction.shape == inputs.shape
    assert stop_prediction.shape == inputs.shape
    assert inferred_action_prediction.shape == inputs.shape
    assert not torch.allclose(wait_prediction, stop_prediction)


def test_legacy_checkpoint_without_world_model_flag_loads_unconditioned(tmp_path: Path) -> None:
    vectorizer = StateVectorizer(hash_bins=0)
    config = NSIModelConfig.smoke(vectorizer.vector_dim)
    config.action_conditioned_world_model = False
    config.residual_world_model = False
    model = NSIReflexModel(config)
    model_config = asdict(config)
    model_config.pop("action_conditioned_world_model")
    model_config.pop("residual_world_model")
    checkpoint = tmp_path / "legacy-nsi.pt"
    torch.save(
        {
            "artifact_version": 1,
            "model_kind": "nsi",
            "model_config": model_config,
            "vectorizer": asdict(vectorizer),
            "training_summary": {},
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    loaded, _vectorizer, payload = load_model_checkpoint(checkpoint)

    assert loaded.config.action_conditioned_world_model is False
    assert loaded.config.residual_world_model is False
    assert payload["checkpoint_load"]["missing_keys"] == []
    assert payload["checkpoint_load"]["unexpected_keys"] == []
