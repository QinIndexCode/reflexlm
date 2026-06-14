from pathlib import Path

import pytest

from reflexlm.llm.native_head_policy import _model_load_kwargs


def test_model_load_kwargs_can_force_single_quantized_gpu() -> None:
    kwargs = _model_load_kwargs(
        device="cuda",
        model_load_strategy="single_device",
        offload_state_dict=True,
        offload_folder=Path("load-offload"),
    )

    assert kwargs == {
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "offload_state_dict": True,
        "offload_folder": "load-offload",
    }


def test_model_load_kwargs_rejects_single_device_cpu() -> None:
    with pytest.raises(ValueError, match="requires device='cuda'"):
        _model_load_kwargs(
            device="cpu",
            model_load_strategy="single_device",
            offload_state_dict=False,
        )
