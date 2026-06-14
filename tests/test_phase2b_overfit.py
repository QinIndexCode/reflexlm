from reflexlm.cli.analyze_phase2b_overfit import (
    _dynamic_prompt_values,
    _loss_warnings,
    _semantic_tokens,
)


def test_dynamic_prompt_values_ignore_static_instructions() -> None:
    prompt = """Interface thesis: receptor -> synaptic state.

Motor action space:
WAIT, DONE.

Receptor state:
stdout_delta=Paste release gate token:
last_command=python licensing/gate.py

Candidate commands:
- <none>

Motor schema constraints:
- Return only JSON.
"""

    values = _dynamic_prompt_values(prompt)

    assert "Paste release gate token:" in values
    assert "python licensing/gate.py" in values
    assert "Return only JSON." not in values


def test_semantic_tokens_focus_on_variable_state() -> None:
    first = _semantic_tokens(
        "Receptor state:\nstdout_delta=Paste release gate token:\nlast_command=python licensing/gate.py"
    )
    second = _semantic_tokens(
        "Receptor state:\nstderr_delta=TypeError: missing config\nlast_command=pytest tests/test_config.py"
    )

    assert "paste" in first
    assert "typeerror" in second
    assert "typeerror" not in first
    assert first != second


def test_loss_warnings_detect_classic_train_val_overfit() -> None:
    warnings = _loss_warnings(
        [
            {
                "adapter_name": "adapter",
                "final_train_loss": 0.01,
                "final_val_loss": 0.05,
                "val_train_ratio": 5.0,
                "loss_drop_rate": 0.9,
            }
        ],
        max_val_train_ratio=2.0,
        min_loss_drop_rate=0.2,
    )

    assert warnings == [
        {
            "type": "classic_train_val_overfit",
            "adapter_name": "adapter",
            "val_train_ratio": 5.0,
            "threshold": 2.0,
        }
    ]


def test_loss_warnings_detect_weak_fit() -> None:
    warnings = _loss_warnings(
        [
            {
                "adapter_name": "adapter",
                "final_train_loss": 0.9,
                "final_val_loss": 0.8,
                "val_train_ratio": 0.89,
                "loss_drop_rate": 0.05,
            }
        ],
        max_val_train_ratio=2.0,
        min_loss_drop_rate=0.2,
    )

    assert warnings == [
        {
            "type": "weak_fit_or_undertraining",
            "adapter_name": "adapter",
            "loss_drop_rate": 0.05,
            "threshold": 0.2,
        }
    ]
