from reflexlm.cli.run_phase2bp_task_family_disjoint_repo_runtime import (
    _action_signature,
    _manifest_signatures,
)


def test_phase2bp_action_signatures_are_exact_sequence_claims() -> None:
    signature = _action_signature(
        [
            {"action_type": "RUN_COMMAND"},
            {"action_type": "READ_STDERR"},
            {"action_type": "READ_STDOUT"},
            {"action_type": "DONE"},
        ]
    )

    assert signature == "RUN_COMMAND -> READ_STDERR -> READ_STDOUT -> DONE"


def test_phase2bp_manifest_signatures_detect_training_overlap() -> None:
    train = {
        "episodes": [
            {
                "episode_id": "train",
                "steps": [{"action_type": "RUN_COMMAND"}, {"action_type": "DONE"}],
            }
        ]
    }
    holdout = {
        "episodes": [
            {
                "episode_id": "new",
                "steps": [
                    {"action_type": "RUN_COMMAND"},
                    {"action_type": "READ_STDERR"},
                    {"action_type": "DONE"},
                ],
            }
        ]
    }

    assert not (_manifest_signatures(train) & _manifest_signatures(holdout))
