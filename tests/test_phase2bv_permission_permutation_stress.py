from reflexlm.cli.run_phase2bv_permission_permutation_stress import _permuted_manifest


def test_phase2bv_reverses_commands_without_changing_completion_requirements() -> None:
    source = {
        "episodes": [
            {
                "permissions": [
                    {"action_type": "RUN_COMMAND", "argv": ["python", "fail.py"]},
                    {"action_type": "READ_STDERR"},
                    {"action_type": "RUN_COMMAND", "argv": ["python", "recover.py"]},
                    {"action_type": "DONE"},
                ],
                "completion_requirements": [
                    {"action_type": "RUN_COMMAND", "argv": ["python", "fail.py"]},
                    {"action_type": "RUN_COMMAND", "argv": ["python", "recover.py"]},
                ],
            }
        ]
    }

    permuted = _permuted_manifest(source)

    commands = [
        row["argv"] for row in permuted["episodes"][0]["permissions"]
        if row["action_type"] == "RUN_COMMAND"
    ]
    assert commands == [["python", "recover.py"], ["python", "fail.py"]]
    assert (
        permuted["episodes"][0]["completion_requirements"]
        == source["episodes"][0]["completion_requirements"]
    )
