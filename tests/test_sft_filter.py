from reflexlm.cli.filter_sft_dataset import _balanced_limit_rows


def test_balanced_limit_rows_round_robins_metadata_buckets() -> None:
    rows = [
        {"task_type": "a", "action_type": "WAIT", "i": 1},
        {"task_type": "a", "action_type": "WAIT", "i": 2},
        {"task_type": "a", "action_type": "WAIT", "i": 3},
        {"task_type": "b", "action_type": "DONE", "i": 4},
        {"task_type": "b", "action_type": "DONE", "i": 5},
        {"task_type": "c", "action_type": "BLOCK", "i": 6},
    ]

    selected = _balanced_limit_rows(
        rows,
        max_rows=3,
        balance_keys=["task_type"],
        seed=13,
    )

    assert len(selected) == 3
    assert {row["task_type"] for row in selected} == {"a", "b", "c"}


def test_balanced_limit_rows_is_deterministic() -> None:
    rows = [
        {"task_type": str(index % 2), "action_type": str(index % 3), "i": index}
        for index in range(20)
    ]

    first = _balanced_limit_rows(
        rows,
        max_rows=8,
        balance_keys=["task_type", "action_type"],
        seed=47,
    )
    second = _balanced_limit_rows(
        rows,
        max_rows=8,
        balance_keys=["task_type", "action_type"],
        seed=47,
    )

    assert first == second
