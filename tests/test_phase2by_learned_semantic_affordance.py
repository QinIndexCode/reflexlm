from reflexlm.cli.run_phase2bx_semantic_signal_ablation import COMMAND_TEXT, RECIPES
from reflexlm.cli.run_phase2by_learned_semantic_affordance import _training_groups


def test_phase2by_training_text_does_not_copy_holdout_failure_or_command_strings() -> None:
    groups = _training_groups()
    training_observations = {
        text for group in groups.values() for text in group["observations"]
    }
    training_commands = {
        text for group in groups.values() for text in group["commands"]
    }

    assert not training_observations.intersection(
        {str(recipe["failure"]) for recipe in RECIPES.values()}
    )
    assert not training_commands.intersection(set(COMMAND_TEXT.values()))
