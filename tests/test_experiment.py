from reflexlm.experiment import MAX_RUN_NAME_SLUG_LENGTH, _slugify


def test_experiment_slug_truncates_long_run_names_without_losing_slug_shape() -> None:
    slug = _slugify("phase2i_semantic_pairwise_" + "very_long_adapter_name_" * 10)

    assert len(slug) <= MAX_RUN_NAME_SLUG_LENGTH
    assert MAX_RUN_NAME_SLUG_LENGTH <= 48
    assert slug.startswith("phase2i-semantic-pairwise")
    assert "--" not in slug
