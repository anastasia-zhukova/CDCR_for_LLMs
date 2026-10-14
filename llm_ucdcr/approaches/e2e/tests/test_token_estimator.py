from llm_ucdcr.approaches.e2e.token_estimator import PromptTokenEstimator


def _estimator() -> PromptTokenEstimator:
    return PromptTokenEstimator("o200k_base")


def test_count_is_positive_for_non_empty_text() -> None:
    assert _estimator().count("hello world") > 0


def test_count_is_zero_for_empty_text() -> None:
    assert _estimator().count("") == 0


def test_count_grows_with_text_length() -> None:
    estimator = _estimator()
    assert estimator.count("one two three four five") > estimator.count("one")
