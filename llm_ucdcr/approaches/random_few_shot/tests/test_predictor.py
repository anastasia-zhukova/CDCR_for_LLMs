import asyncio
from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

from llm_ucdcr.approaches.base import PairDecision
from llm_ucdcr.approaches.random_few_shot.models import (
    CoreferenceDecision,
    LabeledPairView,
    MentionView,
    PairView,
)
from llm_ucdcr.approaches.random_few_shot.predictor import RandomFewShotPredictor


def _mention(
    text: str = "Obama",
    head: str = "Obama",
    context: str = "He met <mention> Obama </mention> there.",
) -> MentionView:
    return MentionView(text=text, head=head, context=context)


def _pair_view() -> PairView:
    return PairView(mention_a=_mention("Obama"), mention_b=_mention("Barack Obama", head="Obama"))


def _labeled(coreferent: bool = True) -> LabeledPairView:
    return LabeledPairView(mention_a=_mention(), mention_b=_mention(), coreferent=coreferent)


def _predictor(examples: Sequence[LabeledPairView] = ()) -> RandomFewShotPredictor:
    predictor = RandomFewShotPredictor(connector=MagicMock())
    predictor.bind_examples(examples)
    return predictor


# --- _to_messages ---


def test_to_messages_returns_single_user_message() -> None:
    messages = _predictor()._to_messages(_pair_view())
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_to_messages_contains_candidate_section() -> None:
    content = _predictor()._to_messages(_pair_view())[0]["content"]
    assert "## Candidate pair" in content


def test_to_messages_contains_mention_texts() -> None:
    content = _predictor()._to_messages(_pair_view())[0]["content"]
    assert "Obama" in content


def test_to_messages_with_examples_contains_examples_section() -> None:
    content = _predictor([_labeled(True), _labeled(False)])._to_messages(_pair_view())[0]["content"]
    assert "## Examples" in content
    assert "### Example 1" in content
    assert "### Example 2" in content


def test_to_messages_examples_include_decision_label() -> None:
    content = _predictor([_labeled(True), _labeled(False)])._to_messages(_pair_view())[0]["content"]
    assert "Decision: coreferent" in content
    assert "Decision: not coreferent" in content


def test_to_messages_without_examples_has_no_examples_section() -> None:
    content = _predictor()._to_messages(_pair_view())[0]["content"]
    assert "## Examples" not in content


def test_to_messages_candidate_section_comes_after_examples() -> None:
    content = _predictor([_labeled()])._to_messages(_pair_view())[0]["content"]
    assert content.index("## Examples") < content.index("## Candidate pair")


# --- predict ---


def test_predict_forwards_connector_decision() -> None:
    connector = MagicMock()
    connector.predict = AsyncMock(return_value=CoreferenceDecision(coreferent=True))
    predictor = RandomFewShotPredictor(connector=connector)

    with patch.object(PairView, "from_pair", return_value=_pair_view()):
        decision = asyncio.run(predictor.predict(MagicMock()))

    assert decision == PairDecision(coreferent=True)
    call_kwargs = connector.predict.call_args.kwargs
    assert "coreference" in call_kwargs["system_prompt"].lower()
    assert call_kwargs["output_model"] is CoreferenceDecision
