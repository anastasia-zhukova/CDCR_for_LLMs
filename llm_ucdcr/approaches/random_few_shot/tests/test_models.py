from typing import Any

from llm_ucdcr.candidates.candidate_pair import CandidatePair
from llm_ucdcr.data.mention import Mention
from llm_ucdcr.approaches.random_few_shot.models import (
    CoreferenceDecision,
    LabeledPairView,
    MentionView,
    PairView,
)

_BASE: dict[str, Any] = {
    "mention_id": "m1",
    "coref_chain": "chain1",
    "tokens_str": "the United States",
    "description": "desc",
    "coref_type": "IDENTITY",
    "mention_type": "LOC",
    "mention_full_type": "LOC_GEO",
    "tokens_text": ["the", "United", "States"],
    "tokens_number": [9, 10, 11],
    "mention_head": "States",
    "mention_head_id": 11,
    "mention_head_pos": "PROPN",
    "mention_head_lemma": "states",
    "mention_ner": "GPE",
    "sent_id": 0,
    "topic_id": "t1",
    "topic": "topic_one",
    "subtopic_id": "s1",
    "subtopic": "subtopic_one",
    "doc_id": "d1",
    "doc": "doc_one",
    "mention_context": ["Stock", "markets", "in", "the", "United", "States", "fell", "today", "."],
    "context_start_end_global_id": [0, 9],
    "mention_sentence_start_end": [0, 9],
    "tokens_number_context": [3, 4, 5],
    "mention_head_id_context": 5,
    "is_singleton": False,
    "conll_doc_key": "t1/s1/d1",
}


def _make(**overrides: Any) -> Mention:
    return Mention.from_dict({**_BASE, **overrides})


def _pair(same_chain: bool = True) -> CandidatePair:
    chain_b = "chain1" if same_chain else "chain2"
    return CandidatePair(
        mention_a=_make(mention_id="m1", coref_chain="chain1"),
        mention_b=_make(mention_id="m2", coref_chain=chain_b),
    )


# --- MentionView ---


def test_mention_view_text_and_head() -> None:
    view = MentionView.from_mention(_make())
    assert view.text == "the United States"
    assert view.head == "States"


def test_mention_view_context_marks_mention() -> None:
    view = MentionView.from_mention(_make())
    assert "<mention>" in view.context
    assert "</mention>" in view.context
    assert "United" in view.context


def test_mention_view_context_tokens_outside_mention_are_unmarked() -> None:
    view = MentionView.from_mention(_make())
    # "Stock" is at index 0, not in tokens_number_context [3,4,5]
    assert view.context.startswith("Stock")


def test_mention_view_context_mention_wrapped_correctly() -> None:
    view = MentionView.from_mention(_make())
    assert "<mention> the United States </mention>" in view.context


def test_build_context_single_token() -> None:
    context = MentionView._build_context(("hello", "world", "there"), (1,))
    assert context == "hello <mention> world </mention> there"


def test_build_context_mention_at_start() -> None:
    context = MentionView._build_context(("hello", "world"), (0,))
    assert context == "<mention> hello </mention> world"


def test_build_context_mention_at_end() -> None:
    context = MentionView._build_context(("hello", "world"), (1,))
    assert context == "hello <mention> world </mention>"


def test_build_context_entire_sequence() -> None:
    context = MentionView._build_context(("only",), (0,))
    assert context == "<mention> only </mention>"


# --- PairView ---


def test_pair_view_from_pair_creates_both_mentions() -> None:
    view = PairView.from_pair(_pair())
    assert isinstance(view.mention_a, MentionView)
    assert isinstance(view.mention_b, MentionView)


def test_pair_view_uses_mention_fields() -> None:
    view = PairView.from_pair(_pair())
    assert view.mention_a.text == "the United States"
    assert view.mention_b.text == "the United States"


# --- LabeledPairView ---


def test_labeled_pair_view_coreferent_true() -> None:
    view = LabeledPairView.from_pair(_pair(same_chain=True))
    assert view.coreferent is True


def test_labeled_pair_view_coreferent_false() -> None:
    view = LabeledPairView.from_pair(_pair(same_chain=False))
    assert view.coreferent is False


def test_labeled_pair_view_has_mention_views() -> None:
    view = LabeledPairView.from_pair(_pair())
    assert isinstance(view.mention_a, MentionView)
    assert isinstance(view.mention_b, MentionView)


# --- CoreferenceDecision ---


def test_coreference_decision_fields() -> None:
    decision = CoreferenceDecision(coreferent=True)
    assert decision.coreferent is True
