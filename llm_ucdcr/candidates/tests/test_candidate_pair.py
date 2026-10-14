from typing import Any

from llm_ucdcr.data.mention import Mention
from llm_ucdcr.candidates.candidate_pair import CandidatePair

_BASE: dict[str, Any] = {
    "mention_id": "m1",
    "coref_chain": "chain1",
    "tokens_str": "Canada",
    "description": "desc",
    "coref_type": "IDENTITY",
    "mention_type": "LOC",
    "mention_full_type": "LOC_GEO",
    "tokens_text": ["Canada"],
    "tokens_number": [1],
    "mention_head": "Canada",
    "mention_head_id": 1,
    "mention_head_pos": "PROPN",
    "mention_head_lemma": "canada",
    "mention_ner": "GPE",
    "sent_id": 0,
    "topic_id": "t1",
    "topic": "topic_one",
    "subtopic_id": "s1",
    "subtopic": "subtopic_one",
    "doc_id": "d1",
    "doc": "doc_one",
    "mention_context": ["Some", "context"],
    "context_start_end_global_id": [0, 2],
    "mention_sentence_start_end": [0, 2],
    "tokens_number_context": [1],
    "mention_head_id_context": 1,
    "is_singleton": False,
    "conll_doc_key": "t1/s1/d1",
}


def _make(**overrides: Any) -> Mention:
    return Mention.from_dict({**_BASE, **overrides})


def test_is_coreferent_true() -> None:
    pair = CandidatePair(
        mention_a=_make(mention_id="m1", coref_chain="chain1"),
        mention_b=_make(mention_id="m2", coref_chain="chain1"),
    )
    assert pair.is_coreferent is True


def test_is_coreferent_false() -> None:
    pair = CandidatePair(
        mention_a=_make(mention_id="m1", coref_chain="chain1"),
        mention_b=_make(mention_id="m2", coref_chain="chain2"),
    )
    assert pair.is_coreferent is False


def test_is_coreferent_false_for_same_chain_in_different_subtopics() -> None:
    """Gold membership is keyed by (coref_chain, subtopic), so a shared chain across subtopics is not a link."""
    pair = CandidatePair(
        mention_a=_make(mention_id="m1", coref_chain="chain1", subtopic_id="s1"),
        mention_b=_make(mention_id="m2", coref_chain="chain1", subtopic_id="s2"),
    )
    assert pair.is_coreferent is False
