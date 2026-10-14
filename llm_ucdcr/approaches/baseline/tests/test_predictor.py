import asyncio
from typing import Any

from llm_ucdcr.approaches.baseline import BaselinePredictor, BaselineStrategy
from llm_ucdcr.candidates.candidate_pair import CandidatePair
from llm_ucdcr.data.mention import Mention

_BASE: dict[str, Any] = {
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
    "mention_context": ["Some", "Canada", "context"],
    "context_start_end_global_id": [0, 3],
    "mention_sentence_start_end": [0, 3],
    "tokens_number_context": [1],
    "mention_head_id_context": 1,
    "is_singleton": False,
    "conll_doc_key": "t1/s1/d1",
}


def _pair(chain_a: str, chain_b: str) -> CandidatePair:
    return CandidatePair(
        mention_a=Mention.from_dict({**_BASE, "mention_id": "m1", "coref_chain": chain_a}),
        mention_b=Mention.from_dict({**_BASE, "mention_id": "m2", "coref_chain": chain_b}),
    )


def _lemma_pair(lemma_a: str, lemma_b: str) -> CandidatePair:
    # Distinct gold chains on purpose, so any link is driven solely by the head lemma.
    return CandidatePair(
        mention_a=Mention.from_dict({**_BASE, "mention_id": "m1", "coref_chain": "c1", "mention_head_lemma": lemma_a}),
        mention_b=Mention.from_dict({**_BASE, "mention_id": "m2", "coref_chain": "c2", "mention_head_lemma": lemma_b}),
    )


def test_constant_true_always_predicts_coreferent() -> None:
    predictor = BaselinePredictor(BaselineStrategy.CONSTANT_TRUE)

    coreferent = asyncio.run(predictor.predict(_pair("c1", "c1")))
    distinct = asyncio.run(predictor.predict(_pair("c1", "c2")))

    assert coreferent.coreferent is True
    assert distinct.coreferent is True  # ignores gold; the recall-ceiling baseline
    assert coreferent.reasoning is None


def test_oracle_mirrors_the_gold_label() -> None:
    predictor = BaselinePredictor(BaselineStrategy.ORACLE)

    same_chain = asyncio.run(predictor.predict(_pair("c1", "c1")))
    other_chain = asyncio.run(predictor.predict(_pair("c1", "c2")))

    assert same_chain.coreferent is True
    assert other_chain.coreferent is False


def test_head_lemma_links_shared_lemmas_ignoring_gold() -> None:
    predictor = BaselinePredictor(BaselineStrategy.HEAD_LEMMA)

    # Same lemma, different gold chains -> still linked (it never looks at the gold label).
    same_lemma = asyncio.run(predictor.predict(_lemma_pair("attack", "attack")))
    other_lemma = asyncio.run(predictor.predict(_lemma_pair("attack", "bombing")))

    assert same_lemma.coreferent is True
    assert other_lemma.coreferent is False


def test_head_lemma_matching_is_case_insensitive() -> None:
    predictor = BaselinePredictor(BaselineStrategy.HEAD_LEMMA)

    decision = asyncio.run(predictor.predict(_lemma_pair("Attack", "attack")))

    assert decision.coreferent is True
