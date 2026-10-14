import asyncio
import random
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from llm_ucdcr.approaches.e2e.models import ClusterAssignment, E2EMentionView
from llm_ucdcr.approaches.e2e.predictor import E2EPredictor
from llm_ucdcr.approaches.e2e.token_estimator import PromptTokenEstimator
from llm_ucdcr.data.loader import MentionCollection
from llm_ucdcr.data.mention import Mention

_BASE: dict[str, Any] = {
    "mention_id": "m1",
    "coref_chain": "chain1",
    "tokens_str": "Obama",
    "description": "desc",
    "coref_type": "IDENTITY",
    "mention_type": "PER",
    "mention_full_type": "PER_IND",
    "tokens_text": ["Obama"],
    "tokens_number": [1],
    "mention_head": "Obama",
    "mention_head_id": 1,
    "mention_head_pos": "PROPN",
    "mention_head_lemma": "obama",
    "mention_ner": "PERSON",
    "sent_id": 0,
    "topic_id": "t1",
    "topic": "topic_one",
    "subtopic_id": "s1",
    "subtopic": "subtopic_one",
    "doc_id": "d1",
    "doc": "doc_one",
    "mention_context": ["He", "met", "Obama"],
    "context_start_end_global_id": [0, 3],
    "mention_sentence_start_end": [0, 3],
    "tokens_number_context": [2],
    "mention_head_id_context": 2,
    "is_singleton": False,
    "conll_doc_key": "t1/s1/d1",
}


def _make(**overrides: Any) -> Mention:
    return Mention.from_dict({**_BASE, **overrides})


def _collection(*ids: str) -> MentionCollection:
    return MentionCollection([_make(mention_id=mention_id) for mention_id in ids])


def _predictor(rng: random.Random | None = None) -> E2EPredictor:
    return E2EPredictor(
        connector=MagicMock(),
        estimator=PromptTokenEstimator("o200k_base"),
        rng=rng,
    )


# --- _to_messages ---


def test_to_messages_returns_single_user_message() -> None:
    views = [E2EMentionView.from_mention(_make(mention_id="m1"))]
    messages = _predictor()._to_messages(views)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_to_messages_lists_every_mention_id_and_text() -> None:
    views = [E2EMentionView.from_mention(m) for m in (_make(mention_id="m1"), _make(mention_id="m2"))]
    content = _predictor()._to_messages(views)[0]["content"]
    assert "id=m1" in content
    assert "id=m2" in content
    assert "Obama" in content


# --- _sanitize ---


def test_sanitize_drops_invented_ids() -> None:
    clusters = _predictor()._sanitize([["m1", "ghost"]], {"m1", "m2"})
    assert clusters == []  # m1 is left alone once ghost is dropped, so the chain collapses


def test_sanitize_drops_single_element_chains() -> None:
    clusters = _predictor()._sanitize([["m1"], ["m2", "m3"]], {"m1", "m2", "m3"})
    assert clusters == [["m2", "m3"]]


def test_sanitize_keeps_valid_chain() -> None:
    clusters = _predictor()._sanitize([["m1", "m2", "m3"]], {"m1", "m2", "m3"})
    assert sorted(clusters[0]) == ["m1", "m2", "m3"]


def test_sanitize_assigns_duplicate_id_to_single_chain() -> None:
    # m2 appears in both chains; it must end up in exactly one of them.
    clusters = _predictor(random.Random(0))._sanitize([["m1", "m2"], ["m2", "m3"]], {"m1", "m2", "m3"})
    occurrences = sum(cluster.count("m2") for cluster in clusters)
    assert occurrences == 1


# --- predict ---


def test_predict_returns_sanitized_clusters() -> None:
    connector = MagicMock()
    connector.predict = AsyncMock(return_value=ClusterAssignment(clusters=[["m1", "m2"], ["bogus"]]))
    predictor = E2EPredictor(connector, PromptTokenEstimator("o200k_base"))

    decision = asyncio.run(predictor.predict(_collection("m1", "m2", "m3")))

    assert decision.clusters == [["m1", "m2"]]
    call_kwargs = connector.predict.call_args.kwargs
    assert call_kwargs["output_model"] is ClusterAssignment
    assert "coreference" in call_kwargs["system_prompt"].lower()
