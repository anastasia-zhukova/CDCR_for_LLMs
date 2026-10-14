import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from llm_ucdcr.approaches.e2e import PromptTokenEstimator
from llm_ucdcr.approaches.e2e.models import ClusterAssignment
from llm_ucdcr.approaches.e2e_grouped.predictor import GroupedE2EPredictor
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
    "context_start_end_global_id": [0, 2],
    "mention_sentence_start_end": [0, 2],
    "tokens_number_context": [2],
    "mention_head_id_context": 2,
    "is_singleton": False,
    "conll_doc_key": "t1/s1/d1",
}


def _make(mention_id: str, doc_id: str = "d1") -> Mention:
    return Mention.from_dict({**_BASE, "mention_id": mention_id, "doc_id": doc_id})


def _collection(*ids: str) -> MentionCollection:
    return MentionCollection([_make(mention_id=mention_id) for mention_id in ids])


def _predictor(connector: Any = None) -> GroupedE2EPredictor:
    return GroupedE2EPredictor(
        connector=connector if connector is not None else MagicMock(),
        estimator=PromptTokenEstimator("o200k_base"),
    )


def test_build_messages_renders_documents_with_id_tags() -> None:
    content = _predictor()._build_messages(_collection("m1", "m2"))[0]["content"]
    assert "## Documents" in content
    assert "### Document d1" in content
    assert "[[m=m1]]" in content
    assert "[[m=m2]]" in content


def test_predict_returns_sanitized_clusters() -> None:
    connector = MagicMock()
    connector.predict = AsyncMock(return_value=ClusterAssignment(clusters=[["m1", "m2"], ["ghost"]]))
    predictor = _predictor(connector)

    decision = asyncio.run(predictor.predict(_collection("m1", "m2", "m3")))

    assert decision.clusters == [["m1", "m2"]]  # invented id dropped; m3 left as a singleton
    call_kwargs = connector.predict.call_args.kwargs
    assert call_kwargs["output_model"] is ClusterAssignment
    assert "[[m=ID]]" in call_kwargs["system_prompt"]
