from typing import Any

from llm_ucdcr.approaches.e2e.models import ClusterAssignment, E2EMentionView
from llm_ucdcr.data.mention import Mention

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
    "mention_context": ["He", "met", "Canada", "today"],
    "context_start_end_global_id": [0, 4],
    "mention_sentence_start_end": [0, 4],
    "tokens_number_context": [2],
    "mention_head_id_context": 2,
    "is_singleton": False,
    "conll_doc_key": "t1/s1/d1",
}


def _make(**overrides: Any) -> Mention:
    return Mention.from_dict({**_BASE, **overrides})


def test_from_mention_copies_identity_fields() -> None:
    view = E2EMentionView.from_mention(_make(mention_id="m42"))
    assert view.mention_id == "m42"
    assert view.text == "Canada"
    assert view.head == "Canada"


def test_from_mention_tags_mention_in_context() -> None:
    view = E2EMentionView.from_mention(_make())
    assert view.context == "He met <mention> Canada </mention> today"


def test_from_mention_tags_trailing_mention() -> None:
    view = E2EMentionView.from_mention(
        _make(mention_context=["He", "met", "Canada"], tokens_number_context=[2]),
    )
    assert view.context == "He met <mention> Canada </mention>"


def test_cluster_assignment_parses_nested_lists() -> None:
    assignment = ClusterAssignment.model_validate({"clusters": [["m1", "m2"], ["m3"]]})
    assert assignment.clusters == [["m1", "m2"], ["m3"]]
