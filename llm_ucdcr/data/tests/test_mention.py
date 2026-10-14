from typing import Any

from llm_ucdcr.data import Mention


_BASE_DICT: dict[str, Any] = {
    "mention_id": "m1",
    "coref_chain": "chain1",
    "tokens_str": "Canada",
    "description": "some description",
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
    "topic_id": "36",
    "topic": "topic_one",
    "subtopic_id": "36ecb",
    "subtopic": "subtopic_one",
    "doc_id": "d1",
    "doc": "doc_one",
    "mention_context": ["Some", "context", "here"],
    "context_start_end_global_id": [0, 3],
    "mention_sentence_start_end": [0, 3],
    "tokens_number_context": [1],
    "mention_head_id_context": 1,
    "is_singleton": False,
    "conll_doc_key": "36/36ecb/d1",
}


def _make_dict(**overrides: Any) -> dict[str, Any]:
    return {**_BASE_DICT, **overrides}


def test_from_dict_creates_mention() -> None:
    mention = Mention.from_dict(_BASE_DICT)
    assert isinstance(mention, Mention)
    assert mention.mention_id == "m1"
    assert mention.coref_chain == "chain1"


def test_from_dict_converts_lists_to_tuples() -> None:
    mention = Mention.from_dict(_BASE_DICT)
    assert isinstance(mention.tokens_text, tuple)
    assert isinstance(mention.tokens_number, tuple)
    assert isinstance(mention.mention_context, tuple)
    assert isinstance(mention.context_start_end_global_id, tuple)
    assert isinstance(mention.tokens_number_context, tuple)


def test_from_dict_topic_id_cast_to_str() -> None:
    mention = Mention.from_dict(_make_dict(topic_id=36))
    assert isinstance(mention.topic_id, str)
    assert mention.topic_id == "36"


def test_from_dict_subtopic_id_cast_to_str() -> None:
    mention = Mention.from_dict(_make_dict(subtopic_id=36))
    assert isinstance(mention.subtopic_id, str)
    assert mention.subtopic_id == "36"


def test_from_dict_language_defaults_to_en() -> None:
    mention = Mention.from_dict(_BASE_DICT)
    assert mention.language == "en"


def test_from_dict_language_explicit() -> None:
    mention = Mention.from_dict(_make_dict(language="de"))
    assert mention.language == "de"
