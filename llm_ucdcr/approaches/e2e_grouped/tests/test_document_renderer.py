from typing import Any

from llm_ucdcr.approaches.e2e_grouped.document_renderer import DocumentRenderer
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


def _make(**overrides: Any) -> Mention:
    return Mention.from_dict({**_BASE, **overrides})


def test_single_mention_is_tagged_in_reconstructed_text() -> None:
    rendered = DocumentRenderer().render_document([_make(mention_id="m1")])
    assert rendered == "He met [[m=m1]] Obama [[/m=m1]]"


def test_multi_token_mention_wraps_whole_span() -> None:
    mention = _make(
        mention_id="m1",
        mention_context=["the", "White", "House"],
        context_start_end_global_id=[0, 2],
        tokens_number_context=[1, 2],
    )
    assert DocumentRenderer().render_document([mention]) == "the [[m=m1]] White House [[/m=m1]]"


def test_overlapping_windows_reconstruct_shared_text_once() -> None:
    # m1's window is tokens 0..2 (a b c), m2's window is tokens 1..3 (b c d): b and c are shared.
    m1 = _make(mention_id="m1", mention_context=["a", "b", "c"], context_start_end_global_id=[0, 2], tokens_number_context=[0])
    m2 = _make(mention_id="m2", mention_context=["b", "c", "d"], context_start_end_global_id=[1, 3], tokens_number_context=[2])
    rendered = DocumentRenderer().render_document([m1, m2])
    # Shared tokens b, c appear exactly once; each mention tagged at its own position.
    assert rendered == "[[m=m1]] a [[/m=m1]] b c [[m=m2]] d [[/m=m2]]"
    assert rendered.count(" b ") == 1


def test_uncovered_stretch_becomes_a_gap_marker() -> None:
    m1 = _make(mention_id="m1", mention_context=["a", "b"], context_start_end_global_id=[0, 1], tokens_number_context=[0])
    m2 = _make(mention_id="m2", mention_context=["f", "g"], context_start_end_global_id=[5, 6], tokens_number_context=[1])
    rendered = DocumentRenderer().render_document([m1, m2])
    assert rendered == "[[m=m1]] a [[/m=m1]] b [...] f [[m=m2]] g [[/m=m2]]"


def test_render_subtopic_groups_by_document_with_headers() -> None:
    mentions = MentionCollection(
        [
            _make(mention_id="m1", doc_id="dA"),
            _make(mention_id="m2", doc_id="dB"),
        ],
    )
    rendered = DocumentRenderer().render_subtopic(mentions)
    assert "### Document dA" in rendered
    assert "### Document dB" in rendered
    assert rendered.index("### Document dA") < rendered.index("### Document dB")  # sorted by doc id
