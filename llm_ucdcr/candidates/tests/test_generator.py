from typing import Any


from llm_ucdcr.data.loader import MentionCollection
from llm_ucdcr.data.mention import Mention
from llm_ucdcr.candidates.candidate_pair import CandidatePair
from llm_ucdcr.candidates.generator import CandidateGenerator
from llm_ucdcr.candidates.strategy import CandidateStrategy

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


def _collection(*mentions: Mention) -> MentionCollection:
    return MentionCollection(list(mentions))


def _all_pairs_generator(**kwargs: Any) -> CandidateGenerator:
    return CandidateGenerator(strategy=CandidateStrategy.ALL_PAIRS, **kwargs)


# --- pair generation ---


def test_two_mentions_same_subtopic_yields_one_pair() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m2", subtopic_id="s1", mention_head_lemma="canada"),
    )
    pairs = _all_pairs_generator().generate(collection)
    assert len(pairs) == 1
    assert isinstance(pairs[0], CandidatePair)


def test_different_subtopics_yields_no_pairs() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m2", subtopic_id="s2", mention_head_lemma="canada"),
    )
    pairs = _all_pairs_generator().generate(collection)
    assert len(pairs) == 0


def test_different_lemmas_same_subtopic_still_pair() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m2", subtopic_id="s1", mention_head_lemma="france"),
    )
    pairs = _all_pairs_generator().generate(collection)
    assert len(pairs) == 1


def test_three_mentions_same_subtopic_yields_three_pairs() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m2", subtopic_id="s1", mention_head_lemma="france"),
        _make(mention_id="m3", subtopic_id="s1", mention_head_lemma="day"),
    )
    pairs = _all_pairs_generator().generate(collection)
    assert len(pairs) == 3


def test_single_mention_yields_no_pairs() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
    )
    pairs = _all_pairs_generator().generate(collection)
    assert len(pairs) == 0


def test_pairs_only_within_subtopic_boundary() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m2", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m3", subtopic_id="s2", mention_head_lemma="canada"),
        _make(mention_id="m4", subtopic_id="s2", mention_head_lemma="canada"),
    )
    pairs = _all_pairs_generator().generate(collection)
    assert len(pairs) == 2
    for pair in pairs:
        assert pair.mention_a.subtopic_id == pair.mention_b.subtopic_id


def test_all_mentions_in_subtopic_are_paired() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m2", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m3", subtopic_id="s1", mention_head_lemma="france"),
        _make(mention_id="m4", subtopic_id="s1", mention_head_lemma="france"),
    )
    pairs = _all_pairs_generator().generate(collection)
    assert len(pairs) == 6


# --- max_candidates ---


def test_all_pairs_returned_when_below_max() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m2", subtopic_id="s1", mention_head_lemma="canada"),
    )
    pairs = _all_pairs_generator(max_candidates=10).generate(collection)
    assert len(pairs) == 1


def test_max_candidates_limits_output() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m2", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m3", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m4", subtopic_id="s1", mention_head_lemma="canada"),
    )
    pairs = _all_pairs_generator(max_candidates=2, seed=0).generate(collection)
    assert len(pairs) == 2


def test_max_candidates_result_are_valid_pairs() -> None:
    mentions = [_make(mention_id=f"m{i}", subtopic_id="s1", mention_head_lemma="canada") for i in range(5)]
    collection = _collection(*mentions)
    pairs = _all_pairs_generator(max_candidates=3, seed=0).generate(collection)
    assert all(isinstance(p, CandidatePair) for p in pairs)


# --- seed reproducibility ---


def test_same_seed_produces_identical_results() -> None:
    mentions = [_make(mention_id=f"m{i}", subtopic_id="s1", mention_head_lemma="canada") for i in range(6)]
    collection = _collection(*mentions)
    run_a = _all_pairs_generator(max_candidates=5, seed=42).generate(collection)
    run_b = _all_pairs_generator(max_candidates=5, seed=42).generate(collection)
    assert run_a == run_b


def test_different_seeds_produce_different_results() -> None:
    mentions = [_make(mention_id=f"m{i}", subtopic_id="s1", mention_head_lemma="canada") for i in range(10)]
    collection = _collection(*mentions)
    run_a = _all_pairs_generator(max_candidates=5, seed=1).generate(collection)
    run_b = _all_pairs_generator(max_candidates=5, seed=2).generate(collection)
    assert run_a != run_b


# --- output ordering without sampling ---


def test_output_is_deterministic_without_seed() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m2", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m3", subtopic_id="s2", mention_head_lemma="france"),
        _make(mention_id="m4", subtopic_id="s2", mention_head_lemma="france"),
    )
    run_a = _all_pairs_generator().generate(collection)
    run_b = _all_pairs_generator().generate(collection)
    assert run_a == run_b


# --- max_candidates_per_mention ---


def _count_pairs_per_mention(pairs: list) -> dict[str, int]:
    degrees: dict[str, int] = {}
    for pair in pairs:
        for mention_id in (pair.mention_a.mention_id, pair.mention_b.mention_id):
            degrees[mention_id] = degrees.get(mention_id, 0) + 1
    return degrees


def test_max_per_mention_limits_degree() -> None:
    mentions = [_make(mention_id=f"m{i}", subtopic_id="s1", mention_head_lemma="canada") for i in range(5)]
    collection = _collection(*mentions)
    pairs = _all_pairs_generator(max_candidates_per_mention=2, seed=0).generate(collection)
    degrees = _count_pairs_per_mention(pairs)
    assert all(degree <= 2 for degree in degrees.values())


def test_max_per_mention_no_effect_when_group_is_small() -> None:
    collection = _collection(
        _make(mention_id="m1", subtopic_id="s1", mention_head_lemma="canada"),
        _make(mention_id="m2", subtopic_id="s1", mention_head_lemma="canada"),
    )
    pairs = _all_pairs_generator(max_candidates_per_mention=10, seed=0).generate(collection)
    assert len(pairs) == 1


def test_max_per_mention_one_yields_at_most_one_pair_per_mention() -> None:
    mentions = [_make(mention_id=f"m{i}", subtopic_id="s1", mention_head_lemma="canada") for i in range(6)]
    collection = _collection(*mentions)
    pairs = _all_pairs_generator(max_candidates_per_mention=1, seed=0).generate(collection)
    degrees = _count_pairs_per_mention(pairs)
    assert all(degree <= 1 for degree in degrees.values())


def test_max_per_mention_same_seed_reproducible() -> None:
    mentions = [_make(mention_id=f"m{i}", subtopic_id="s1", mention_head_lemma="canada") for i in range(6)]
    collection = _collection(*mentions)
    run_a = _all_pairs_generator(max_candidates_per_mention=2, seed=7).generate(collection)
    run_b = _all_pairs_generator(max_candidates_per_mention=2, seed=7).generate(collection)
    assert run_a == run_b


def test_max_per_mention_different_seeds_differ() -> None:
    mentions = [_make(mention_id=f"m{i}", subtopic_id="s1", mention_head_lemma="canada") for i in range(8)]
    collection = _collection(*mentions)
    run_a = _all_pairs_generator(max_candidates_per_mention=2, seed=1).generate(collection)
    run_b = _all_pairs_generator(max_candidates_per_mention=2, seed=2).generate(collection)
    assert run_a != run_b


def test_max_per_mention_combined_with_max_candidates() -> None:
    mentions = [_make(mention_id=f"m{i}", subtopic_id="s1", mention_head_lemma="canada") for i in range(6)]
    collection = _collection(*mentions)
    pairs = _all_pairs_generator(max_candidates_per_mention=3, max_candidates=2, seed=0).generate(collection)
    assert len(pairs) == 2
    degrees = _count_pairs_per_mention(pairs)
    assert all(degree <= 3 for degree in degrees.values())
