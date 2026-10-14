"""Gold clusters are scoped per subtopic, so a chain spanning subtopics is not punished.

Coreference is resolved one subtopic at a time, and no approach in this repo can link two
mentions in different subtopics. To keep scoring fair, the gold clustering is keyed by
Mention.gold_cluster_key = (coref_chain, subtopic_id) rather than by coref_chain alone
(see MentionCollection.gold_clusters): a gold chain that spans subtopics becomes one gold
cluster per subtopic — exactly the shape a subtopic-level system can reach.

These tests pin that contract down. A flawless within-subtopic clustering scores a perfect 1.0
even when the underlying chain spans subtopics, because gold no longer expects the cross-subtopic
merge. The control (a chain confined to one subtopic) also scores 1.0, and a chain spread over
more subtopics still scores 1.0 — the cross-subtopic split is no longer a recall penalty.
"""

from collections import defaultdict
from typing import Any

import pytest

from llm_ucdcr.data.loader import MentionCollection
from llm_ucdcr.data.mention import Mention
from llm_ucdcr.experiments.runner import score_clusters

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


def _best_subtopic_clusters(mentions: MentionCollection) -> list[list[str]]:
    """The ceiling for any subtopic-scoped system: gold chains, cut along subtopic boundaries.

    Mentions of the same gold chain that share a subtopic land together (perfect within-subtopic
    recall); mentions of that chain in another subtopic become a separate cluster, because no
    subtopic-level approach can ever join them. Under the per-subtopic gold key this is now exactly
    the gold clustering, so it must score perfectly.
    """
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for mention in mentions:
        groups[(mention.coref_chain, mention.subtopic_id)].append(mention.mention_id)
    return list(groups.values())


def _spanning_chain() -> MentionCollection:
    """One gold chain of three mentions split across two subtopics (2 in s1, 1 in s2)."""
    return MentionCollection(
        [
            _make(mention_id="m1", coref_chain="c", subtopic_id="s1"),
            _make(mention_id="m2", coref_chain="c", subtopic_id="s1"),
            _make(mention_id="m3", coref_chain="c", subtopic_id="s2"),
        ],
    )


def _confined_chain() -> MentionCollection:
    """Same three-mention chain, but all in one subtopic — the control with no cross-subtopic split."""
    return MentionCollection(
        [
            _make(mention_id="m1", coref_chain="c", subtopic_id="s1"),
            _make(mention_id="m2", coref_chain="c", subtopic_id="s1"),
            _make(mention_id="m3", coref_chain="c", subtopic_id="s1"),
        ],
    )


def test_gold_clusters_split_a_chain_along_subtopic_boundaries() -> None:
    """A chain spanning two subtopics is two gold clusters, keyed by (coref_chain, subtopic)."""
    mentions = _spanning_chain()

    gold = mentions.gold_clusters()

    assert {frozenset(cluster) for cluster in gold} == {frozenset({"m1", "m2"}), frozenset({"m3"})}


def test_gold_clusters_keep_a_within_subtopic_chain_whole() -> None:
    """When the chain stays in one subtopic, the per-subtopic key leaves it as a single cluster."""
    mentions = _confined_chain()

    gold = mentions.gold_clusters()

    assert len(gold) == 1
    assert sorted(gold[0]) == ["m1", "m2", "m3"]


def test_cross_subtopic_chain_is_not_punished() -> None:
    """The core contract: the best within-subtopic clustering scores 1.0 despite the spanning chain.

    Gold no longer expects the cross-subtopic merge, so the unavoidable split costs nothing.
    """
    mentions = _spanning_chain()

    metrics = score_clusters(mentions.gold_clusters(), _best_subtopic_clusters(mentions))

    for metric in ("muc_f1", "b_cubed_f1", "ceafe_f1", "conll_f1"):
        assert metrics[metric] == pytest.approx(1.0), f"{metric} should be perfect: {metrics[metric]}"
    for metric in ("muc_recall", "b_cubed_recall", "muc_precision", "b_cubed_precision"):
        assert metrics[metric] == pytest.approx(1.0), f"{metric} should be perfect: {metrics[metric]}"


def test_confined_chain_also_scores_perfectly() -> None:
    """Control: a chain inside one subtopic scores perfectly too — the spanning case now matches it."""
    mentions = _confined_chain()

    metrics = score_clusters(mentions.gold_clusters(), _best_subtopic_clusters(mentions))

    for metric in ("muc_f1", "b_cubed_f1", "ceafe_f1", "conll_f1"):
        assert metrics[metric] == pytest.approx(1.0), f"{metric} should be perfect: {metrics[metric]}"


def test_spreading_a_chain_over_more_subtopics_still_scores_perfectly() -> None:
    """No matter how many subtopics a chain is spread over, the per-subtopic ceiling stays at 1.0.

    Two mentions per subtopic so each gold cluster has a real coreference link (MUC is degenerate
    for all-singleton clusterings), isolating the cross-subtopic split as the thing under test.
    """
    three_way = MentionCollection(
        [
            _make(mention_id="m1a", coref_chain="c", subtopic_id="s1"),
            _make(mention_id="m1b", coref_chain="c", subtopic_id="s1"),
            _make(mention_id="m2a", coref_chain="c", subtopic_id="s2"),
            _make(mention_id="m2b", coref_chain="c", subtopic_id="s2"),
            _make(mention_id="m3a", coref_chain="c", subtopic_id="s3"),
            _make(mention_id="m3b", coref_chain="c", subtopic_id="s3"),
        ],
    )

    metrics = score_clusters(three_way.gold_clusters(), _best_subtopic_clusters(three_way))

    for metric in ("muc_f1", "b_cubed_f1", "ceafe_f1", "conll_f1"):
        assert metrics[metric] == pytest.approx(1.0), f"{metric} should be perfect: {metrics[metric]}"
