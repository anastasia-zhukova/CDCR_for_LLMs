import json
from pathlib import Path
from typing import Any

import pytest

from llm_ucdcr.data import DatasetLoader, DatasetName, Mention, MentionCollection, MentionKind, Split


_MENTION_BASE: dict[str, Any] = {
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
    "topic_id": "t1",
    "topic": "topic_one",
    "subtopic_id": "s1",
    "subtopic": "subtopic_one",
    "doc_id": "d1",
    "doc": "doc_one",
    "mention_context": ["Some", "context", "here"],
    "context_start_end_global_id": [0, 3],
    "mention_sentence_start_end": [0, 3],
    "tokens_number_context": [1],
    "mention_head_id_context": 1,
    "is_singleton": False,
    "conll_doc_key": "t1/s1/d1",
}


def _mention_dict(**overrides: Any) -> dict[str, Any]:
    return {**_MENTION_BASE, **overrides}


@pytest.fixture
def collection() -> MentionCollection:
    mentions = [
        Mention.from_dict(_mention_dict(mention_id="m1", topic_id="t1", subtopic_id="s1", doc_id="d1", coref_chain="chain1", conll_doc_key="t1/s1/d1")),
        Mention.from_dict(_mention_dict(mention_id="m2", topic_id="t1", subtopic_id="s1", doc_id="d2", coref_chain="chain1", conll_doc_key="t1/s1/d2")),
        Mention.from_dict(_mention_dict(mention_id="m3", topic_id="t2", subtopic_id="s2", doc_id="d3", coref_chain="chain2", conll_doc_key="t2/s2/d3")),
    ]
    return MentionCollection(mentions, dataset=DatasetName.ECBplus)


@pytest.fixture
def loader_with_data(tmp_path: Path) -> DatasetLoader:
    # Distinct ids per physical kind so a BOTH load can be asserted to pool both files.
    per_kind = {
        MentionKind.ENTITY: [_mention_dict(mention_id="ent1")],
        MentionKind.EVENT: [_mention_dict(mention_id="evt1"), _mention_dict(mention_id="evt2")],
    }
    for split in Split:
        split_dir = tmp_path / DatasetName.ECBplus.value / split.value
        split_dir.mkdir(parents=True, exist_ok=True)
        for kind, mention_dicts in per_kind.items():
            (split_dir / f"{kind.value}_mentions.json").write_text(json.dumps(mention_dicts))
    return DatasetLoader(base_path=tmp_path)


# --- MentionCollection ---

def test_collection_len(collection: MentionCollection) -> None:
    assert len(collection) == 3


def test_collection_iter(collection: MentionCollection) -> None:
    mentions = list(collection)
    assert len(mentions) == 3
    assert all(isinstance(m, Mention) for m in mentions)


def test_collection_getitem(collection: MentionCollection) -> None:
    assert collection[0].mention_id == "m1"
    assert collection[2].mention_id == "m3"


def test_collection_repr(collection: MentionCollection) -> None:
    result = repr(collection)
    assert "ECBplus" in result
    assert "3" in result


def test_get_returns_mention(collection: MentionCollection) -> None:
    mention = collection.get("m2")
    assert mention is not None
    assert mention.mention_id == "m2"


def test_get_returns_none_for_missing(collection: MentionCollection) -> None:
    assert collection.get("does_not_exist") is None


def test_by_topic_groups_correctly(collection: MentionCollection) -> None:
    groups = collection.by_topic()
    assert set(groups.keys()) == {"t1", "t2"}
    assert len(groups["t1"]) == 2
    assert len(groups["t2"]) == 1


def test_by_subtopic_groups_correctly(collection: MentionCollection) -> None:
    groups = collection.by_subtopic()
    assert set(groups.keys()) == {"s1", "s2"}
    assert len(groups["s1"]) == 2


def test_by_doc_groups_correctly(collection: MentionCollection) -> None:
    groups = collection.by_doc()
    assert set(groups.keys()) == {"d1", "d2", "d3"}
    assert all(len(g) == 1 for g in groups.values())


def test_by_coref_chain_groups_correctly(collection: MentionCollection) -> None:
    groups = collection.by_coref_chain()
    assert set(groups.keys()) == {"chain1", "chain2"}
    assert len(groups["chain1"]) == 2


def test_by_conll_key_groups_correctly(collection: MentionCollection) -> None:
    groups = collection.by_conll_key()
    assert len(groups) == 3


def test_without_singletons_drops_per_subtopic_singletons() -> None:
    """A chain spanning subtopics keeps the multi-mention part and drops the lone-in-subtopic part.

    All four mentions have is_singleton=False (chain has >1 member globally), so the filter must
    rely on the per-subtopic gold key, not the dataset field.
    """
    mentions = [
        Mention.from_dict(_mention_dict(mention_id="m1", coref_chain="c", subtopic_id="s1")),
        Mention.from_dict(_mention_dict(mention_id="m2", coref_chain="c", subtopic_id="s1")),
        Mention.from_dict(_mention_dict(mention_id="m3", coref_chain="c", subtopic_id="s2")),
        Mention.from_dict(_mention_dict(mention_id="m4", coref_chain="d", subtopic_id="s2")),
    ]
    kept = list(MentionCollection(mentions).without_singletons())

    assert {m.mention_id for m in kept} == {"m1", "m2"}


def test_without_singletons_keeps_a_within_subtopic_chain() -> None:
    """When a chain stays inside one subtopic, none of its mentions are per-subtopic singletons."""
    mentions = [
        Mention.from_dict(_mention_dict(mention_id="m1", coref_chain="c", subtopic_id="s1")),
        Mention.from_dict(_mention_dict(mention_id="m2", coref_chain="c", subtopic_id="s1")),
    ]
    kept = list(MentionCollection(mentions).without_singletons())

    assert {m.mention_id for m in kept} == {"m1", "m2"}


def test_group_preserves_dataset(collection: MentionCollection) -> None:
    groups = collection.by_topic()
    assert all(g.dataset == DatasetName.ECBplus for g in groups.values())


def test_group_returns_mention_collections(collection: MentionCollection) -> None:
    groups = collection.by_topic()
    assert all(isinstance(g, MentionCollection) for g in groups.values())


# --- DatasetLoader ---

def test_loader_load_returns_mention_collection(loader_with_data: DatasetLoader) -> None:
    result = loader_with_data.load(DatasetName.ECBplus, Split.TEST)
    assert isinstance(result, MentionCollection)


def test_loader_load_assigns_dataset(loader_with_data: DatasetLoader) -> None:
    result = loader_with_data.load(DatasetName.ECBplus, Split.TEST)
    assert result.dataset == DatasetName.ECBplus


def test_loader_load_creates_mention_instances(loader_with_data: DatasetLoader) -> None:
    result = loader_with_data.load(DatasetName.ECBplus, Split.TEST)
    assert len(result) == 1
    assert isinstance(result[0], Mention)


def test_loader_load_event_kind(loader_with_data: DatasetLoader) -> None:
    result = loader_with_data.load(DatasetName.ECBplus, Split.TRAIN, kind=MentionKind.EVENT)
    assert isinstance(result, MentionCollection)
    assert len(result) == 2


def test_loader_load_both_pools_entity_and_event(loader_with_data: DatasetLoader) -> None:
    result = loader_with_data.load(DatasetName.ECBplus, Split.TEST, kind=MentionKind.BOTH)
    assert {m.mention_id for m in result} == {"ent1", "evt1", "evt2"}


def test_mention_kind_file_kinds() -> None:
    assert MentionKind.ENTITY.file_kinds == (MentionKind.ENTITY,)
    assert MentionKind.EVENT.file_kinds == (MentionKind.EVENT,)
    assert MentionKind.BOTH.file_kinds == (MentionKind.ENTITY, MentionKind.EVENT)


def test_loader_load_splits_returns_all_splits(loader_with_data: DatasetLoader) -> None:
    splits = loader_with_data.load_splits(DatasetName.ECBplus)
    assert set(splits.keys()) == set(Split)
    assert all(isinstance(v, MentionCollection) for v in splits.values())
