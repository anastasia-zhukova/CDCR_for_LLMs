from __future__ import annotations

import json
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from collections.abc import Iterator

from llm_ucdcr.logs.accessors import get_logger

from .mention import Mention

logger = get_logger(__name__)


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MentionKind(StrEnum):
    ENTITY = "entity"
    EVENT = "event"
    BOTH = "both"  # mixed: load entity and event mentions into one pool

    @property
    def file_kinds(self) -> tuple[MentionKind, ...]:
        """Physical mention files backing this kind.

        ENTITY/EVENT map to their own file; BOTH spans both so the two mention
        types are pooled and scored together.
        """
        if self is MentionKind.BOTH:
            return (MentionKind.ENTITY, MentionKind.EVENT)
        return (self,)


class DatasetName(StrEnum):
    CD2CR = "CD2CR"
    CEREC = "CEREC"
    ECBplus = "ECBplus"
    ECBplusMETAm = "ECBplusMETAm"
    FCC = "FCC"
    GVC = "GVC"
    HyperCoref = "HyperCoref"
    MEANTIME = "MEANTIME"
    NewsWCL50 = "NewsWCL50"
    NiDENT = "NiDENT"
    NP4E = "NP4E"
    WECEng = "WECEng"


class MentionCollection:
    def __init__(
        self,
        mentions: list[Mention],
        dataset: DatasetName | None = None,
    ) -> None:
        self._mentions = mentions
        self.dataset = dataset
        self._index: dict[str, Mention] | None = None

    def __len__(self) -> int:
        return len(self._mentions)

    def __iter__(self) -> Iterator[Mention]:
        return iter(self._mentions)

    def __getitem__(self, idx: int) -> Mention:
        return self._mentions[idx]

    def __repr__(self) -> str:
        return f"MentionCollection(dataset={self.dataset!r}, n={len(self)})"

    def get(self, mention_id: str) -> Mention | None:
        if self._index is None:
            self._index = {m.mention_id: m for m in self._mentions}
        return self._index.get(mention_id)

    def by_topic(self) -> dict[str, MentionCollection]:
        return self._group_by("topic_id")

    def by_subtopic(self) -> dict[str, MentionCollection]:
        return self._group_by("subtopic_id")

    def by_doc(self) -> dict[str, MentionCollection]:
        return self._group_by("doc_id")

    def by_coref_chain(self) -> dict[str, MentionCollection]:
        return self._group_by("coref_chain")

    def by_conll_key(self) -> dict[str, MentionCollection]:
        return self._group_by("conll_doc_key")

    def by_head_lemma(self) -> dict[str, MentionCollection]:
        return self._group_by("mention_head_lemma")

    def by_normalized_head_lemma(self) -> dict[str, MentionCollection]:
        return self._group_by("mention_head_lemma_normalized")

    def without_singletons(self) -> MentionCollection:
        """Drop mentions that are alone in their gold cluster, computed per subtopic.

        A singleton is a gold cluster of size one under Mention.gold_cluster_key
        (coref_chain, subtopic) — consistent with gold_clusters(). This is stricter than the
        dataset's global is_singleton field: a chain spanning subtopics can leave a lone mention
        in one subtopic that is a singleton under the per-subtopic gold key even though its chain
        has members elsewhere. Counts are relative to this collection, so subsets (e.g. a per-split
        few-shot pool) are filtered against their own membership.
        """
        key_counts: Counter[tuple[str, str]] = Counter(m.gold_cluster_key for m in self._mentions)
        return MentionCollection([m for m in self._mentions if key_counts[m.gold_cluster_key] > 1], self.dataset)

    def gold_clusters(self) -> list[list[str]]:
        """Gold coreference clusters as groups of mention ids, one list per (coref chain, subtopic).

        Keyed on Mention.gold_cluster_key: coreference is scored per subtopic, so a chain that
        spans subtopics yields one gold cluster per subtopic rather than a single merged cluster.
        """
        clusters: dict[tuple[str, str], list[str]] = defaultdict(list)
        for m in self._mentions:
            clusters[m.gold_cluster_key].append(m.mention_id)
        return list(clusters.values())

    def _group_by(self, field: str) -> dict[str, MentionCollection]:
        groups: dict[str, list[Mention]] = defaultdict(list)
        for m in self._mentions:
            groups[getattr(m, field)].append(m)
        return {k: MentionCollection(v, self.dataset) for k, v in groups.items()}


class DatasetLoader:
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        logger.debug("DatasetLoader initialized", base_path=str(base_path))

    def load(
        self,
        dataset: DatasetName,
        split: Split,
        kind: MentionKind = MentionKind.ENTITY,
    ) -> MentionCollection:
        # For BOTH this concatenates the entity and event files; mention ids and
        # coref chains are globally unique across the two, so pooling them is safe.
        mentions: list[Mention] = []
        for file_kind in kind.file_kinds:
            mentions.extend(self._load_file(dataset, split, file_kind))
        collection = MentionCollection(mentions, dataset=dataset)
        logger.info("Loaded mentions", dataset=dataset.value, split=split.value, kind=kind.value, count=len(collection))
        return collection

    def _load_file(self, dataset: DatasetName, split: Split, kind: MentionKind) -> list[Mention]:
        path = self.base_path / dataset.value / split.value / f"{kind.value}_mentions.json"
        logger.debug("Loading mentions", dataset=dataset.value, split=split.value, kind=kind.value, path=str(path))
        if not path.exists():
            logger.error(
                "Mentions file not found",
                path=str(path),
                dataset=dataset.value,
                split=split.value,
                kind=kind.value,
            )
            raise FileNotFoundError(f"Mentions file not found: {path}")
        with path.open() as f:
            raw: list[dict] = json.load(f)
        mentions = [Mention.from_dict(d) for d in raw]
        return self._drop_duplicates(mentions, dataset, split, kind)

    @staticmethod
    def _drop_duplicates(
        mentions: list[Mention],
        dataset: DatasetName,
        split: Split,
        kind: MentionKind,
    ) -> list[Mention]:
        """Drop fully-duplicated mention rows, keeping the first occurrence.

        Some dataset releases ship rows that are identical across every field;
        left in, they inflate cluster-alignment metrics past 1.0. Dedup is by the
        whole mention, not by mention_id alone: some datasets (e.g. CD2CR) reuse a
        mention_id for genuinely distinct mentions, which must be kept.
        """
        seen: set[Mention] = set()
        deduped: list[Mention] = []
        for mention in mentions:
            if mention in seen:
                continue
            seen.add(mention)
            deduped.append(mention)
        dropped = len(mentions) - len(deduped)
        if dropped:
            logger.warning(
                "Dropped duplicate mentions",
                dataset=dataset.value,
                split=split.value,
                kind=kind.value,
                dropped=dropped,
            )
        return deduped

    def load_splits(
        self,
        dataset: DatasetName,
        kind: MentionKind = MentionKind.ENTITY,
    ) -> dict[Split, MentionCollection]:
        logger.debug("Loading all splits", dataset=dataset.value, kind=kind.value)
        return {split: self.load(dataset, split, kind) for split in Split}
