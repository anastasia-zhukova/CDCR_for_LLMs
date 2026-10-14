import itertools
import random
from collections import defaultdict

from llm_ucdcr.data.loader import MentionCollection
from llm_ucdcr.data.mention import Mention
from llm_ucdcr.logs.accessors import get_logger

from .candidate_pair import CandidatePair
from .strategy import CandidateStrategy

logger = get_logger(__name__)


class CandidateGenerator:
    def __init__(
        self,
        *,
        strategy: CandidateStrategy,
        max_candidates: int | None = None,
        max_candidates_per_mention: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._max_candidates = max_candidates
        self._max_candidates_per_mention = max_candidates_per_mention
        self._seed = seed
        self._strategy = strategy

    def generate(self, collection: MentionCollection) -> list[CandidatePair]:
        uses_sampling = self._max_candidates is not None or self._max_candidates_per_mention is not None
        if uses_sampling and self._seed is None:
            logger.warning("Sampling is configured without a seed — results are not reproducible")
        rng = random.Random(self._seed)
        all_pairs = self._generate_all_pairs(collection, rng)
        logger.debug(
            "Generated candidate pairs",
            total=len(all_pairs),
            max_candidates=self._max_candidates,
            max_candidates_per_mention=self._max_candidates_per_mention,
        )
        if self._max_candidates is None or len(all_pairs) <= self._max_candidates:
            return all_pairs
        sampled = rng.sample(all_pairs, self._max_candidates)
        logger.info(
            "Sampled candidate pairs",
            requested=self._max_candidates,
            total_available=len(all_pairs),
            seed=self._seed,
        )
        return sampled

    def _generate_all_pairs(self, collection: MentionCollection, rng: random.Random) -> list[CandidatePair]:
        pairs: list[CandidatePair] = []
        subtopics = collection.by_subtopic()
        for subtopic_id in sorted(subtopics):
            mentions = sorted(subtopics[subtopic_id], key=lambda m: m.mention_id)
            pairs.extend(self._pairs_for_mentions(mentions, rng))
        return pairs

    def _pairs_for_mentions(self, mentions: list[Mention], rng: random.Random) -> list[CandidatePair]:
        all_pairs = [
            CandidatePair(mention_a=mentions[i], mention_b=mentions[j])
            for i, j in itertools.combinations(range(len(mentions)), 2)
        ]
        if self._max_candidates_per_mention is None:
            return all_pairs
        rng.shuffle(all_pairs)
        degrees: dict[str, int] = defaultdict(int)
        selected: list[CandidatePair] = []
        for pair in all_pairs:
            a_id = pair.mention_a.mention_id
            b_id = pair.mention_b.mention_id
            if degrees[a_id] < self._max_candidates_per_mention and degrees[b_id] < self._max_candidates_per_mention:
                selected.append(pair)
                degrees[a_id] += 1
                degrees[b_id] += 1
        return selected
