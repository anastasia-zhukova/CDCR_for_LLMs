from dataclasses import dataclass

from llm_ucdcr.data.mention import Mention


@dataclass(frozen=True)
class CandidatePair:
    mention_a: Mention
    mention_b: Mention

    @property
    def is_coreferent(self) -> bool:
        return self.mention_a.gold_cluster_key == self.mention_b.gold_cluster_key
