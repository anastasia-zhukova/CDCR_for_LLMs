from llm_ucdcr.approaches.base import PairDecision, PairPredictor
from llm_ucdcr.candidates.candidate_pair import CandidatePair

from .strategy import BaselineStrategy


class BaselinePredictor(PairPredictor):

    """LLM-free predictor that applies a fixed labeling rule to every candidate pair.

    Reuses the real CandidateGenerator and cluster scoring through the runner, so the
    CoNLL numbers match exactly what an LLM run on the same candidate set would report
    — only the per-pair decision is stubbed. See BaselineStrategy for the rules.
    """

    def __init__(self, strategy: BaselineStrategy) -> None:
        self._strategy = strategy

    async def predict(self, pair: CandidatePair) -> PairDecision:
        match self._strategy:
            case BaselineStrategy.CONSTANT_TRUE:
                return PairDecision(coreferent=True)
            case BaselineStrategy.ORACLE:
                return PairDecision(coreferent=pair.is_coreferent)
            case BaselineStrategy.HEAD_LEMMA:
                return PairDecision(
                    coreferent=pair.mention_a.mention_head_lemma_normalized
                    == pair.mention_b.mention_head_lemma_normalized,
                )
            case _:
                raise ValueError(f"Unknown baseline strategy: {self._strategy}")
