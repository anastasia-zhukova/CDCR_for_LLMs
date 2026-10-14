from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llm_ucdcr.approaches.random_few_shot.models import LabeledPairView
    from llm_ucdcr.candidates.candidate_pair import CandidatePair
    from llm_ucdcr.data.loader import MentionCollection


@dataclass(frozen=True)
class PairDecision:

    """A predictor's verdict on a single candidate pair."""

    coreferent: bool
    reasoning: str | None = None


@dataclass(frozen=True)
class SubtopicDecision:

    """A predictor's coreference chains for a whole subtopic.

    ``clusters`` lists the predicted chains as groups of mention ids; any mention in the
    subtopic absent from every cluster is treated as a singleton by the runner.
    """

    clusters: list[list[str]]


class PairPredictor(ABC):

    """Maps a candidate pair to a coreference decision.

    This is the seam the ExperimentRunner scores against, decoupled from *how* the
    decision is reached: the random_few_shot approach prompts an LLM, the baseline
    approach applies a fixed rule. Implementations may raise llm.errors.LLMError to
    signal a recoverable per-pair failure; the runner records it and moves on.
    """

    def bind_examples(self, examples: Sequence[LabeledPairView]) -> None:  # noqa: B027
        """Receive the few-shot pool before scoring starts.

        An optional hook: approaches that don't use in-context examples (e.g. the
        baselines) inherit this no-op rather than overriding it.
        """

    @abstractmethod
    async def predict(self, pair: CandidatePair) -> PairDecision: ...


class SubtopicPredictor(ABC):

    """Maps all mentions of a single subtopic to predicted coreference chains.

    The end-to-end counterpart to PairPredictor: instead of one verdict per candidate pair, the
    approach receives every mention in a subtopic and returns the clusters directly, folding
    candidate generation and decision-making into one step. Implementations may raise
    llm.errors.LLMError to signal a recoverable per-subtopic failure; the runner records it and
    treats that subtopic's mentions as singletons.
    """

    @abstractmethod
    async def predict(self, mentions: MentionCollection) -> SubtopicDecision: ...
