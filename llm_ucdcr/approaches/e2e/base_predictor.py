import random
from abc import abstractmethod

from openai.types.chat import ChatCompletionMessageParam

from llm_ucdcr.approaches.base import SubtopicDecision, SubtopicPredictor
from llm_ucdcr.data.loader import MentionCollection
from llm_ucdcr.llm.connector import LLMConnector
from llm_ucdcr.logs.accessors import get_logger

from .models import ClusterAssignment
from .token_estimator import PromptTokenEstimator

logger = get_logger(__name__)


class BaseE2EPredictor(SubtopicPredictor):

    """Shared machinery for end-to-end CDCR predictors over an LLM.

    Owns the call flow common to every end-to-end variant: estimate and log the prompt size, ask
    the connector for a ClusterAssignment, then sanitize the raw clusters into disjoint chains of
    real mention ids. Subclasses supply the system prompt and decide how the subtopic's mentions are
    rendered into the user message (``system_prompt`` and ``_build_messages``). LLMError raised by
    the connector propagates to the runner, which records the subtopic as failed.
    """

    system_prompt: str

    def __init__(
        self,
        connector: LLMConnector,
        estimator: PromptTokenEstimator,
        rng: random.Random | None = None,
    ) -> None:
        self._connector = connector
        self._estimator = estimator
        # Injected so tests can seed the duplicate-id assignment; production stays non-deterministic,
        # which is fine since the LLM responses are non-deterministic anyway. Not security-sensitive.
        self._rng = rng if rng is not None else random.Random()  # noqa: S311

    @abstractmethod
    def _build_messages(self, mentions: MentionCollection) -> list[ChatCompletionMessageParam]: ...

    async def predict(self, mentions: MentionCollection) -> SubtopicDecision:
        known_ids = {mention.mention_id for mention in mentions}
        messages = self._build_messages(mentions)
        self._log_estimate(messages)
        assignment = await self._connector.predict(
            system_prompt=self.system_prompt,
            messages=messages,
            output_model=ClusterAssignment,
        )
        clusters = self._sanitize(assignment.clusters, known_ids)
        return SubtopicDecision(clusters=clusters)

    def _log_estimate(self, messages: list[ChatCompletionMessageParam]) -> None:
        text = self.system_prompt + "\n" + "".join(str(message["content"]) for message in messages)
        logger.info("Estimated subtopic prompt tokens", estimated_tokens=self._estimator.count(text))

    def _sanitize(self, clusters: list[list[str]], known_ids: set[str]) -> list[list[str]]:
        """Clean the raw LLM clusters into disjoint chains of real mention ids.

        Drops ids the model invented (not in the subtopic). When an id appears in several chains,
        it is kept in one chosen at random among them. Resulting single-element or empty chains are
        dropped — the runner turns leftover ids into singletons.
        """
        candidates: dict[str, list[int]] = {}
        for index, cluster in enumerate(clusters):
            for mention_id in cluster:
                if mention_id in known_ids:
                    candidates.setdefault(mention_id, []).append(index)
        chosen: dict[int, list[str]] = {}
        for mention_id, indices in candidates.items():
            target = indices[0] if len(indices) == 1 else self._rng.choice(indices)
            chosen.setdefault(target, []).append(mention_id)
        return [members for members in chosen.values() if len(members) > 1]
