from collections.abc import Sequence

from openai.types.chat import ChatCompletionMessageParam

from llm_ucdcr.approaches.base import PairDecision, PairPredictor
from llm_ucdcr.candidates.candidate_pair import CandidatePair
from llm_ucdcr.llm.connector import LLMConnector

from .models import CoreferenceDecision, LabeledPairView, MentionView, PairView

_SYSTEM_PROMPT = (
    "You are an expert in cross-document coreference resolution (CDCR). "
    "Your task is to decide whether two entity or event mentions, which may appear in different documents, "
    "refer to the same real-world entity or event.\n\n"
    "You will first be shown a set of labeled example pairs to guide your reasoning, "
    "followed by a candidate pair that you must classify. "
    "Each mention is presented with its surface text, its syntactic head word, and the surrounding "
    "passage with the mention enclosed in <mention> … </mention> tags.\n\n"
    "Base your decision on the semantics and context of both mentions. "
    "Return whether they are coreferent."
)


class RandomFewShotPredictor(PairPredictor):

    """Few-shot CDCR predictor over an LLM.

    Owns the prompt-building business logic: holds the few-shot pool bound by the runner
    and, for each candidate, formats the system prompt plus a single user message and asks
    the LLM connector for a structured decision. LLMError raised by the connector propagates
    to the runner, which turns it into a failed PredictionRecord.
    """

    def __init__(self, connector: LLMConnector) -> None:
        self._connector = connector
        self._examples: list[LabeledPairView] = []

    def bind_examples(self, examples: Sequence[LabeledPairView]) -> None:
        self._examples = list(examples)

    async def predict(self, pair: CandidatePair) -> PairDecision:
        decision = await self._connector.predict(
            system_prompt=_SYSTEM_PROMPT,
            messages=self._to_messages(PairView.from_pair(pair)),
            output_model=CoreferenceDecision,
        )
        return PairDecision(coreferent=decision.coreferent)

    def _to_messages(self, candidate: PairView) -> list[ChatCompletionMessageParam]:
        parts: list[str] = []
        if self._examples:
            parts.append("## Examples")
            for i, example in enumerate(self._examples, 1):
                label = "coreferent" if example.coreferent else "not coreferent"
                parts.append(
                    f"### Example {i}\n"
                    f"{self._format_pair(example.mention_a, example.mention_b)}\n"
                    f"Decision: {label}",
                )
        parts.append(
            f"## Candidate pair\n\n{self._format_pair(candidate.mention_a, candidate.mention_b)}",
        )
        return [{"role": "user", "content": "\n\n".join(parts)}]

    @staticmethod
    def _format_pair(mention_a: MentionView, mention_b: MentionView) -> str:
        return (
            f"Mention A: {mention_a.text} (head: {mention_a.head})\n"
            f"Context: {mention_a.context}\n\n"
            f"Mention B: {mention_b.text} (head: {mention_b.head})\n"
            f"Context: {mention_b.context}"
        )
