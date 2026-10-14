from pydantic import BaseModel

from llm_ucdcr.candidates.candidate_pair import CandidatePair
from llm_ucdcr.data.mention import Mention


class MentionView(BaseModel):
    text: str
    head: str
    context: str

    @classmethod
    def from_mention(cls, mention: Mention) -> "MentionView":
        return cls(
            text=mention.tokens_str,
            head=mention.mention_head,
            context=cls._build_context(mention.mention_context, mention.tokens_number_context),
        )

    @staticmethod
    def _build_context(context_tokens: tuple[str, ...], mention_indices: tuple[int, ...]) -> str:
        mention_set = set(mention_indices)
        result: list[str] = []
        in_mention = False
        for i, token in enumerate(context_tokens):
            is_mention = i in mention_set
            if is_mention and not in_mention:
                result.append("<mention>")
                in_mention = True
            elif not is_mention and in_mention:
                result.append("</mention>")
                in_mention = False
            result.append(token)
        if in_mention:
            result.append("</mention>")
        return " ".join(result)


class PairView(BaseModel):
    mention_a: MentionView
    mention_b: MentionView

    @classmethod
    def from_pair(cls, pair: CandidatePair) -> "PairView":
        return cls(
            mention_a=MentionView.from_mention(pair.mention_a),
            mention_b=MentionView.from_mention(pair.mention_b),
        )


class LabeledPairView(BaseModel):
    mention_a: MentionView
    mention_b: MentionView
    coreferent: bool

    @classmethod
    def from_pair(cls, pair: CandidatePair) -> "LabeledPairView":
        return cls(
            mention_a=MentionView.from_mention(pair.mention_a),
            mention_b=MentionView.from_mention(pair.mention_b),
            coreferent=pair.is_coreferent,
        )


class CoreferenceDecision(BaseModel):
    coreferent: bool
