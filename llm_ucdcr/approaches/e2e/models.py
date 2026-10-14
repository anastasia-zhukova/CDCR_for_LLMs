from pydantic import BaseModel, Field

from llm_ucdcr.approaches.random_few_shot.models import MentionView
from llm_ucdcr.data.mention import Mention


class E2EMentionView(MentionView):

    """A MentionView that also carries its mention id.

    The end-to-end prompt lists every mention up front and the LLM answers with chains of ids, so
    unlike the pairwise MentionView each entry needs to be addressable by id. Everything else (the
    surface text, head, and <mention>-tagged context) is exactly the pairwise view, so we extend it
    rather than re-derive the same fields.
    """

    mention_id: str

    @classmethod
    def from_mention(cls, mention: Mention) -> "E2EMentionView":
        return cls(
            mention_id=mention.mention_id,
            text=mention.tokens_str,
            head=mention.mention_head,
            context=cls._build_context(mention.mention_context, mention.tokens_number_context),
        )


class ClusterAssignment(BaseModel):

    """Output model the LLM returns: coreference chains as groups of mention ids.

    Mentions omitted from every cluster are singletons; the runner fills them in.
    """

    clusters: list[list[str]] = Field(
        description=(
            "The coreference chains. Each inner list is one chain: the mention ids that all refer to "
            "the same real-world entity or event. Use only ids from the input and put each id in at "
            "most one chain; omit ids that do not corefer with anything (they become singletons)."
        ),
    )
