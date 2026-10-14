from openai.types.chat import ChatCompletionMessageParam

from llm_ucdcr.data.loader import MentionCollection

from .base_predictor import BaseE2EPredictor
from .models import E2EMentionView

_SYSTEM_PROMPT = (
    "You are an expert in cross-document coreference resolution (CDCR). "
    "You are given every entity or event mention from a single subtopic, which may span several "
    "documents. Your task is to group the mentions that refer to the same real-world entity or "
    "event into coreference chains.\n\n"
    "Each mention is listed with a unique id, its surface text, its syntactic head word, and the "
    "surrounding passage with the mention enclosed in <mention> … </mention> tags.\n\n"
    "Include only ids that appear in the input, and place each id in at most one chain. Omit "
    "mentions that do not corefer with any other mention — these are treated as singletons "
    "automatically, so do not return single-element chains.\n\n"
    # DeepSeek lacks native structured-output enforcement, so spell the schema out in the prompt
    # (the connector also appends the JSON schema, but a concrete shape steers the model better).
    'Respond with a single JSON object {"clusters": [...]} whose "clusters" value is a list of '
    "chains, where each chain is a list of mention id strings. For example:\n"
    '{"clusters": [["m3", "m7"], ["m1", "m4", "m9"]]}'
)


class E2EPredictor(BaseE2EPredictor):

    """End-to-end CDCR predictor that lists every mention with its own context window.

    Each mention is rendered as a self-contained list entry (id, surface text, head, and its own
    left/right context), so a mention that appears in several documents repeats its passage. The
    document-grouped variant trades this redundancy for shared, once-rendered documents.
    """

    system_prompt = _SYSTEM_PROMPT

    def _build_messages(self, mentions: MentionCollection) -> list[ChatCompletionMessageParam]:
        views = [E2EMentionView.from_mention(mention) for mention in mentions]
        return self._to_messages(views)

    def _to_messages(self, views: list[E2EMentionView]) -> list[ChatCompletionMessageParam]:
        lines = ["## Mentions"]
        lines.extend(
            f"- id={view.mention_id}: {view.text} (head: {view.head})\n  Context: {view.context}" for view in views
        )
        return [{"role": "user", "content": "\n".join(lines)}]
