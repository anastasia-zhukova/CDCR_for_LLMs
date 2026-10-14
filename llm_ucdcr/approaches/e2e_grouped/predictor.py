import random

from openai.types.chat import ChatCompletionMessageParam

from llm_ucdcr.approaches.e2e import BaseE2EPredictor
from llm_ucdcr.data.loader import MentionCollection
from llm_ucdcr.llm.connector import LLMConnector
from llm_ucdcr.approaches.e2e.token_estimator import PromptTokenEstimator

from .document_renderer import GAP_MARKER, MENTION_CLOSE, MENTION_OPEN, DocumentRenderer

_SYSTEM_PROMPT = (
    "You are an expert in cross-document coreference resolution (CDCR). "
    "You are given every document of a single subtopic. Your task is to group the mentions that "
    "refer to the same real-world entity or event into coreference chains.\n\n"
    "Each document is shown once. Inside it, every entity or event mention is wrapped in markers "
    f"{MENTION_OPEN.format(mention_id='ID')} ... {MENTION_CLOSE.format(mention_id='ID')}, where ID "
    f"is the mention's unique id. A {GAP_MARKER} marker stands for a stretch of the document "
    "omitted from the input.\n\n"
    "Return the coreference chains as a list of lists of mention ids: each inner list is one chain "
    "of mentions that corefer. Include only ids that appear in the input, and place each id in at "
    "most one chain. Omit mentions that do not corefer with any other mention — these are treated "
    "as singletons automatically, so do not return single-element chains."
)


class GroupedE2EPredictor(BaseE2EPredictor):

    """End-to-end CDCR predictor that renders each shared document once.

    Reconstructs the subtopic's documents from the union of the mentions' overlapping context
    windows and tags mentions inline by id (see DocumentRenderer), so shared context is sent a
    single time instead of repeated per mention. This keeps prompts far smaller on subtopics where
    many mentions live in the same documents, at the cost of the more complex rendering.
    """

    system_prompt = _SYSTEM_PROMPT

    def __init__(
        self,
        connector: LLMConnector,
        estimator: PromptTokenEstimator,
        rng: random.Random | None = None,
        renderer: DocumentRenderer | None = None,
    ) -> None:
        super().__init__(connector, estimator, rng)
        self._renderer = renderer if renderer is not None else DocumentRenderer()

    def _build_messages(self, mentions: MentionCollection) -> list[ChatCompletionMessageParam]:
        content = f"## Documents\n\n{self._renderer.render_subtopic(mentions)}"
        return [{"role": "user", "content": content}]
