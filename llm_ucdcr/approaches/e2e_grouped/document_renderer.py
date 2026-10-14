from collections import defaultdict

from llm_ucdcr.data.loader import MentionCollection
from llm_ucdcr.data.mention import Mention

# Inline markers wrapping a mention's tokens. Both ends carry the id so overlapping or adjacent
# mentions stay unambiguous regardless of nesting order.
MENTION_OPEN = "[[m={mention_id}]]"
MENTION_CLOSE = "[[/m={mention_id}]]"
# Stands in for a stretch of the document not covered by any mention's context window.
GAP_MARKER = "[...]"


class DocumentRenderer:

    """Renders a subtopic's documents once each, with mentions tagged inline by id.

    Mentions in a subtopic overwhelmingly share documents, and each mention carries an overlapping
    window of the same document text (``mention_context`` + the absolute span in
    ``context_start_end_global_id``). Listing every mention with its own window therefore repeats
    the document many times. This renderer instead reconstructs each document once from the union of
    its windows — overlapping windows agree token-for-token — and wraps each mention's tokens in
    [[m=ID]] ... [[/m=ID]] so the LLM can cluster by id while reading the shared context a single
    time. Stretches no window covers are collapsed to a gap marker.
    """

    def render_subtopic(self, mentions: MentionCollection) -> str:
        by_doc: dict[str, list[Mention]] = defaultdict(list)
        for mention in mentions:
            by_doc[mention.doc_id].append(mention)
        parts: list[str] = []
        for doc_id, doc_mentions in sorted(by_doc.items()):
            parts.append(f"### Document {doc_id}")
            parts.append(self.render_document(doc_mentions))
        return "\n\n".join(parts)

    def render_document(self, mentions: list[Mention]) -> str:
        tokens: dict[int, str] = {}
        starts: dict[int, list[str]] = defaultdict(list)
        ends: dict[int, list[str]] = defaultdict(list)
        for mention in mentions:
            self._index_mention(mention, tokens, starts, ends)
        return self._assemble(tokens, starts, ends)

    @staticmethod
    def _index_mention(
        mention: Mention,
        tokens: dict[int, str],
        starts: dict[int, list[str]],
        ends: dict[int, list[str]],
    ) -> None:
        start = mention.context_start_end_global_id[0]
        for local_index, token in enumerate(mention.mention_context):
            tokens[start + local_index] = token
        # tokens_number_context are local indices into the window; shift by start to get the
        # mention's absolute positions in the document.
        absolute = [start + local_index for local_index in mention.tokens_number_context]
        if len(absolute) == 0:
            return
        starts[min(absolute)].append(mention.mention_id)
        ends[max(absolute)].append(mention.mention_id)

    @staticmethod
    def _assemble(
        tokens: dict[int, str],
        starts: dict[int, list[str]],
        ends: dict[int, list[str]],
    ) -> str:
        out: list[str] = []
        previous: int | None = None
        for position in sorted(tokens):
            if previous is not None and position != previous + 1:
                out.append(GAP_MARKER)
            out.extend(MENTION_OPEN.format(mention_id=mention_id) for mention_id in starts.get(position, []))
            out.append(tokens[position])
            out.extend(MENTION_CLOSE.format(mention_id=mention_id) for mention_id in ends.get(position, []))
            previous = position
        return " ".join(out)
