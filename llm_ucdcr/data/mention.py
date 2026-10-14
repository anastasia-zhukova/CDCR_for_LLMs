from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True)
class Mention:
    mention_id: str
    coref_chain: str
    tokens_str: str
    description: str
    coref_type: str
    mention_type: str
    mention_full_type: str
    tokens_text: tuple[str, ...]
    tokens_number: tuple[int, ...]
    mention_head: str
    mention_head_id: int
    mention_head_pos: str
    mention_head_lemma: str
    mention_ner: str
    sent_id: int
    topic_id: str
    topic: str
    subtopic_id: str
    subtopic: str
    doc_id: str
    doc: str
    mention_context: tuple[str, ...]
    context_start_end_global_id: tuple[int, int]
    mention_sentence_start_end: tuple[int, int]
    tokens_number_context: tuple[int, ...]
    mention_head_id_context: int
    is_singleton: bool
    conll_doc_key: str
    language: str = "en"

    @property
    def mention_head_lemma_normalized(self) -> str:
        return self.mention_head_lemma.casefold()

    @property
    def gold_cluster_key(self) -> tuple[str, str]:
        """Identity of the gold cluster this mention belongs to.

        Coreference is resolved one subtopic at a time, so a gold cluster is keyed by
        (coref_chain, subtopic_id), not coref_chain alone: a chain that spans subtopics
        forms one gold cluster per subtopic, matching what a subtopic-level system can ever
        produce. This is the single source of truth for gold membership — group and compare
        mentions through it rather than touching coref_chain directly.
        """
        return (self.coref_chain, self.subtopic_id)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            mention_id=d["mention_id"],
            coref_chain=d["coref_chain"],
            tokens_str=d["tokens_str"],
            description=d["description"],
            coref_type=d["coref_type"],
            mention_type=d["mention_type"],
            mention_full_type=d["mention_full_type"],
            tokens_text=tuple(d["tokens_text"]),
            tokens_number=tuple(d["tokens_number"]),
            mention_head=d["mention_head"],
            mention_head_id=d["mention_head_id"],
            mention_head_pos=d["mention_head_pos"],
            mention_head_lemma=d["mention_head_lemma"],
            mention_ner=d["mention_ner"],
            sent_id=d["sent_id"],
            topic_id=str(d["topic_id"]),
            topic=d["topic"],
            subtopic_id=str(d["subtopic_id"]),
            subtopic=d["subtopic"],
            doc_id=d["doc_id"],
            doc=d["doc"],
            mention_context=tuple(d["mention_context"]),
            context_start_end_global_id=tuple(d["context_start_end_global_id"]),
            mention_sentence_start_end=tuple(d["mention_sentence_start_end"]),
            tokens_number_context=tuple(d["tokens_number_context"]),
            mention_head_id_context=d["mention_head_id_context"],
            is_singleton=d["is_singleton"],
            conll_doc_key=d["conll_doc_key"],
            language=d.get("language", "en"),
        )
