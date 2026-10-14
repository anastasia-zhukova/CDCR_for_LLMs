from enum import StrEnum


class BaselineStrategy(StrEnum):

    """How a baseline labels each candidate pair, replacing the LLM decision.

    CONSTANT_TRUE — every candidate is called coreferent, so the predicted clustering
    is exactly the connected components of the candidate graph. The recall ceiling of
    the candidate set, with no precision.

    ORACLE — each pair is labeled with its gold answer. Link precision is perfect, so
    any gap from CoNLL=1.0 is purely the candidate generator's recall ceiling (gold
    links it never proposed). The upper bound the LLM could reach on this candidate set.

    HEAD_LEMMA — a pair is coreferent iff its mentions share a normalized head lemma.
    The classic lexical CDCR baseline as a labeling rule: run on the ALL_PAIRS candidate
    set, its connected components are the per-(subtopic, head lemma) groups.
    """

    CONSTANT_TRUE = "constant_true"
    ORACLE = "oracle"
    HEAD_LEMMA = "head_lemma"
