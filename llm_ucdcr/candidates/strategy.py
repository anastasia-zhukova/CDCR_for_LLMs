from enum import StrEnum


class CandidateStrategy(StrEnum):
    # Pair every two mentions within a subtopic (upper triangle).
    ALL_PAIRS = "all_pairs"
