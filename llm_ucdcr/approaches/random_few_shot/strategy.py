from enum import StrEnum


class FewShotStrategy(StrEnum):
    # Draw few-shot example pairs from the train split of the dataset under evaluation
    # (or `train_dataset` when overridden).
    SAME_DATASET = "same_dataset"
    # Draw few-shot example pairs from the train splits of every dataset, round-robin,
    # so the pool spans domains rather than a single one. Still train-only.
    ALL_DATASETS = "all_datasets"
