from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from llm_ucdcr.approaches.baseline import BaselineStrategy
from llm_ucdcr.approaches.random_few_shot import FewShotStrategy
from llm_ucdcr.candidates import CandidateStrategy
from llm_ucdcr.data.loader import DatasetName, MentionKind, Split


@dataclass
class DataConfig:
    dataset: DatasetName = MISSING
    eval_split: Split = Split.TEST
    kind: MentionKind = MentionKind.ENTITY  # ENTITY / EVENT to score one type; BOTH to mix them into one pool
    exclude_singletons: bool = True


@dataclass
class CandidatesConfig:
    strategy: CandidateStrategy = MISSING
    max_candidates: int | None = None
    max_candidates_per_mention: int | None = None
    seed: int | None = 42


@dataclass
class LLMConfig:
    model: str | None = None  # exact model name; None -> constants.LLM_MODEL_NAME
    api_url: str | None = None  # endpoint base URL; None -> constants.LLM_API_URL
    api_key_env: str = "LLM_API_TOKEN"  # env var holding this endpoint's API key
    output_mode: str = "json_schema"  # "json_schema" (native structured outputs) or "json_object" (prompted JSON)
    params: dict[str, Any] = field(default_factory=dict)  # forwarded to the chat-completions call
    max_concurrency: int = MISSING  # max in-flight async requests; under the endpoint's concurrency limit
    max_retries: int = MISSING  # transient-failure retries (SDK exponential backoff)
    timeout: float = MISSING  # per-request timeout in seconds


@dataclass
class FewShotConfig:
    num_examples: int = 5
    # SAME_DATASET: pool from data.dataset (or train_dataset). ALL_DATASETS: pool across every
    # dataset's train split, round-robin, ignoring train_dataset. Both stay train-only.
    strategy: FewShotStrategy = FewShotStrategy.SAME_DATASET
    train_dataset: DatasetName | None = None  # defaults to data.dataset when None; ignored for ALL_DATASETS
    train_kind: MentionKind | None = None  # defaults to data.kind when None
    # Subsample at most this many train mentions per (dataset, kind) *before* pairing, bounding the
    # work at O(pool_mentions^2) instead of the full split's n^2. None = use every train mention.
    pool_mentions: int | None = 512
    seed: int = 42  # RNG seed for the mention subsample; fixed so the pool is constant across the run


@dataclass
class E2EConfig:
    # tiktoken encoding used to estimate (and log) prompt token counts before sending; a proxy for
    # DeepSeek's tokenizer. Informational only — oversized subtopics are sent anyway.
    tokenizer_encoding: str = "o200k_base"


@dataclass
class ApproachConfig:
    name: str = "random_few_shot"
    # At most one of these is set per approach; the others stay None and are absent from the config:
    strategy: BaselineStrategy | None = None  # baselines: the labeling rule (constant_true / oracle / head_lemma)
    few_shot: FewShotConfig | None = None  # random_few_shot: the in-context example pool
    e2e: E2EConfig | None = None  # e2e: end-to-end subtopic clustering, one context window per mention
    e2e_grouped: E2EConfig | None = None  # e2e_grouped: end-to-end, each shared document rendered once


@dataclass
class MLflowExperimentConfig:
    experiment_name: str = "${approach.name}"
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class RunConfig:
    max_predictions: int | None = None  # cap LLM calls; useful for smoke runs
    progress_log_every: int = 50
    persist_predictions: bool = True
    transitive_grouping: bool = MISSING  # skip a pair's LLM request when its mentions are already co-clustered
    resume: bool = MISSING  # skip pairs already recorded in the checkpoint store
    checkpoint_dir: str = MISSING  # directory holding warm-start checkpoints
    max_consecutive_failures: int = MISSING  # circuit breaker; abort after this many in a row; <=0 disables


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    candidates: CandidatesConfig = field(default_factory=CandidatesConfig)
    # None for LLM-free approaches (baselines). Composed via the approach's defaults list
    # (configs/approach/random_few_shot.yaml pulls in `/llm: deepseek`), so a baseline config
    # carries no model/endpoint fields at all rather than logging stale ones to MLflow.
    llm: LLMConfig | None = None
    approach: ApproachConfig = field(default_factory=ApproachConfig)
    mlflow: MLflowExperimentConfig = field(default_factory=MLflowExperimentConfig)
    run: RunConfig = field(default_factory=RunConfig)
    extra_params: dict[str, Any] = field(default_factory=dict)

    def flatten(self) -> dict[str, Any]:
        """Flatten the config into dotted-key params for experiment tracking."""
        config_dict = OmegaConf.to_container(OmegaConf.structured(self), resolve=True)
        assert isinstance(config_dict, dict)
        return self._flatten(config_dict)

    @staticmethod
    def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for key, value in payload.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(ExperimentConfig._flatten(value, full_key))
            else:
                flat[full_key] = value
        return flat


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_experiment_config", node=ExperimentConfig)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="candidates", name="base_candidates", node=CandidatesConfig)
    cs.store(group="llm", name="base_llm", node=LLMConfig)
    cs.store(group="approach", name="base_approach", node=ApproachConfig)
    cs.store(group="run", name="base_run", node=RunConfig)
