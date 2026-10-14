"""Run a single CDCR experiment configured via Hydra and log to MLflow.

Examples:
    uv run python scripts/run_experiment.py
    uv run python scripts/run_experiment.py data=CD2CR run=smoke
    uv run python scripts/run_experiment.py data=single/ECBplus_entity
    uv run python scripts/run_experiment.py approach.few_shot.num_examples=10 \
        mlflow.run_name=ecbplus_k10

    # LLM-free sanity baselines (no API key needed):
    uv run python scripts/run_experiment.py approach=baseline_constant_true
    uv run python scripts/run_experiment.py approach=baseline_oracle
    uv run python scripts/run_experiment.py approach=baseline_head_lemma candidates=full

"""
import asyncio
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from llm_ucdcr.approaches.base import PairPredictor, SubtopicPredictor
from llm_ucdcr.approaches.baseline import BaselinePredictor, BaselineStrategy
from llm_ucdcr.approaches.e2e import E2EPredictor, PromptTokenEstimator
from llm_ucdcr.approaches.e2e_grouped import GroupedE2EPredictor
from llm_ucdcr.approaches.random_few_shot import RandomFewShotPredictor
from llm_ucdcr.constants import (
    CONFIG_DIR,
    DATA_ROOT,
    LLM_API_URL,
    LLM_MODEL_NAME,
    LOG_CONFIG,
    MLFLOW_TRACKING_URI,
)
from llm_ucdcr.data.loader import DatasetLoader
from llm_ucdcr.experiments import ExperimentConfig, RunAbortedError, build_runner, register_configs
from llm_ucdcr.llm.connector import LLMConnector
from llm_ucdcr.logs.accessors import get_logger
from llm_ucdcr.tracking import MLflowTracker


register_configs()

logger = get_logger(__name__)


def _build_predictor(config: ExperimentConfig) -> PairPredictor | SubtopicPredictor:
    """Pick the predictor for the configured approach.

    A set approach.strategy marks an LLM-free baseline: it needs neither an API key nor
    an endpoint, and its config carries no llm block. The e2e approaches score whole subtopics
    via a SubtopicPredictor; otherwise we wrap the LLM connector in the pairwise predictor.
    """
    if config.approach.strategy is not None:
        return BaselinePredictor(BaselineStrategy(config.approach.strategy))
    if config.llm is None:
        raise ValueError("LLM approach requires an llm config (none was composed)")
    connector = LLMConnector(
        model=config.llm.model,
        base_url=config.llm.api_url,
        api_key=os.environ[config.llm.api_key_env],
        params=dict(config.llm.params),
        max_retries=config.llm.max_retries,
        timeout=config.llm.timeout,
        output_mode=config.llm.output_mode,
    )
    if config.approach.e2e_grouped is not None:
        return GroupedE2EPredictor(connector, PromptTokenEstimator(config.approach.e2e_grouped.tokenizer_encoding))
    if config.approach.e2e is not None:
        return E2EPredictor(connector, PromptTokenEstimator(config.approach.e2e.tokenizer_encoding))
    return RandomFewShotPredictor(connector)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(config: DictConfig) -> None:
    LOG_CONFIG.apply()
    if config.llm is not None:  # absent for LLM-free baselines
        if config.llm.model is None:
            config.llm.model = LLM_MODEL_NAME
        if config.llm.api_url is None:
            config.llm.api_url = LLM_API_URL
    logger.info("Resolved experiment config", config=OmegaConf.to_yaml(config))
    typed_config: ExperimentConfig = OmegaConf.to_object(config)  # type: ignore[assignment]
    loader = DatasetLoader(DATA_ROOT)
    tracker = MLflowTracker(tracking_uri=MLFLOW_TRACKING_URI)
    runner = build_runner(
        config=typed_config,
        loader=loader,
        tracker=tracker,
        predictor=_build_predictor(typed_config),
    )
    try:
        asyncio.run(runner.run())
    except RunAbortedError as exc:
        # Expected control flow (the breaker stopped the run); a traceback would only add noise.
        logger.error("Run aborted before completion", reason=str(exc))  # noqa: TRY400
        logger.error(  # noqa: TRY400
            "Fix the cause (e.g. an invalid API key or exhausted quota), then re-run to warm-start from the checkpoint",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
