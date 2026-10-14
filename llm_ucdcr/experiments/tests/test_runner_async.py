import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from llm_ucdcr.approaches.baseline import BaselinePredictor, BaselineStrategy
from llm_ucdcr.approaches.random_few_shot import FewShotStrategy, RandomFewShotPredictor
from llm_ucdcr.approaches.random_few_shot.models import CoreferenceDecision
from llm_ucdcr.candidates import CandidateStrategy
from llm_ucdcr.data.loader import DatasetName, MentionCollection, MentionKind, Split
from llm_ucdcr.data.mention import Mention
from llm_ucdcr.experiments.config import (
    ApproachConfig,
    CandidatesConfig,
    DataConfig,
    ExperimentConfig,
    FewShotConfig,
    LLMConfig,
    RunConfig,
)
from llm_ucdcr.experiments.runner import PairwiseRunner, RunAbortedError
from llm_ucdcr.experiments.store import PredictionStore
from llm_ucdcr.llm.errors import LLMError

_BASE: dict[str, Any] = {
    "mention_id": "m1",
    "coref_chain": "chain1",
    "tokens_str": "Canada",
    "description": "desc",
    "coref_type": "IDENTITY",
    "mention_type": "LOC",
    "mention_full_type": "LOC_GEO",
    "tokens_text": ["Canada"],
    "tokens_number": [1],
    "mention_head": "Canada",
    "mention_head_id": 1,
    "mention_head_pos": "PROPN",
    "mention_head_lemma": "canada",
    "mention_ner": "GPE",
    "sent_id": 0,
    "topic_id": "t1",
    "topic": "topic_one",
    "subtopic_id": "s1",
    "subtopic": "subtopic_one",
    "doc_id": "d1",
    "doc": "doc_one",
    "mention_context": ["Some", "Canada", "context"],
    "context_start_end_global_id": [0, 3],
    "mention_sentence_start_end": [0, 3],
    "tokens_number_context": [1],
    "mention_head_id_context": 1,
    "is_singleton": False,
    "conll_doc_key": "t1/s1/d1",
}


def _mentions() -> MentionCollection:
    # Three mentions, same subtopic + head lemma -> 3 candidate pairs (m1,m2), (m1,m3), (m2,m3).
    return MentionCollection(
        [
            Mention.from_dict({**_BASE, "mention_id": "m1", "coref_chain": "c1"}),
            Mention.from_dict({**_BASE, "mention_id": "m2", "coref_chain": "c2"}),
            Mention.from_dict({**_BASE, "mention_id": "m3", "coref_chain": "c3"}),
        ],
    )


class _FakeLoader:
    def __init__(self, mentions: MentionCollection) -> None:
        self._mentions = mentions

    def load(self, dataset: DatasetName, split: Split, kind: MentionKind) -> MentionCollection:  # noqa: ARG002
        return self._mentions


class _FakeTracker:
    @contextmanager
    def run(self, **_: Any) -> Iterator[None]:
        yield None

    def log_metrics(self, *_: Any, **__: Any) -> None: ...
    def log_jsonl(self, *_: Any, **__: Any) -> None: ...
    def log_dict(self, *_: Any, **__: Any) -> None: ...


class _FakeConnector:
    def __init__(self, exc: Exception | None = None, *, coreferent: bool = False) -> None:
        self.calls = 0
        self._exc = exc
        self._coreferent = coreferent

    async def predict(self, **_: Any) -> CoreferenceDecision:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return CoreferenceDecision(coreferent=self._coreferent, reasoning="stub")


def _config(tmp_path: Path, *, resume: bool, transitive_grouping: bool = True) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            dataset=DatasetName.ECBplus,
            kind=MentionKind.ENTITY,
            eval_split=Split.TEST,
            exclude_singletons=False,
        ),
        candidates=CandidatesConfig(),
        llm=LLMConfig(model="fake", max_concurrency=1, max_retries=0, timeout=10.0),
        approach=ApproachConfig(few_shot=FewShotConfig(num_examples=0)),
        run=RunConfig(
            checkpoint_dir=str(tmp_path),
            resume=resume,
            persist_predictions=False,
            progress_log_every=1000,
            max_consecutive_failures=5,
            transitive_grouping=transitive_grouping,
        ),
    )


def _runner(config: ExperimentConfig, connector: _FakeConnector) -> PairwiseRunner:
    return PairwiseRunner(
        config=config,
        loader=_FakeLoader(_mentions()),
        tracker=_FakeTracker(),
        predictor=RandomFewShotPredictor(connector),  # type: ignore[arg-type]
    )


def test_resume_skips_already_done_pairs(tmp_path: Path) -> None:
    config = _config(tmp_path, resume=True)
    connector = _FakeConnector()
    runner = _runner(config, connector)

    # Pre-seed the checkpoint with one of the three pairs.
    store = PredictionStore(runner._checkpoint_path())
    store.append(
        {
            "pair_index": 0,
            "mention_a_id": "m1",
            "mention_b_id": "m2",
            "gold_coreferent": False,
            "predicted_coreferent": False,
            "reasoning": "prior",
            "error": None,
            "latency_seconds": 0.0,
            "error_category": None,
            "http_status": None,
        },
    )

    asyncio.run(runner.run())

    assert connector.calls == 2  # only the two not-yet-done pairs
    assert len(PredictionStore(runner._checkpoint_path()).records()) == 3


def test_fatal_error_aborts_run(tmp_path: Path) -> None:
    config = _config(tmp_path, resume=False)
    connector = _FakeConnector(exc=LLMError("auth", 401, "bad key"))
    runner = _runner(config, connector)

    with pytest.raises(RunAbortedError):
        asyncio.run(runner.run())

    assert connector.calls == 1  # first fatal failure trips the breaker; the rest short-circuit


def test_transitive_grouping_skips_the_closing_pair(tmp_path: Path) -> None:
    # Pairs are generated in combinations order: (m1,m2), (m1,m3), (m2,m3). With every pair
    # predicted coreferent, m1~m2 and m1~m3 link all three, so (m2,m3) is inferred, not requested.
    config = _config(tmp_path, resume=False, transitive_grouping=True)
    connector = _FakeConnector(coreferent=True)
    runner = PairwiseRunner(
        config=config,
        loader=_FakeLoader(_mentions()),
        tracker=_FakeTracker(),
        predictor=RandomFewShotPredictor(connector),  # type: ignore[arg-type]
    )

    asyncio.run(runner.run())

    assert connector.calls == 2  # the third pair is inferred via transitive closure


def test_transitive_grouping_disabled_requests_every_pair(tmp_path: Path) -> None:
    config = _config(tmp_path, resume=False, transitive_grouping=False)
    connector = _FakeConnector(coreferent=True)
    runner = PairwiseRunner(
        config=config,
        loader=_FakeLoader(_mentions()),
        tracker=_FakeTracker(),
        predictor=RandomFewShotPredictor(connector),  # type: ignore[arg-type]
    )

    asyncio.run(runner.run())

    assert connector.calls == 3  # no inference: every pair hits the LLM


def test_transitive_grouping_does_not_propagate_negatives(tmp_path: Path) -> None:
    # All pairs predicted non-coreferent: nothing links, so no pair can be inferred away.
    config = _config(tmp_path, resume=False, transitive_grouping=True)
    connector = _FakeConnector(coreferent=False)
    runner = PairwiseRunner(
        config=config,
        loader=_FakeLoader(_mentions()),
        tracker=_FakeTracker(),
        predictor=RandomFewShotPredictor(connector),  # type: ignore[arg-type]
    )

    asyncio.run(runner.run())

    assert connector.calls == 3


def _baseline_config(tmp_path: Path, strategy: BaselineStrategy) -> ExperimentConfig:
    config = _config(tmp_path, resume=False)
    config.approach = ApproachConfig(name="baseline", strategy=strategy, few_shot=FewShotConfig(num_examples=0))
    return config


def test_baseline_constant_true_predicts_every_pair_coreferent(tmp_path: Path) -> None:
    config = _baseline_config(tmp_path, BaselineStrategy.CONSTANT_TRUE)
    runner = PairwiseRunner(
        config=config,
        loader=_FakeLoader(_mentions()),
        tracker=_FakeTracker(),
        predictor=BaselinePredictor(BaselineStrategy.CONSTANT_TRUE),
    )

    asyncio.run(runner.run())

    records = PredictionStore(runner._checkpoint_path()).records()
    assert len(records) == 3  # the three same-head-lemma, same-subtopic pairs
    assert all(r["predicted_coreferent"] is True for r in records)


def test_baseline_oracle_predicts_gold_answer_per_pair(tmp_path: Path) -> None:
    # m1 and m2 share a gold chain; m3 is on its own. Same head lemma + subtopic -> 3 candidate pairs.
    mentions = MentionCollection(
        [
            Mention.from_dict({**_BASE, "mention_id": "m1", "coref_chain": "c1"}),
            Mention.from_dict({**_BASE, "mention_id": "m2", "coref_chain": "c1"}),
            Mention.from_dict({**_BASE, "mention_id": "m3", "coref_chain": "c2"}),
        ],
    )
    config = _baseline_config(tmp_path, BaselineStrategy.ORACLE)
    runner = PairwiseRunner(
        config=config,
        loader=_FakeLoader(mentions),
        tracker=_FakeTracker(),
        predictor=BaselinePredictor(BaselineStrategy.ORACLE),
    )

    asyncio.run(runner.run())

    records = PredictionStore(runner._checkpoint_path()).records()
    by_pair = {frozenset((r["mention_a_id"], r["mention_b_id"])): r for r in records}
    assert by_pair[frozenset(("m1", "m2"))]["predicted_coreferent"] is True
    assert by_pair[frozenset(("m1", "m3"))]["predicted_coreferent"] is False
    assert by_pair[frozenset(("m2", "m3"))]["predicted_coreferent"] is False
    # Oracle never errs: every prediction matches its gold label.
    assert all(r["predicted_coreferent"] == r["gold_coreferent"] for r in records)


def _coref_collection(prefix: str, n: int) -> MentionCollection:
    # n mentions sharing one chain in one subtopic -> C(n, 2) coreferent all-pairs candidates.
    return MentionCollection(
        [Mention.from_dict({**_BASE, "mention_id": f"{prefix}{i}", "coref_chain": f"{prefix}_c"}) for i in range(n)],
    )


class _PerDatasetLoader:
    def __init__(self, by_dataset_kind: dict[tuple[DatasetName, MentionKind], MentionCollection]) -> None:
        self._by_dataset_kind = by_dataset_kind

    def load(self, dataset: DatasetName, split: Split, kind: MentionKind) -> MentionCollection:  # noqa: ARG002
        # A (dataset, kind) we didn't populate stands in for a missing train file for that kind.
        if (dataset, kind) not in self._by_dataset_kind:
            raise FileNotFoundError(f"no train split for {dataset.value}/{kind.value}")
        return self._by_dataset_kind[(dataset, kind)]


def _few_shot_config(strategy: FewShotStrategy, num_examples: int) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            dataset=DatasetName.ECBplus,
            kind=MentionKind.ENTITY,
            eval_split=Split.TEST,
            exclude_singletons=False,
        ),
        candidates=CandidatesConfig(strategy=CandidateStrategy.ALL_PAIRS),
        llm=LLMConfig(model="fake", max_concurrency=1, max_retries=0, timeout=10.0),
        approach=ApproachConfig(few_shot=FewShotConfig(num_examples=num_examples, strategy=strategy)),
        run=RunConfig(
            checkpoint_dir="unused",
            resume=False,
            persist_predictions=False,
            progress_log_every=1000,
            max_consecutive_failures=5,
        ),
    )


def _few_shot_runner(config: ExperimentConfig, loader: object) -> PairwiseRunner:
    return PairwiseRunner(
        config=config,
        loader=loader,  # type: ignore[arg-type]
        tracker=_FakeTracker(),
        predictor=BaselinePredictor(BaselineStrategy.ORACLE),  # unused by pool building
    )


def test_few_shot_same_dataset_builds_pool() -> None:
    # Regression: num_examples > 0 must build a pool (the candidate generator needs a strategy).
    runner = _few_shot_runner(
        _few_shot_config(FewShotStrategy.SAME_DATASET, num_examples=2),
        _FakeLoader(_mentions()),  # 3 mentions -> 3 all-pairs candidates
    )
    pool = runner._build_few_shot_pool()
    assert len(pool) == 2


def test_few_shot_all_datasets_round_robins_across_datasets() -> None:
    by_dataset_kind = {
        (DatasetName.CD2CR, MentionKind.ENTITY): _coref_collection("cd", 3),  # CD2CR sorts before CEREC
        (DatasetName.CEREC, MentionKind.ENTITY): _coref_collection("ce", 3),
    }
    runner = _few_shot_runner(
        _few_shot_config(FewShotStrategy.ALL_DATASETS, num_examples=2),
        _PerDatasetLoader(by_dataset_kind),
    )
    positives, negatives = runner._few_shot_pairs_all_datasets()
    # Round-robin: alternate datasets rather than draining the alphabetically-first one.
    assert [p.mention_a.mention_id[:2] for p in positives[:2]] == ["cd", "ce"]
    assert negatives == []  # every pair shares a chain -> all coreferent


def test_few_shot_all_datasets_uses_both_kinds_when_available() -> None:
    # CD2CR has only entity train data; CEREC has both entity and event.
    by_dataset_kind = {
        (DatasetName.CD2CR, MentionKind.ENTITY): _coref_collection("cdent", 2),
        (DatasetName.CEREC, MentionKind.ENTITY): _coref_collection("ceent", 2),
        (DatasetName.CEREC, MentionKind.EVENT): _coref_collection("ceevt", 2),
    }
    runner = _few_shot_runner(
        _few_shot_config(FewShotStrategy.ALL_DATASETS, num_examples=100),
        _PerDatasetLoader(by_dataset_kind),
    )
    positives, _ = runner._few_shot_pairs_all_datasets()
    prefixes = {p.mention_a.mention_id[:5] for p in positives}
    # CEREC contributes both kinds; CD2CR contributes its only (entity) kind.
    assert prefixes == {"cdent", "ceent", "ceevt"}


def test_few_shot_all_datasets_caps_at_total_available() -> None:
    by_dataset_kind = {
        (DatasetName.CD2CR, MentionKind.ENTITY): _coref_collection("cd", 3),  # 3 pairs
        (DatasetName.CEREC, MentionKind.ENTITY): _coref_collection("ce", 3),  # 3 pairs
    }
    runner = _few_shot_runner(
        _few_shot_config(FewShotStrategy.ALL_DATASETS, num_examples=100),
        _PerDatasetLoader(by_dataset_kind),
    )
    pool = runner._build_few_shot_pool()
    assert len(pool) == 6  # everything available across both datasets, nothing fabricated


def test_few_shot_pool_is_balanced_positive_and_negative() -> None:
    # Two chains of 2 mentions each -> C(4,2)=6 pairs: 2 coreferent (within-chain), 4 not.
    mentions = MentionCollection(
        [
            Mention.from_dict({**_BASE, "mention_id": "a0", "coref_chain": "A"}),
            Mention.from_dict({**_BASE, "mention_id": "a1", "coref_chain": "A"}),
            Mention.from_dict({**_BASE, "mention_id": "b0", "coref_chain": "B"}),
            Mention.from_dict({**_BASE, "mention_id": "b1", "coref_chain": "B"}),
        ],
    )
    runner = _few_shot_runner(
        _few_shot_config(FewShotStrategy.SAME_DATASET, num_examples=4),
        _FakeLoader(mentions),
    )
    pool = runner._build_few_shot_pool()
    assert len(pool) == 4
    assert sum(1 for v in pool if v.coreferent) == 2  # enforced half/half
    assert sum(1 for v in pool if not v.coreferent) == 2


def test_few_shot_pool_tops_up_with_negatives_when_positives_scarce() -> None:
    # _mentions(): three distinct chains -> 3 all-pairs candidates, all non-coreferent.
    runner = _few_shot_runner(
        _few_shot_config(FewShotStrategy.SAME_DATASET, num_examples=2),
        _FakeLoader(_mentions()),
    )
    pool = runner._build_few_shot_pool()
    assert len(pool) == 2  # no positives available, so the pool is filled with negatives
    assert all(not v.coreferent for v in pool)
