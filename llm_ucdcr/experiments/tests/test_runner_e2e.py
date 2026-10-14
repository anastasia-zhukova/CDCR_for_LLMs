import asyncio
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from llm_ucdcr.approaches.base import SubtopicDecision, SubtopicPredictor
from llm_ucdcr.data.loader import DatasetName, MentionCollection, MentionKind, Split
from llm_ucdcr.data.mention import Mention
from llm_ucdcr.experiments.config import (
    ApproachConfig,
    CandidatesConfig,
    DataConfig,
    E2EConfig,
    ExperimentConfig,
    LLMConfig,
    RunConfig,
)
from llm_ucdcr.experiments.runner import RunAbortedError, SubtopicRecord, SubtopicRunner
from llm_ucdcr.experiments.store import SubtopicStore
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


def _mention(mention_id: str, chain: str, subtopic: str) -> Mention:
    return Mention.from_dict({**_BASE, "mention_id": mention_id, "coref_chain": chain, "subtopic_id": subtopic})


def _mentions() -> MentionCollection:
    # Subtopic s1: m1+m2 corefer (c1), m3 is a singleton (c3). Subtopic s2: m4+m5 corefer (c4).
    return MentionCollection(
        [
            _mention("m1", "c1", "s1"),
            _mention("m2", "c1", "s1"),
            _mention("m3", "c3", "s1"),
            _mention("m4", "c4", "s2"),
            _mention("m5", "c4", "s2"),
        ],
    )


class _FakeLoader:
    def __init__(self, mentions: MentionCollection) -> None:
        self._mentions = mentions

    def load(self, dataset: DatasetName, split: Split, kind: MentionKind) -> MentionCollection:  # noqa: ARG002
        return self._mentions


class _FakeTracker:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    @contextmanager
    def run(self, **_: Any) -> Iterator[None]:
        yield None

    def log_metrics(self, metrics: dict[str, float], *_: Any, **__: Any) -> None:
        self.metrics.update(metrics)

    def log_jsonl(self, *_: Any, **__: Any) -> None: ...
    def log_dict(self, *_: Any, **__: Any) -> None: ...


class _GoldSubtopicPredictor(SubtopicPredictor):

    """Returns the gold multi-mention chains for whichever subtopic it is given."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls = 0
        self._exc = exc

    async def predict(self, mentions: MentionCollection) -> SubtopicDecision:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        by_chain: dict[str, list[str]] = defaultdict(list)
        for mention in mentions:
            by_chain[mention.coref_chain].append(mention.mention_id)
        return SubtopicDecision(clusters=[ids for ids in by_chain.values() if len(ids) > 1])


def _config(tmp_path: Path, *, resume: bool) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            dataset=DatasetName.ECBplus,
            kind=MentionKind.ENTITY,
            eval_split=Split.TEST,
            exclude_singletons=False,
        ),
        candidates=CandidatesConfig(),
        llm=LLMConfig(model="fake", max_concurrency=2, max_retries=0, timeout=10.0),
        approach=ApproachConfig(name="e2e", few_shot=None, e2e=E2EConfig()),
        run=RunConfig(
            checkpoint_dir=str(tmp_path),
            resume=resume,
            persist_predictions=False,
            progress_log_every=1000,
            max_consecutive_failures=5,
        ),
    )


def _runner(config: ExperimentConfig, predictor: SubtopicPredictor, tracker: _FakeTracker) -> SubtopicRunner:
    return SubtopicRunner(
        config=config,
        loader=_FakeLoader(_mentions()),
        tracker=tracker,
        predictor=predictor,
    )


def test_assemble_clusters_fills_singletons_for_uncovered_mentions() -> None:
    records = [
        SubtopicRecord("s1", n_mentions=3, n_predicted_clusters=1, clusters=[["m1", "m2"]], error=None, latency_seconds=0.0),
        SubtopicRecord("s2", n_mentions=2, n_predicted_clusters=1, clusters=[["m4", "m5"]], error=None, latency_seconds=0.0),
    ]
    clusters = SubtopicRunner._assemble_clusters(records, _mentions())
    # m3 was never placed in a chain, so it shows up as its own singleton.
    assert ["m3"] in clusters
    assert sorted(len(c) for c in clusters) == [1, 2, 2]


def test_failed_subtopic_contributes_singletons_only() -> None:
    records = [
        SubtopicRecord("s1", n_mentions=3, n_predicted_clusters=0, clusters=[], error="boom", latency_seconds=0.0),
        SubtopicRecord("s2", n_mentions=2, n_predicted_clusters=1, clusters=[["m4", "m5"]], error=None, latency_seconds=0.0),
    ]
    clusters = SubtopicRunner._assemble_clusters(records, _mentions())
    # The failed subtopic's mentions all collapse to singletons; s2's chain survives.
    assert ["m1"] in clusters
    assert ["m2"] in clusters
    assert ["m4", "m5"] in clusters


def test_gold_predictor_yields_perfect_metrics(tmp_path: Path) -> None:
    tracker = _FakeTracker()
    runner = _runner(_config(tmp_path, resume=False), _GoldSubtopicPredictor(), tracker)

    asyncio.run(runner.run())

    for metric in ("muc_f1", "b_cubed_f1", "ceafe_f1", "conll_f1"):
        assert tracker.metrics[metric] == pytest.approx(1.0), f"{metric}={tracker.metrics[metric]}"
    assert tracker.metrics["n_subtopics"] == 2.0
    assert tracker.metrics["n_failed_subtopics"] == 0.0


def test_successful_subtopics_are_checkpointed(tmp_path: Path) -> None:
    runner = _runner(_config(tmp_path, resume=False), _GoldSubtopicPredictor(), _FakeTracker())

    asyncio.run(runner.run())

    records = SubtopicStore(runner._checkpoint_path()).records()
    assert {r["subtopic_id"] for r in records} == {"s1", "s2"}


def test_resume_skips_already_done_subtopics(tmp_path: Path) -> None:
    config = _config(tmp_path, resume=True)
    predictor = _GoldSubtopicPredictor()
    runner = _runner(config, predictor, _FakeTracker())

    store = SubtopicStore(runner._checkpoint_path())
    store.append(
        {
            "subtopic_id": "s1",
            "n_mentions": 3,
            "n_predicted_clusters": 1,
            "clusters": [["m1", "m2"]],
            "error": None,
            "latency_seconds": 0.0,
            "error_category": None,
            "http_status": None,
        },
    )
    store.close()

    asyncio.run(runner.run())

    assert predictor.calls == 1  # only s2 is pending; s1 was resumed from the checkpoint


def test_fatal_error_aborts_run(tmp_path: Path) -> None:
    config = _config(tmp_path, resume=False)
    predictor = _GoldSubtopicPredictor(exc=LLMError("auth", 401, "bad key"))
    runner = _runner(config, predictor, _FakeTracker())

    with pytest.raises(RunAbortedError):
        asyncio.run(runner.run())
