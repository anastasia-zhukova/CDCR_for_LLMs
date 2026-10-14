import asyncio
import hashlib
import json
import random
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TypeVar

from coval.conll.reader import get_mention_assignments
from coval.eval.evaluator import Evaluator, b_cubed, ceafe, muc
from tqdm import tqdm

from llm_ucdcr.approaches.base import PairPredictor, SubtopicPredictor
from llm_ucdcr.approaches.random_few_shot import FewShotStrategy, LabeledPairView
from llm_ucdcr.candidates.candidate_pair import CandidatePair
from llm_ucdcr.candidates.generator import CandidateGenerator
from llm_ucdcr.candidates.strategy import CandidateStrategy
from llm_ucdcr.data.loader import DatasetLoader, DatasetName, MentionCollection, MentionKind, Split
from llm_ucdcr.llm.errors import FATAL_CATEGORIES, LLMError
from llm_ucdcr.logs.accessors import get_logger
from llm_ucdcr.tracking import MLflowTracker

from .config import ExperimentConfig
from .store import PredictionStore, SubtopicStore
from .union_find import UnionFind

logger = get_logger(__name__)

_HIGH_FAILURE_RATE = 0.1

_CONLL_METRICS = ("muc", "b_cubed", "ceafe")


def score_clusters(gold_clusters: list[list[str]], predicted_clusters: list[list[str]]) -> dict[str, float]:
    """Score predicted clusters against gold with MUC, B-cubed, CEAF_e and the CoNLL F1.

    Pure evaluation: it takes the two clusterings as given and never builds them, so the pairwise
    path (clusters via union-find over positive pairs) and the end-to-end path (clusters returned by
    the LLM) share one metric definition while each owns how its clusters are assembled.
    """
    coref_info = (
        gold_clusters,
        predicted_clusters,
        get_mention_assignments(gold_clusters, predicted_clusters),
        get_mention_assignments(predicted_clusters, gold_clusters),
    )
    results: dict[str, float] = {
        "n_gold_clusters": float(len(gold_clusters)),
        "n_predicted_clusters": float(len(predicted_clusters)),
    }
    f1_scores: list[float] = []
    for name, metric_fn in (("muc", muc), ("b_cubed", b_cubed), ("ceafe", ceafe)):
        evaluator = Evaluator(metric_fn)
        evaluator.update(coref_info)
        precision, recall, f1 = evaluator.get_prf()
        results[f"{name}_precision"] = precision
        results[f"{name}_recall"] = recall
        results[f"{name}_f1"] = f1
        if name in _CONLL_METRICS:
            f1_scores.append(f1)
    results["conll_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return results


class RunAbortedError(Exception):

    """Raised when the circuit breaker stops a run early (e.g. exhausted balance, auth failure)."""


@dataclass(frozen=True)
class PredictionRecord:
    pair_index: int
    mention_a_id: str
    mention_b_id: str
    gold_coreferent: bool
    predicted_coreferent: bool | None
    reasoning: str | None
    error: str | None
    latency_seconds: float
    error_category: str | None = None
    http_status: int | None = None
    transitive: bool = False  # inferred coreferent via transitive closure, no LLM request spent


@dataclass(frozen=True)
class SubtopicRecord:
    subtopic_id: str
    n_mentions: int
    n_predicted_clusters: int
    clusters: list[list[str]]
    error: str | None
    latency_seconds: float
    error_category: str | None = None
    http_status: int | None = None


_Record = TypeVar("_Record", PredictionRecord, SubtopicRecord)
_Item = TypeVar("_Item")


class _ProgressTracker:

    """Drives a tqdm bar for completed predictions and logs progress at a fixed interval."""

    def __init__(self, total: int, log_every: int, unit: str = "pair", desc: str = "Scoring pairs") -> None:
        self._total = total
        self._log_every = log_every
        self._done = 0
        self._failures = 0
        self._bar = tqdm(
            total=total,
            unit=unit,
            desc=desc,
            disable=None,  # auto-disabled when stderr is not a TTY, keeping captured logs clean
        )

    def update(self, record: "PredictionRecord | SubtopicRecord") -> None:
        self._done += 1
        if record.error is not None:
            self._failures += 1
        self._bar.set_postfix(failures=self._failures, refresh=False)
        self._bar.update(1)
        if self._done % self._log_every == 0 or self._done == self._total:
            logger.info("Prediction progress", done=self._done, total=self._total, failures=self._failures)

    def close(self) -> None:
        self._bar.close()


class _BreakerState:

    """Circuit breaker: aborts the run on a fatal error or too many consecutive failures."""

    def __init__(self, max_consecutive_failures: int) -> None:
        self._max_consecutive = max_consecutive_failures
        self._consecutive = 0
        self.abort_reason: str | None = None

    @property
    def aborted(self) -> bool:
        return self.abort_reason is not None

    def register(self, record: "PredictionRecord | SubtopicRecord") -> None:
        if record.error is None:
            self._consecutive = 0
        else:
            self._consecutive += 1
            self._maybe_trip(record)

    def _maybe_trip(self, record: "PredictionRecord | SubtopicRecord") -> None:
        if self.aborted:
            return
        if record.error_category in FATAL_CATEGORIES:
            self._trip(f"fatal error ({record.error_category}): {record.error}")
        elif 0 < self._max_consecutive <= self._consecutive:
            self._trip(f"{self._consecutive} consecutive failures")

    def _trip(self, reason: str) -> None:
        self.abort_reason = reason
        logger.error("Circuit breaker tripped — aborting run", reason=reason)


class ExperimentRunner(ABC):

    """Runs a CDCR experiment end-to-end and logs results to MLflow.

    Base class for the two units of work: PairwiseRunner scores candidate pairs, SubtopicRunner
    scores whole subtopics. Subclasses implement ``_execute``; the base owns the MLflow run, the
    dataset split loading, the checkpoint path, and the shared async worker pool (``_drain``).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        loader: DatasetLoader,
        tracker: MLflowTracker,
        predictor: PairPredictor | SubtopicPredictor,
    ) -> None:
        self._config = config
        self._loader = loader
        self._tracker = tracker
        self._predictor = predictor

    async def run(self) -> None:
        with self._tracker.run(
            experiment_name=self._config.mlflow.experiment_name,
            run_name=self._config.mlflow.run_name,
            params=self._config.flatten(),
            tags=self._config.mlflow.tags,
        ):
            await self._execute()

    @abstractmethod
    async def _execute(self) -> None: ...

    def _load_split(self, split: Split) -> MentionCollection:
        mentions = self._loader.load(
            self._config.data.dataset,
            split,
            self._config.data.kind,
        )
        if self._config.data.exclude_singletons:
            mentions = mentions.without_singletons()
        return mentions

    def _checkpoint_path(self) -> Path:
        base = Path(self._config.run.checkpoint_dir)
        identity = {
            "dataset": str(self._config.data.dataset),
            "kind": str(self._config.data.kind),
            "eval_split": str(self._config.data.eval_split),
            "exclude_singletons": self._config.data.exclude_singletons,
            "candidates": asdict(self._config.candidates),
            "approach": asdict(self._config.approach),
            "transitive_grouping": self._config.run.transitive_grouping,
            "model": self._config.llm.model if self._config.llm is not None else None,
        }
        digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]  # noqa: S324
        return base / f"{self._config.data.dataset}_{self._config.data.kind}_{digest}.jsonl"

    async def _drain(
        self,
        items: list[_Item],
        predict_one: Callable[[int, _Item], Awaitable[_Record]],
        store: PredictionStore | SubtopicStore,
        *,
        unit: str,
        desc: str,
    ) -> tuple[list[_Record], str | None]:
        """Score ``items`` concurrently, checkpointing successes and honouring the circuit breaker.

        Shared by both paths: only ``predict_one`` (and the progress labels) differ. Returns the
        records in input order plus the breaker's abort reason, if any.
        """
        cap = self._config.run.max_predictions
        if cap is not None and cap < len(items):
            logger.warning("Capping predictions", unit=unit, cap=cap, total=len(items))
            items = items[:cap]

        total = len(items)
        results: list[_Record | None] = [None] * total
        store_lock = asyncio.Lock()
        state = _BreakerState(self._config.run.max_consecutive_failures)
        progress = _ProgressTracker(total, self._config.run.progress_log_every, unit=unit, desc=desc)
        # A fixed pool of workers drains one shared iterator, rather than scheduling a task per item
        # up front — memory stays flat for large datasets. next() runs synchronously, so sharing the
        # iterator is safe under asyncio's single-threaded scheduling.
        work = iter(enumerate(items))

        async def worker() -> None:
            for index, item in work:
                if state.aborted:
                    return  # breaker tripped: stop pulling work, leave the rest unscored for warm-start
                record = await predict_one(index, item)
                results[index] = record  # indexed write keeps output aligned with input order
                if record.error is None:
                    async with store_lock:
                        store.append(asdict(record))
                state.register(record)
                progress.update(record)

        # LLM-free predictors (baselines) are synchronous, so concurrency buys nothing: one worker.
        configured = self._config.llm.max_concurrency if self._config.llm is not None else 1
        n_workers = min(configured, total) if total > 0 else 0
        try:
            await asyncio.gather(*(worker() for _ in range(n_workers)))
        finally:
            progress.close()
        return [record for record in results if record is not None], state.abort_reason


class PairwiseRunner(ExperimentRunner):

    """Scores candidate pairs with a PairPredictor and clusters them via union-find."""

    _clusters: UnionFind | None = None  # live transitive-grouping union-find, set in _execute

    async def _execute(self) -> None:
        test_mentions = self._load_split(self._config.data.eval_split)
        test_candidates = self._build_candidates(test_mentions)
        few_shot_examples = self._build_few_shot_pool()
        self._predictor.bind_examples(few_shot_examples)
        self._log_dataset_stats(test_mentions, test_candidates, len(few_shot_examples))

        store = PredictionStore(self._checkpoint_path())
        try:
            prior_records, pending = self._resolve_pending(store, test_candidates)
            # Dynamic transitive grouping: a live union-find over positive predictions lets us infer
            # A~C once A~B and B~C are decided, skipping its LLM request. None when disabled. Seeded
            # from prior positive records so resumed runs keep inferring across the checkpoint boundary.
            self._clusters = (
                self._seed_clusters(pending, prior_records) if self._config.run.transitive_grouping else None
            )
            new_records, abort_reason = await self._drain(
                pending,
                self._predict_one,
                store,
                unit="pair",
                desc="Scoring pairs",
            )
        finally:
            store.close()
        records = prior_records + new_records
        self._log_evaluation(records, test_mentions)

        if self._config.run.persist_predictions:
            self._log_predictions_artifact(records, few_shot_examples)

        if abort_reason is not None:
            raise RunAbortedError(abort_reason)

    def _resolve_pending(
        self,
        store: PredictionStore,
        candidates: list[CandidatePair],
    ) -> tuple[list[PredictionRecord], list[CandidatePair]]:
        if not self._config.run.resume:
            return [], candidates
        if store.is_empty():
            logger.warning("Resume requested but checkpoint is empty — scoring all candidates")
            return [], candidates
        prior_records = [PredictionRecord(**record) for record in store.records()]
        pending = [
            pair
            for pair in candidates
            if not store.is_done(pair.mention_a.mention_id, pair.mention_b.mention_id)
        ]
        logger.info(
            "Resuming from checkpoint",
            total=len(candidates),
            already_done=len(prior_records),
            pending=len(pending),
        )
        return prior_records, pending

    def _build_candidates(self, mentions: MentionCollection) -> list[CandidatePair]:
        generator = CandidateGenerator(
            max_candidates=self._config.candidates.max_candidates,
            max_candidates_per_mention=self._config.candidates.max_candidates_per_mention,
            seed=self._config.candidates.seed,
            strategy=self._config.candidates.strategy,
        )
        return generator.generate(mentions)

    def _build_few_shot_pool(self) -> list[LabeledPairView]:
        few_shot_cfg = self._config.approach.few_shot
        if few_shot_cfg is None or few_shot_cfg.num_examples <= 0:
            return []
        if few_shot_cfg.strategy is FewShotStrategy.ALL_DATASETS:
            positives, negatives = self._few_shot_pairs_all_datasets()
        else:
            train_kind = few_shot_cfg.train_kind if few_shot_cfg.train_kind is not None else self._config.data.kind
            train_dataset = (
                few_shot_cfg.train_dataset if few_shot_cfg.train_dataset is not None else self._config.data.dataset
            )
            pairs = self._few_shot_pairs(train_dataset, train_kind)
            positives = [p for p in pairs if p.is_coreferent]
            negatives = [p for p in pairs if not p.is_coreferent]
        picked = self._balanced_pick(positives, negatives, few_shot_cfg.num_examples)
        if len(picked) < few_shot_cfg.num_examples:
            logger.warning(
                "Few-shot pool smaller than requested",
                requested=few_shot_cfg.num_examples,
                available=len(picked),
            )
        return [LabeledPairView.from_pair(p) for p in picked]

    def _few_shot_pairs(self, dataset: DatasetName, kind: MentionKind) -> list[CandidatePair]:
        """All-pairs candidates from a deterministic sample of one dataset's train split.

        Mentions are subsampled to at most few_shot.pool_mentions *before* pairing, so the work is
        O(pool_mentions^2) per (dataset, kind) rather than the full split's n^2 — the difference
        between thousands of pairs and the hundreds of millions a 50k-mention split would enumerate.
        The fixed seed keeps the selected mentions, and hence the example pool, constant across runs.
        """
        train_mentions = self._loader.load(dataset, Split.TRAIN, kind)
        if self._config.data.exclude_singletons:
            train_mentions = train_mentions.without_singletons()
        train_mentions = self._sample_few_shot_mentions(train_mentions)
        return CandidateGenerator(strategy=CandidateStrategy.ALL_PAIRS).generate(train_mentions)

    def _sample_few_shot_mentions(self, mentions: MentionCollection) -> MentionCollection:
        pool = self._config.approach.few_shot.pool_mentions
        ordered = sorted(mentions, key=lambda m: m.mention_id)
        if pool is None or len(ordered) <= pool:
            return MentionCollection(ordered, mentions.dataset)
        rng = random.Random(self._config.approach.few_shot.seed)
        return MentionCollection(rng.sample(ordered, pool), mentions.dataset)

    def _few_shot_pairs_any_kind(self, dataset: DatasetName) -> list[CandidatePair]:
        """One dataset's train candidates across entity and event kinds.

        Loads both kinds when both have a train split, otherwise whichever single kind exists. Pairs
        are generated per kind and concatenated, so no meaningless cross-kind (entity vs. event)
        pairs are produced.
        """
        pairs: list[CandidatePair] = []
        for kind in (MentionKind.ENTITY, MentionKind.EVENT):
            try:
                pairs.extend(self._few_shot_pairs(dataset, kind))
            except FileNotFoundError:
                logger.warning("No train split for kind", dataset=dataset.value, kind=kind.value)
        return pairs

    def _few_shot_pairs_all_datasets(self) -> tuple[list[CandidatePair], list[CandidatePair]]:
        """Positive and negative train pairs round-robined across every dataset.

        Each pool spans domains instead of draining the alphabetically-first dataset, and every
        dataset contributes both mention kinds when available (see _few_shot_pairs_any_kind).
        """
        per_dataset_pos: list[list[CandidatePair]] = []
        per_dataset_neg: list[list[CandidatePair]] = []
        for dataset in sorted(DatasetName, key=lambda d: d.value):
            pairs = self._few_shot_pairs_any_kind(dataset)
            positives = [p for p in pairs if p.is_coreferent]
            negatives = [p for p in pairs if not p.is_coreferent]
            if positives:
                per_dataset_pos.append(positives)
            if negatives:
                per_dataset_neg.append(negatives)
        return self._round_robin(per_dataset_pos), self._round_robin(per_dataset_neg)

    @staticmethod
    def _round_robin(per_dataset: list[list[CandidatePair]]) -> list[CandidatePair]:
        flat: list[CandidatePair] = []
        for layer in zip_longest(*per_dataset):
            flat.extend(pair for pair in layer if pair is not None)
        return flat

    @staticmethod
    def _balanced_pick(
        positives: list[CandidatePair],
        negatives: list[CandidatePair],
        num_examples: int,
    ) -> list[CandidatePair]:
        """Build a pool of up to num_examples that is half positive and half negative.

        When one class is short, top up from the other so the pool still reaches num_examples
        (positives are rare in all-pairs candidates). Positives and negatives are interleaved so the
        pool alternates label rather than front-loading one class.
        """
        n_pos = min(len(positives), num_examples // 2)
        n_neg = min(len(negatives), num_examples - n_pos)
        n_pos = min(len(positives), num_examples - n_neg)  # top up positives if negatives were short
        picked: list[CandidatePair] = []
        for pos, neg in zip_longest(positives[:n_pos], negatives[:n_neg]):
            if pos is not None:
                picked.append(pos)
            if neg is not None:
                picked.append(neg)
        return picked

    async def _predict_one(self, index: int, pair: CandidatePair) -> PredictionRecord:
        a_id, b_id = pair.mention_a.mention_id, pair.mention_b.mention_id
        # Dynamic transitive grouping: when A~B and B~C are already linked, infer A~C without an LLM
        # request. Best-effort under concurrency — in-flight pairs may not have updated the union-find
        # yet, so we skip fewer pairs than a serial pass would, never more.
        if self._clusters is not None and self._clusters.connected(a_id, b_id):
            return self._transitive_record(index, pair)  # already linked: no LLM request
        start = time.monotonic()
        try:
            decision = await self._predictor.predict(pair)
        except LLMError as err:
            logger.warning("Prediction failed", index=index, category=err.category, error=str(err))
            return PredictionRecord(
                pair_index=index,
                mention_a_id=a_id,
                mention_b_id=b_id,
                gold_coreferent=pair.is_coreferent,
                predicted_coreferent=None,
                reasoning=None,
                error=str(err),
                error_category=err.category,
                http_status=err.http_status,
                latency_seconds=time.monotonic() - start,
            )
        if self._clusters is not None and decision.coreferent:
            self._clusters.union(a_id, b_id)
        return PredictionRecord(
            pair_index=index,
            mention_a_id=a_id,
            mention_b_id=b_id,
            gold_coreferent=pair.is_coreferent,
            predicted_coreferent=decision.coreferent,
            reasoning=decision.reasoning,
            error=None,
            latency_seconds=time.monotonic() - start,
        )

    @staticmethod
    def _seed_clusters(candidates: list[CandidatePair], prior_records: list[PredictionRecord]) -> UnionFind:
        node_ids = {pair.mention_a.mention_id for pair in candidates}
        node_ids.update(pair.mention_b.mention_id for pair in candidates)
        clusters = UnionFind(node_ids)
        for record in prior_records:
            if record.predicted_coreferent and record.mention_a_id in clusters and record.mention_b_id in clusters:
                clusters.union(record.mention_a_id, record.mention_b_id)
        return clusters

    @staticmethod
    def _transitive_record(index: int, pair: CandidatePair) -> PredictionRecord:
        return PredictionRecord(
            pair_index=index,
            mention_a_id=pair.mention_a.mention_id,
            mention_b_id=pair.mention_b.mention_id,
            gold_coreferent=pair.is_coreferent,
            predicted_coreferent=True,
            reasoning="Inferred coreferent via positive transitive closure (no LLM request).",
            error=None,
            latency_seconds=0.0,
            transitive=True,
        )

    @staticmethod
    def _build_predicted_clusters(
        records: list[PredictionRecord],
        mentions: MentionCollection,
    ) -> list[list[str]]:
        """Predicted clustering: union-find over positive pairs, leftover mentions as singletons.

        Mentions that never appear in a positive prediction (including failed, capped, or unresumed
        pairs) stay in their own singleton, matching the "failures = non-links" convention.
        """
        parent: dict[str, str] = {mention.mention_id: mention.mention_id for mention in mentions}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: str, b: str) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_a] = root_b

        for record in records:
            if (
                record.predicted_coreferent
                and record.mention_a_id in parent
                and record.mention_b_id in parent
            ):
                union(record.mention_a_id, record.mention_b_id)

        groups: dict[str, list[str]] = defaultdict(list)
        for mention_id in parent:
            groups[find(mention_id)].append(mention_id)
        return list(groups.values())

    @staticmethod
    def _compute_dataset_stats(
        mentions: MentionCollection,
        candidates: list[CandidatePair],
        few_shot_pool_size: int,
    ) -> dict[str, float]:
        coreferent = sum(1 for c in candidates if c.is_coreferent)
        return {
            "n_mentions": float(len(mentions)),
            "n_candidates": float(len(candidates)),
            "n_coreferent_gold": float(coreferent),
            "coreferent_ratio_gold": coreferent / len(candidates) if candidates else 0.0,
            "few_shot_pool_size": float(few_shot_pool_size),
        }

    def _log_dataset_stats(
        self,
        mentions: MentionCollection,
        candidates: list[CandidatePair],
        few_shot_pool_size: int,
    ) -> None:
        stats = self._compute_dataset_stats(mentions, candidates, few_shot_pool_size)
        self._tracker.log_metrics(stats)
        logger.info("Logged dataset stats", **stats)

    @staticmethod
    def _compute_metrics(records: list[PredictionRecord], mentions: MentionCollection) -> dict[str, float]:
        predicted = PairwiseRunner._build_predicted_clusters(
            [r for r in records if r.predicted_coreferent is not None],
            mentions,
        )
        metrics = score_clusters(mentions.gold_clusters(), predicted)
        failures = sum(1 for r in records if r.error is not None)
        scored = sum(1 for r in records if r.predicted_coreferent is not None)
        transitive = sum(1 for r in records if r.transitive)
        latencies = [r.latency_seconds for r in records if r.error is None and not r.transitive]
        metrics["n_candidates"] = float(len(records))
        metrics["n_scored"] = float(scored)
        # Every record is either a transitive inference or one LLM request (success or failure);
        # n_transitive is the count of requests saved by dynamic transitive grouping.
        metrics["n_transitive"] = float(transitive)
        metrics["n_llm_requests"] = float(len(records) - transitive)
        metrics["failure_count"] = float(failures)
        metrics["failure_rate"] = failures / len(records) if records else 0.0
        metrics["mean_latency_seconds"] = sum(latencies) / len(latencies) if latencies else 0.0
        by_category = Counter(r.error_category for r in records if r.error is not None and r.error_category)
        for category, count in by_category.items():
            metrics[f"failure_{category}"] = float(count)
        return metrics

    def _log_evaluation(self, records: list[PredictionRecord], mentions: MentionCollection) -> None:
        metrics = self._compute_metrics(records, mentions)
        self._tracker.log_metrics(metrics)
        logger.info("Logged evaluation", **metrics)
        if metrics["failure_rate"] > _HIGH_FAILURE_RATE:
            logger.warning(
                "High failure rate — coreference metrics treat failed pairs as non-links and are unreliable",
                failure_rate=metrics["failure_rate"],
                n_scored=metrics["n_scored"],
                n_candidates=metrics["n_candidates"],
            )

    def _log_predictions_artifact(
        self,
        records: list[PredictionRecord],
        few_shot_examples: list[LabeledPairView],
    ) -> None:
        self._tracker.log_jsonl(
            (asdict(r) for r in records),
            artifact_file="predictions.jsonl",
        )
        self._tracker.log_dict(
            {"examples": [e.model_dump() for e in few_shot_examples]},
            "few_shot_pool.json",
        )


class SubtopicRunner(ExperimentRunner):

    """End-to-end path: scores whole subtopics with a SubtopicPredictor.

    The LLM returns the chains for each subtopic directly, folding candidate generation and
    decision-making into one call. Per-subtopic chains are stitched into a global clustering, with a
    singleton for every uncovered mention, then scored with the same score_clusters as the pairwise
    path.
    """

    async def _execute(self) -> None:
        test_mentions = self._load_split(self._config.data.eval_split)
        subtopics = sorted(test_mentions.by_subtopic().items())
        self._log_subtopic_dataset_stats(test_mentions, subtopics)

        store = SubtopicStore(self._checkpoint_path())
        try:
            prior_records, pending = self._resolve_pending_subtopics(store, subtopics)
            new_records, abort_reason = await self._drain(
                pending,
                self._predict_one_subtopic,
                store,
                unit="subtopic",
                desc="Scoring subtopics",
            )
        finally:
            store.close()
        records = prior_records + new_records
        predicted_clusters = self._assemble_clusters(records, test_mentions)
        self._log_subtopic_evaluation(records, test_mentions, predicted_clusters)

        if self._config.run.persist_predictions:
            self._log_subtopic_artifact(records)

        if abort_reason is not None:
            raise RunAbortedError(abort_reason)

    def _resolve_pending_subtopics(
        self,
        store: SubtopicStore,
        subtopics: list[tuple[str, MentionCollection]],
    ) -> tuple[list[SubtopicRecord], list[tuple[str, MentionCollection]]]:
        if not self._config.run.resume:
            return [], subtopics
        if store.is_empty():
            logger.warning("Resume requested but checkpoint is empty — scoring all subtopics")
            return [], subtopics
        prior_records = [SubtopicRecord(**record) for record in store.records()]
        pending = [(subtopic_id, mentions) for subtopic_id, mentions in subtopics if not store.is_done(subtopic_id)]
        logger.info(
            "Resuming from checkpoint",
            total=len(subtopics),
            already_done=len(prior_records),
            pending=len(pending),
        )
        return prior_records, pending

    async def _predict_one_subtopic(
        self,
        index: int,  # noqa: ARG002 — _drain passes the work index; subtopics are keyed by id, not index
        subtopic: tuple[str, MentionCollection],
    ) -> SubtopicRecord:
        subtopic_id, mentions = subtopic
        start = time.monotonic()
        try:
            decision = await self._predictor.predict(mentions)
        except LLMError as err:
            logger.warning("Subtopic prediction failed", subtopic_id=subtopic_id, category=err.category, error=str(err))
            return SubtopicRecord(
                subtopic_id=subtopic_id,
                n_mentions=len(mentions),
                n_predicted_clusters=0,
                clusters=[],
                error=str(err),
                error_category=err.category,
                http_status=err.http_status,
                latency_seconds=time.monotonic() - start,
            )
        return SubtopicRecord(
            subtopic_id=subtopic_id,
            n_mentions=len(mentions),
            n_predicted_clusters=len(decision.clusters),
            clusters=decision.clusters,
            error=None,
            latency_seconds=time.monotonic() - start,
        )

    @staticmethod
    def _assemble_clusters(
        records: list[SubtopicRecord],
        mentions: MentionCollection,
    ) -> list[list[str]]:
        """Global predicted clustering: the LLM chains plus a singleton for every uncovered mention.

        Mentions in failed, capped, or unresumed subtopics fall through to singletons, matching the
        pairwise path's "failures = non-links" convention.
        """
        predicted_clusters = [cluster for record in records for cluster in record.clusters]
        covered = {mention_id for cluster in predicted_clusters for mention_id in cluster}
        predicted_clusters.extend([mention.mention_id] for mention in mentions if mention.mention_id not in covered)
        return predicted_clusters

    @staticmethod
    def _compute_subtopic_metrics(
        records: list[SubtopicRecord],
        mentions: MentionCollection,
        predicted_clusters: list[list[str]],
    ) -> dict[str, float]:
        metrics = score_clusters(mentions.gold_clusters(), predicted_clusters)
        failures = sum(1 for r in records if r.error is not None)
        latencies = [r.latency_seconds for r in records if r.error is None]
        metrics["n_mentions"] = float(len(mentions))
        metrics["n_subtopics"] = float(len(records))
        metrics["n_failed_subtopics"] = float(failures)
        metrics["failure_rate"] = failures / len(records) if records else 0.0
        metrics["mean_latency_seconds"] = sum(latencies) / len(latencies) if latencies else 0.0
        by_category = Counter(r.error_category for r in records if r.error is not None and r.error_category)
        for category, count in by_category.items():
            metrics[f"failure_{category}"] = float(count)
        return metrics

    def _log_subtopic_dataset_stats(
        self,
        mentions: MentionCollection,
        subtopics: list[tuple[str, MentionCollection]],
    ) -> None:
        n_subtopics = len(subtopics)
        stats = {
            "n_mentions": float(len(mentions)),
            "n_subtopics": float(n_subtopics),
            "mean_mentions_per_subtopic": len(mentions) / n_subtopics if n_subtopics > 0 else 0.0,
            "max_mentions_per_subtopic": float(max((len(m) for _, m in subtopics), default=0)),
        }
        self._tracker.log_metrics(stats)
        logger.info("Logged dataset stats", **stats)

    def _log_subtopic_evaluation(
        self,
        records: list[SubtopicRecord],
        mentions: MentionCollection,
        predicted_clusters: list[list[str]],
    ) -> None:
        metrics = self._compute_subtopic_metrics(records, mentions, predicted_clusters)
        self._tracker.log_metrics(metrics)
        logger.info("Logged evaluation", **metrics)
        if metrics["failure_rate"] > _HIGH_FAILURE_RATE:
            logger.warning(
                "High failure rate — failed subtopics fall back to singletons, so metrics are unreliable",
                failure_rate=metrics["failure_rate"],
                n_failed_subtopics=metrics["n_failed_subtopics"],
                n_subtopics=metrics["n_subtopics"],
            )

    def _log_subtopic_artifact(self, records: list[SubtopicRecord]) -> None:
        self._tracker.log_jsonl(
            (asdict(r) for r in records),
            artifact_file="subtopic_predictions.jsonl",
        )


def build_runner(
    config: ExperimentConfig,
    loader: DatasetLoader,
    tracker: MLflowTracker,
    predictor: PairPredictor | SubtopicPredictor,
) -> ExperimentRunner:
    """Pick the runner for the predictor's unit of work (subtopic vs. candidate pair)."""
    if isinstance(predictor, SubtopicPredictor):
        return SubtopicRunner(config, loader, tracker, predictor)
    return PairwiseRunner(config, loader, tracker, predictor)
