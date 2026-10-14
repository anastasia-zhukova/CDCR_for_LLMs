from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pytest

from llm_ucdcr.constants import DATA_ROOT
from llm_ucdcr.data.loader import DatasetLoader, DatasetName, MentionCollection, MentionKind, Split
from llm_ucdcr.data.mention import Mention
from llm_ucdcr.experiments.runner import PairwiseRunner, PredictionRecord, score_clusters
from llm_ucdcr.experiments.store import PredictionStore


def _metrics(records: list[PredictionRecord], mentions: MentionCollection) -> dict[str, float]:
    """Mirror PairwiseRunner scoring: union-find the positive pairs, then score against gold."""
    predicted = PairwiseRunner._build_predicted_clusters(records, mentions)
    return score_clusters(mentions.gold_clusters(), predicted)


@pytest.fixture
def meantime_test_mentions() -> MentionCollection:
    if not DATA_ROOT.exists():
        pytest.skip(f"Dataset directory not found: {DATA_ROOT}")
    loader = DatasetLoader(DATA_ROOT)
    return loader.load(DatasetName.MEANTIME, Split.TEST, MentionKind.ENTITY)


def _spanning_tree_predictions(mentions: MentionCollection) -> list[PredictionRecord]:
    """Build (N-1) positive predictions per gold cluster, enough for union-find to
    reproduce gold clustering exactly.

    Gold clusters are keyed by (coref_chain, subtopic), so the spanning trees stay within a
    subtopic: linking the same chain across subtopics would now be a false positive.
    """
    by_cluster: dict[tuple[str, str], list[Mention]] = defaultdict(list)
    for mention in mentions:
        by_cluster[mention.gold_cluster_key].append(mention)
    by_cluster = dict(by_cluster)

    records: list[PredictionRecord] = []
    index = 0
    for chain_mentions in by_cluster.values():
        if len(chain_mentions) < 2:
            continue
        anchor = chain_mentions[0]
        for other in chain_mentions[1:]:
            records.append(
                PredictionRecord(
                    pair_index=index,
                    mention_a_id=anchor.mention_id,
                    mention_b_id=other.mention_id,
                    gold_coreferent=True,
                    predicted_coreferent=True,
                    reasoning=None,
                    error=None,
                    latency_seconds=0.0,
                ),
            )
            index += 1
    return records


def test_evaluator_returns_perfect_metrics_for_gold_predictions(
    meantime_test_mentions: MentionCollection,
) -> None:
    records = _spanning_tree_predictions(meantime_test_mentions)
    metrics = _metrics(records, meantime_test_mentions)

    for metric in ("muc_f1", "b_cubed_f1", "ceafe_f1", "conll_f1"):
        assert metrics[metric] == pytest.approx(1.0), f"{metric} not perfect: {metrics[metric]}"
    for metric in ("muc_precision", "muc_recall", "b_cubed_precision", "b_cubed_recall"):
        assert metrics[metric] == pytest.approx(1.0), f"{metric} not perfect: {metrics[metric]}"
    assert metrics["n_gold_clusters"] == metrics["n_predicted_clusters"]
    assert "lea_f1" not in metrics  # LEA was dropped from the reported metrics


def test_metrics_unchanged_when_records_resumed_from_store(
    meantime_test_mentions: MentionCollection,
    tmp_path: Path,
) -> None:
    """Records reconstructed from the warm-start store must score identically to a single-pass run.

    Mirrors the resume path in ExperimentRunner: prior successes are reloaded from the store
    (PredictionRecord(**record)) and concatenated with freshly produced records before scoring.
    """
    records = _spanning_tree_predictions(meantime_test_mentions)
    baseline = _metrics(records, meantime_test_mentions)

    split = len(records) // 2
    prior, fresh = records[:split], records[split:]

    store = PredictionStore(tmp_path / "ckpt.jsonl")
    for record in prior:
        store.append(asdict(record))
    store.close()
    # Reopen so the records go through the on-disk JSONL round-trip the resume path uses.
    reopened = PredictionStore(tmp_path / "ckpt.jsonl")
    prior_from_store = [PredictionRecord(**record) for record in reopened.records()]
    reopened.close()

    resumed = _metrics(prior_from_store + fresh, meantime_test_mentions)
    assert resumed == pytest.approx(baseline)


def test_metrics_treat_unprocessed_pairs_as_singletons(
    meantime_test_mentions: MentionCollection,
) -> None:
    """Pairs that were never scored (aborted/capped run) collapse to singletons, not errors."""
    records = _spanning_tree_predictions(meantime_test_mentions)
    full = _metrics(records, meantime_test_mentions)

    partial = records[: len(records) // 2]  # the remaining pairs go unprocessed
    metrics = _metrics(partial, meantime_test_mentions)

    # Dropping links splits gold clusters apart: recall falls and more (singleton) clusters appear.
    assert metrics["b_cubed_recall"] < full["b_cubed_recall"]
    assert metrics["n_predicted_clusters"] > full["n_predicted_clusters"]
