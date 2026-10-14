from pathlib import Path

from llm_ucdcr.experiments.store import PredictionStore


def _record(a: str, b: str, predicted: bool = True) -> dict:
    return {
        "pair_index": 0,
        "mention_a_id": a,
        "mention_b_id": b,
        "gold_coreferent": True,
        "predicted_coreferent": predicted,
        "reasoning": None,
        "error": None,
        "latency_seconds": 0.1,
        "error_category": None,
        "http_status": None,
    }


def test_pair_key_is_order_independent() -> None:
    assert PredictionStore.pair_key("m1", "m2") == PredictionStore.pair_key("m2", "m1")


def test_new_store_is_empty(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path / "ckpt.jsonl")
    assert store.is_empty()
    store.append(_record("m1", "m2"))
    assert not store.is_empty()


def test_append_then_is_done(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path / "ckpt.jsonl")
    assert not store.is_done("m1", "m2")
    store.append(_record("m1", "m2"))
    assert store.is_done("m1", "m2")
    assert store.is_done("m2", "m1")  # order independent


def test_records_reloaded_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.jsonl"
    store = PredictionStore(path)
    store.append(_record("m1", "m2"))
    store.close()
    reloaded = PredictionStore(path)
    assert reloaded.is_done("m1", "m2")
    assert len(reloaded.records()) == 1


def test_appending_a_done_pair_again_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.jsonl"
    store = PredictionStore(path)
    store.append(_record("m1", "m2"))
    store.append(_record("m1", "m2"))  # exact same pair -> deduped, not written twice
    store.close()
    assert len(store.records()) == 1
    assert path.read_text().count("\n") == 1


def test_tolerates_truncated_final_line(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.jsonl"
    store = PredictionStore(path)
    store.append(_record("m1", "m2"))
    store.close()
    with path.open("a") as f:
        f.write('{"mention_a_id": "m3", "mention_b_id"')  # torn line from a hard kill
    reloaded = PredictionStore(path)
    assert reloaded.is_done("m1", "m2")
    assert len(reloaded.records()) == 1
