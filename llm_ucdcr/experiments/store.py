import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TextIO

from llm_ucdcr.logs.accessors import get_logger

logger = get_logger(__name__)


class _JsonlCheckpointStore(ABC):

    """Append-only JSONL of completed work units for warm-start.

    Source of truth that survives crashes: each successful result is flushed as its own line, so an
    interrupted run keeps everything done so far. Only successes are stored — failures are
    deliberately absent so a resumed run retries exactly the units that never got a result.
    Subclasses define how a record maps to its dedup key via ``_record_key``.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._done: dict[str, dict[str, Any]] = {}
        self._handle: TextIO | None = None
        self._load()

    @abstractmethod
    def _record_key(self, record: dict[str, Any]) -> str: ...

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("No checkpoint file found — starting with an empty store", path=str(self._path))
            return
        malformed = 0
        with self._path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    malformed += 1  # tolerate a torn final line from a hard kill
                    continue
                self._done[self._record_key(record)] = record
        logger.info("Loaded checkpoint", path=str(self._path), records=len(self._done), malformed=malformed)

    def is_empty(self) -> bool:
        return not self._done

    def records(self) -> list[dict[str, Any]]:
        return list(self._done.values())

    def append(self, record: dict[str, Any]) -> None:
        key = self._record_key(record)
        if key in self._done:
            return
        if self._handle is None:  # open once and keep it; per-record flush keeps the file crash-safe
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a")
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        self._done[key] = record

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class PredictionStore(_JsonlCheckpointStore):

    """Pair-keyed checkpoint of completed pairwise predictions."""

    @staticmethod
    def pair_key(mention_a_id: str, mention_b_id: str) -> str:
        """Stable, order-independent key for a mention pair.

        Keyed on mention ids (not pair_index, which shifts with list order/capping between runs),
        so warm-start matches a pair regardless of which run produced it.
        """
        low, high = sorted((mention_a_id, mention_b_id))
        return f"{low}|{high}"

    def _record_key(self, record: dict[str, Any]) -> str:
        return self.pair_key(record["mention_a_id"], record["mention_b_id"])

    def is_done(self, mention_a_id: str, mention_b_id: str) -> bool:
        return self.pair_key(mention_a_id, mention_b_id) in self._done


class SubtopicStore(_JsonlCheckpointStore):

    """Subtopic-keyed checkpoint of completed end-to-end subtopic predictions."""

    def _record_key(self, record: dict[str, Any]) -> str:
        return record["subtopic_id"]

    def is_done(self, subtopic_id: str) -> bool:
        return subtopic_id in self._done
