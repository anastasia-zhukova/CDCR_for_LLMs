import json
import logging
import re
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run

from llm_ucdcr.logs.accessors import get_logger

logger = get_logger(__name__)

# structlog renders through the stdlib logger, so log lines reach handlers already formatted
# (and, for the console renderer, wrapped in ANSI colour codes). Strip them so the captured
# artifact stays plain text.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_RUN_LOG_ARTIFACT = "run.log"
_RUN_LOG_DIR = "logs"  # artifact subdirectory the run log is uploaded under


class _AnsiStrippingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_ESCAPE.sub("", super().format(record))


class MLflowTracker:

    """Thin OOP wrapper over the MLflow client.

    Owns the tracking URI and exposes a context-manager `run()` for experiment
    scoping. Credentials are read by the MLflow client directly from
    MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD env vars.
    """

    def __init__(self, tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        logger.debug("MLflow tracker configured", tracking_uri=tracking_uri)

    @contextmanager
    def run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> Iterator[Run]:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name, tags=tags) as active_run, self._capture_logs():
            run_id = active_run.info.run_id
            logger.info(
                "Started MLflow run",
                run_id=run_id,
                experiment=experiment_name,
                run_name=run_name,
            )
            if params:
                mlflow.log_params(params)
            try:
                yield active_run
            except Exception:
                logger.exception("MLflow run failed", run_id=run_id)
                raise
            else:
                logger.info("Finished MLflow run", run_id=run_id)

    @contextmanager
    def _capture_logs(self) -> Iterator[None]:
        """Tee everything logged during the run into a file and attach it to the active run.

        The artifact is uploaded in ``finally`` so a failed or aborted run still keeps its logs.
        """
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / _RUN_LOG_ARTIFACT
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(_AnsiStrippingFormatter("%(message)s"))
            root = logging.getLogger()
            root.addHandler(handler)
            try:
                yield
            finally:
                root.removeHandler(handler)
                handler.close()
                mlflow.log_artifact(str(log_path), artifact_path=_RUN_LOG_DIR)

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_dict(
        self,
        payload: dict[str, Any],
        artifact_file: str,
    ) -> None:
        mlflow.log_dict(payload, artifact_file)

    def log_jsonl(
        self,
        records: Iterable[dict[str, Any]],
        artifact_file: str,
        artifact_path: str | None = None,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / artifact_file
            with path.open("w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
