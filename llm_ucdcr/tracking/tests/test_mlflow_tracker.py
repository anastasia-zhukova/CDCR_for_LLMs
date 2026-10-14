import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_ucdcr.tracking import mlflow_tracker
from llm_ucdcr.tracking.mlflow_tracker import MLflowTracker


@pytest.fixture
def captured_artifacts(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Patch the mlflow module so a run executes without a server and capture artifact contents."""
    captured: dict[str, str] = {}

    @contextmanager
    def fake_start_run(*_args: Any, **_kwargs: Any) -> Any:
        run = MagicMock()
        run.info.run_id = "run-123"
        yield run

    def fake_log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        # Read eagerly: the tracker writes into a TemporaryDirectory that is gone after the run.
        captured[Path(local_path).name] = Path(local_path).read_text(encoding="utf-8")
        captured["__artifact_path__"] = artifact_path or ""

    monkeypatch.setattr(mlflow_tracker.mlflow, "set_tracking_uri", lambda *_a, **_k: None)
    monkeypatch.setattr(mlflow_tracker.mlflow, "set_experiment", lambda *_a, **_k: None)
    monkeypatch.setattr(mlflow_tracker.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(mlflow_tracker.mlflow, "log_params", lambda *_a, **_k: None)
    monkeypatch.setattr(mlflow_tracker.mlflow, "log_artifact", fake_log_artifact)
    return captured


def test_run_uploads_log_artifact_with_run_output(captured_artifacts: dict[str, str]) -> None:
    tracker = MLflowTracker("http://example.test")
    with tracker.run(experiment_name="exp"):
        logging.getLogger("test.capture").error("a line emitted during the run")
    assert "run.log" in captured_artifacts
    assert "a line emitted during the run" in captured_artifacts["run.log"]
    assert captured_artifacts["__artifact_path__"] == "logs"


def test_run_strips_ansi_codes_from_logs(captured_artifacts: dict[str, str]) -> None:
    tracker = MLflowTracker("http://example.test")
    with tracker.run(experiment_name="exp"):
        logging.getLogger("test.capture").error("\x1b[31mcoloured\x1b[0m message")
    contents = captured_artifacts["run.log"]
    assert "coloured message" in contents
    assert "\x1b[" not in contents


def test_run_uploads_logs_even_when_body_raises(captured_artifacts: dict[str, str]) -> None:
    tracker = MLflowTracker("http://example.test")
    with pytest.raises(ValueError, match="boom"):  # noqa: PT012, SIM117
        with tracker.run(experiment_name="exp"):
            logging.getLogger("test.capture").error("logged before failure")
            raise ValueError("boom")
    assert "logged before failure" in captured_artifacts["run.log"]


@pytest.mark.usefixtures("captured_artifacts")
def test_run_removes_its_log_handler_afterwards() -> None:
    tracker = MLflowTracker("http://example.test")
    before = list(logging.getLogger().handlers)
    with tracker.run(experiment_name="exp"):
        pass
    assert logging.getLogger().handlers == before
