import logging
from dataclasses import dataclass
from typing import Literal

import structlog
from structlog.dev import ConsoleRenderer, RichTracebackFormatter
from structlog.processors import JSONRenderer

from .log_formats import LogFormat
from .log_levels import LogLevel
from .bound_logger_with_sensitive_debug_level import BoundLoggerWithSensitiveDebugLevel


timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

# Third-party loggers that emit one line per HTTP request (e.g. httpx's
# "HTTP Request: POST ... 200 OK"). They drown out our own logs during LLM
# sweeps, so we pin them to WARNING regardless of the app log level.
_NOISY_LOGGERS = ("httpx", "httpcore", "openai")


@dataclass(frozen=True)
class LogConfig:
    format: LogFormat
    level: LogLevel
    env: Literal["prod", "dev"]

    def __post_init__(self) -> None:
        if self.level <= LogLevel.SENSITIVE_DEBUG and self.env != "dev":
            raise ValueError("SENSITIVE_DEBUG level is only allowed in 'dev' environment!")

    def processors(self) -> list[structlog.typing.Processor]:
        base: list[structlog.typing.Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            timestamper,
            # Add option to invoke full stack of log execution (i.e. all function that lead to this log firing)
            structlog.processors.StackInfoRenderer(),
            # Add location of log in code.
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ],
            ),
        ]
        if self.format == LogFormat.CONSOLE:
            return [
                *base,
                ConsoleRenderer(
                    exception_formatter=RichTracebackFormatter(show_locals=True),
                ),
            ]
        return [
            *base,
            structlog.processors.dict_tracebacks,
            JSONRenderer(),
        ]

    def apply(self) -> None:
        logging.basicConfig(level=self.level, format="%(message)s", force=True)
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
        structlog.configure(
            processors=self.processors(),
            wrapper_class=BoundLoggerWithSensitiveDebugLevel,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            # NOTE: caching freezes the processor chain on first use.
            # fine for prod, but call structlog.reset_defaults() before configure_logging()
            # in test setup to avoid stale cached loggers.
            cache_logger_on_first_use=True,
        )
