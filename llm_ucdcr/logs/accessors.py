import structlog

from .bound_logger_with_sensitive_debug_level import BoundLoggerWithSensitiveDebugLevel


def get_logger(
    *args,
    **kwargs,
) -> BoundLoggerWithSensitiveDebugLevel:
    return structlog.get_logger(*args, **kwargs)
