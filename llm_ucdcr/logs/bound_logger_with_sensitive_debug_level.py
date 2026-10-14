import structlog

from .log_levels import SENSITIVE_DEBUG


class BoundLoggerWithSensitiveDebugLevel(structlog.stdlib.BoundLogger):
    def sensitive_debug(self, event: str, *args, **kwargs):
        return self.log(SENSITIVE_DEBUG, event, *args, **kwargs)
