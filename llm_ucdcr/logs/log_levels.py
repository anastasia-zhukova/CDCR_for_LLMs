import logging
from enum import IntEnum


SENSITIVE_DEBUG = 5
logging.addLevelName(SENSITIVE_DEBUG, "SENSITIVE_DEBUG")


class LogLevel(IntEnum):
    SENSITIVE_DEBUG = SENSITIVE_DEBUG
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
