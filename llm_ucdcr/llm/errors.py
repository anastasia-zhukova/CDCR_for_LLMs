import openai

# Error categories whose presence means the whole run should abort (retrying is pointless
# and, for billing/auth, dangerous): the circuit breaker trips immediately on these.
FATAL_CATEGORIES = frozenset({"auth", "permission", "payment"})

# Typed OpenAI exceptions checked in order (subclasses before their bases): (type, category, status).
_TYPED_ERRORS: tuple[tuple[type[Exception], str, int | None], ...] = (
    (openai.APITimeoutError, "timeout", None),
    (openai.APIConnectionError, "connection", None),
    (openai.RateLimitError, "rate_limit", 429),
    (openai.AuthenticationError, "auth", 401),
    (openai.PermissionDeniedError, "permission", 403),
)
# Fallback mapping for any other APIStatusError by HTTP status. 402 Payment Required is the
# standard signal for an exhausted balance, so it is fatal (see FATAL_CATEGORIES).
_STATUS_CATEGORIES: dict[int, str] = {402: "payment", 400: "bad_request"}


class LLMError(Exception):

    """A normalized LLM failure: a stable category, optional HTTP status, and whether it is fatal.

    Raised by the connector so callers never branch on raw SDK exception types. The originating
    exception is preserved as ``__cause__`` for the full traceback.
    """

    def __init__(self, category: str, http_status: int | None, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.fatal = category in FATAL_CATEGORIES

    @classmethod
    def from_exception(cls, exc: Exception) -> "LLMError":
        """Normalize a raw exception into an LLMError.

        Uses the OpenAI SDK's typed exceptions, so the categories are provider-agnostic (they work
        for any OpenAI-compatible endpoint).
        """
        if isinstance(exc, cls):
            return exc
        message = f"{type(exc).__name__}: {exc}"
        for exc_type, category, status in _TYPED_ERRORS:
            if isinstance(exc, exc_type):
                return cls(category, status, message)
        if isinstance(exc, openai.APIStatusError):
            return cls(_STATUS_CATEGORIES.get(exc.status_code, "api_status"), exc.status_code, message)
        return cls("other", None, message)
