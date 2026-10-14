import httpx
import openai
import pytest

from llm_ucdcr.llm.errors import FATAL_CATEGORIES, LLMError

_REQUEST = httpx.Request("POST", "https://api.test/v1/chat/completions")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQUEST)


def test_timeout() -> None:
    info = LLMError.from_exception(openai.APITimeoutError(request=_REQUEST))
    assert info.category == "timeout"
    assert info.fatal is False


def test_rate_limit() -> None:
    info = LLMError.from_exception(openai.RateLimitError("slow down", response=_response(429), body=None))
    assert info.category == "rate_limit"
    assert info.http_status == 429
    assert info.fatal is False


def test_auth_is_fatal() -> None:
    info = LLMError.from_exception(openai.AuthenticationError("bad key", response=_response(401), body=None))
    assert info.category == "auth"
    assert info.fatal is True


def test_payment_402_is_fatal() -> None:
    info = LLMError.from_exception(openai.APIStatusError("insufficient balance", response=_response(402), body=None))
    assert info.category == "payment"
    assert info.http_status == 402
    assert info.fatal is True


def test_bad_request() -> None:
    info = LLMError.from_exception(openai.BadRequestError("nope", response=_response(400), body=None))
    assert info.category == "bad_request"
    assert info.fatal is False


def test_unknown_is_other() -> None:
    info = LLMError.from_exception(RuntimeError("boom"))
    assert info.category == "other"
    assert info.fatal is False


def test_llm_error_passes_through() -> None:
    original = LLMError("truncated", None, "response truncated: finish_reason=length")
    assert LLMError.from_exception(original) is original


@pytest.mark.parametrize("category", ["auth", "permission", "payment"])
def test_fatal_categories_are_fatal(category: str) -> None:
    assert category in FATAL_CATEGORIES
    assert LLMError(category, None, "x").fatal is True
