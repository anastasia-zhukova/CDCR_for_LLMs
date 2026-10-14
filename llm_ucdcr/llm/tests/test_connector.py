import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from llm_ucdcr.llm.connector import LLMConnector
from llm_ucdcr.llm.errors import LLMError

_SYSTEM_PROMPT = "You are a test assistant."
_USER_MESSAGES: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]


class _TestOutput(BaseModel):
    result: str


def _predict(connector: LLMConnector, messages: list[ChatCompletionMessageParam] | None = None) -> _TestOutput:
    return asyncio.run(
        connector.predict(
            system_prompt=_SYSTEM_PROMPT,
            messages=messages if messages is not None else _USER_MESSAGES,
            output_model=_TestOutput,
        ),
    )


def _make_mock_client(parsed: _TestOutput | None = None) -> MagicMock:
    if parsed is None:
        parsed = _TestOutput(result="hello")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = parsed
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
    return mock_client


def _make_json_object_client(content: str = '{"result": "hello"}') -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def connector() -> LLMConnector:
    return LLMConnector(model="test-model", client=_make_mock_client())


def test_predict_returns_validated_output(connector: LLMConnector) -> None:
    output = _predict(connector)
    assert isinstance(output, _TestOutput)
    assert output.result == "hello"


def test_predict_passes_system_prompt(connector: LLMConnector) -> None:
    _predict(connector)
    call_kwargs = connector._client.beta.chat.completions.parse.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == _SYSTEM_PROMPT


def test_predict_appends_user_messages(connector: LLMConnector) -> None:
    _predict(connector, [{"role": "user", "content": "some input"}])
    call_kwargs = connector._client.beta.chat.completions.parse.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "some input"


def test_predict_uses_output_model_as_response_format(connector: LLMConnector) -> None:
    _predict(connector)
    call_kwargs = connector._client.beta.chat.completions.parse.call_args.kwargs
    assert call_kwargs["response_format"] is _TestOutput


def test_predict_raises_on_empty_parsed() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = None
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
    connector = LLMConnector(model="test-model", client=mock_client)
    with pytest.raises(LLMError, match="empty response"):
        _predict(connector)


def test_json_object_mode_parses_content() -> None:
    connector = LLMConnector(
        model="test-model",
        output_mode="json_object",
        client=_make_json_object_client('{"result": "world"}'),
    )
    output = _predict(connector)
    assert isinstance(output, _TestOutput)
    assert output.result == "world"


def test_json_object_mode_requests_json_object_format() -> None:
    connector = LLMConnector(
        model="test-model",
        output_mode="json_object",
        client=_make_json_object_client(),
    )
    _predict(connector)
    call_kwargs = connector._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_json_object_mode_injects_schema_into_system_prompt() -> None:
    connector = LLMConnector(
        model="test-model",
        output_mode="json_object",
        client=_make_json_object_client(),
    )
    _predict(connector)
    call_kwargs = connector._client.chat.completions.create.call_args.kwargs
    system_content = call_kwargs["messages"][0]["content"]
    assert _SYSTEM_PROMPT in system_content
    assert "JSON" in system_content
    assert "result" in system_content  # the output model's field appears in the embedded schema


def test_json_object_mode_raises_on_invalid_json() -> None:
    connector = LLMConnector(
        model="test-model",
        output_mode="json_object",
        client=_make_json_object_client("not json at all"),
    )
    with pytest.raises(LLMError, match="did not match schema"):
        _predict(connector)
