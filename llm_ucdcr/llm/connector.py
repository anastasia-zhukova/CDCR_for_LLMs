import json
from typing import Any, Literal, TypeVar, cast

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from llm_ucdcr.llm.errors import LLMError
from llm_ucdcr.logs.accessors import get_logger

logger = get_logger(__name__)

OutputT = TypeVar("OutputT", bound=BaseModel)

# How the connector asks the endpoint for structured output:
#   "json_schema"  -> native structured outputs via the SDK's .parse() helper (OpenAI, vLLM, ...)
#   "json_object"  -> plain JSON mode; the schema is described in the prompt and parsed client-side.
# Some OpenAI-compatible endpoints (e.g. DeepSeek) reject the json_schema response_format and only
# support json_object, hence the toggle.
OutputMode = Literal["json_schema", "json_object"]


class LLMConnector:

    """A thin client over an OpenAI-compatible chat endpoint.

    Purely about *making the call*: the caller supplies the system prompt, the user
    messages, and the Pydantic model the response must parse into. All prompt-building
    ("business") logic lives in the approach's predictor, not here.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        params: dict[str, Any] | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        output_mode: OutputMode = "json_schema",
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self._model = model
        self._params = params or {}
        self._output_mode: OutputMode = output_mode
        if client is not None:
            self._client = client
        else:
            extra: dict[str, Any] = {}
            if max_retries is not None:
                extra["max_retries"] = max_retries
            if timeout is not None:
                extra["timeout"] = timeout
            self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, **extra)

    async def predict(
        self,
        *,
        system_prompt: str,
        messages: list[ChatCompletionMessageParam],
        output_model: type[OutputT],
    ) -> OutputT:
        system_content = system_prompt
        if self._output_mode == "json_object":
            # json_object mode does not enforce a schema, so describe it in the prompt. DeepSeek also
            # requires the literal word "JSON" somewhere in the messages, which the instructions provide.
            system_content = f"{system_content}\n\n{self._json_object_instructions(output_model)}"
        full_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content},
            *messages,
        ]
        logger.debug(
            "Sending LLM request",
            model=self._model,
            output_model=output_model.__name__,
            output_mode=self._output_mode,
        )
        try:
            response = await self._request(full_messages, output_model)
        except Exception as exc:
            raise LLMError.from_exception(exc) from exc
        choice = response.choices[0]
        if choice.finish_reason in ("length", "content_filter"):
            # A truncated response can still parse into a valid-looking object, so treat it as a failure.
            raise LLMError("truncated", None, f"response truncated: finish_reason={choice.finish_reason}")
        parsed = self._extract(choice, output_model)
        if parsed is None:
            raise LLMError("empty_response", None, "LLM returned empty response content")
        logger.debug(
            "Received LLM response",
            model=self._model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
        )
        return cast("OutputT", parsed)

    async def _request(
        self,
        messages: list[ChatCompletionMessageParam],
        output_model: type[BaseModel],
    ) -> Any:
        if self._output_mode == "json_object":
            return await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                **self._params,
            )
        return await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=output_model,
            **self._params,
        )

    def _extract(self, choice: Any, output_model: type[BaseModel]) -> BaseModel | None:
        if self._output_mode != "json_object":
            return choice.message.parsed
        content = choice.message.content
        if not content:
            return None
        try:
            return output_model.model_validate_json(content)
        except ValidationError as exc:
            raise LLMError("invalid_response", None, f"response did not match schema: {exc}") from exc

    def _json_object_instructions(self, output_model: type[BaseModel]) -> str:
        schema = json.dumps(output_model.model_json_schema(), separators=(",", ":"))
        return (
            "Respond with a single JSON object that conforms to this JSON Schema:\n"
            f"{schema}\n"
            "Output only the JSON object — no surrounding prose, explanations, or code fences."
        )
