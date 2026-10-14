import tiktoken


class PromptTokenEstimator:

    """Counts tokens in a prompt string with a tiktoken encoding.

    DeepSeek does not expose a pre-flight token-count endpoint, so we estimate locally with an
    OpenAI encoding (``o200k_base`` by default) as a proxy. The count is informational — it is
    logged before each request so oversized subtopics are visible, but the request is sent
    regardless.
    """

    def __init__(self, encoding_name: str) -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
