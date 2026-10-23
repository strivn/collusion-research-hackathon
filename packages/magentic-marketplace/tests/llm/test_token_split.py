"""Tests that every LLM call reports input and output tokens separately.

Scenario: Token counts are split
  When any agent makes an LLM call
  Then its log entry holds input tokens and output tokens as separate numbers
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anthropic.types
import google.genai.types
import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ParsedChatCompletion,
    ParsedChatCompletionMessage,
    ParsedChoice,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from magentic_marketplace.marketplace.llm.base import LLMCallLog
from magentic_marketplace.marketplace.llm.clients.anthropic import (
    AnthropicClient,
    AnthropicConfig,
)
from magentic_marketplace.marketplace.llm.clients.gemini import (
    GeminiClient,
    GeminiConfig,
)
from magentic_marketplace.marketplace.llm.clients.openai import (
    OpenAIClient,
    OpenAIConfig,
)

# Distinct, non-symmetric numbers so a swap or a combined count is caught
INPUT_TOKENS = 11
OUTPUT_TOKENS = 7
TOTAL_TOKENS = INPUT_TOKENS + OUTPUT_TOKENS


class Answer(BaseModel):
    """Structured output used by the structured-call tests."""

    answer: str


# --- Provider SDK response builders ----------------------------------------


def anthropic_text_message(usage: anthropic.types.Usage) -> anthropic.types.Message:
    """Build an Anthropic Messages API text response."""
    return anthropic.types.Message(
        id="msg-test",
        type="message",
        role="assistant",
        content=[anthropic.types.TextBlock(type="text", text="hello")],
        model="claude-test",
        usage=usage,
    )


def anthropic_tool_message() -> anthropic.types.Message:
    """Build an Anthropic Messages API tool_use response holding an Answer."""
    return anthropic.types.Message(
        id="msg-test",
        type="message",
        role="assistant",
        content=[
            anthropic.types.ToolUseBlock(
                type="tool_use",
                id="tool-test",
                name="generateAnswer",
                input={"answer": "42"},
            )
        ],
        model="claude-test",
        usage=anthropic.types.Usage(
            input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS
        ),
    )


def openai_usage() -> CompletionUsage:
    """Build an OpenAI Chat Completions usage block."""
    return CompletionUsage(
        prompt_tokens=INPUT_TOKENS,
        completion_tokens=OUTPUT_TOKENS,
        total_tokens=TOTAL_TOKENS,
    )


def openai_text_completion(usage: CompletionUsage | None) -> ChatCompletion:
    """Build an OpenAI Chat Completions text response."""
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model="gpt-4o-test",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="hello"),
            )
        ],
        usage=usage,
    )


def openai_parsed_completion() -> ParsedChatCompletion[Answer]:
    """Build an OpenAI Chat Completions parse() response holding an Answer."""
    return ParsedChatCompletion[Answer](
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model="gpt-4o-test",
        choices=[
            ParsedChoice[Answer](
                index=0,
                finish_reason="stop",
                message=ParsedChatCompletionMessage[Answer](
                    role="assistant",
                    content='{"answer": "42"}',
                    parsed=Answer(answer="42"),
                ),
            )
        ],
        usage=openai_usage(),
    )


def gemini_usage() -> google.genai.types.GenerateContentResponseUsageMetadata:
    """Build a Gemini usage_metadata block."""
    return google.genai.types.GenerateContentResponseUsageMetadata(
        prompt_token_count=INPUT_TOKENS,
        candidates_token_count=OUTPUT_TOKENS,
        total_token_count=TOTAL_TOKENS,
    )


def gemini_response(
    text: str,
    usage: google.genai.types.GenerateContentResponseUsageMetadata | None,
) -> google.genai.types.GenerateContentResponse:
    """Build a Gemini generate_content response with one text candidate."""
    return google.genai.types.GenerateContentResponse(
        candidates=[
            google.genai.types.Candidate(
                content=google.genai.types.Content(
                    role="model", parts=[google.genai.types.Part(text=text)]
                ),
                finish_reason=google.genai.types.FinishReason.STOP,
            )
        ],
        usage_metadata=usage,
    )


# --- Client builders: the provider SDK client is replaced by a mock ---------


def anthropic_client() -> AnthropicClient:
    """Build an AnthropicClient whose SDK client is a mock."""
    client = AnthropicClient(
        AnthropicConfig(provider="anthropic", api_key="test-key", model="claude-test")
    )
    client.client = MagicMock()
    return client


def openai_client() -> OpenAIClient:
    """Build an OpenAIClient whose SDK client is a mock."""
    client = OpenAIClient(
        OpenAIConfig(provider="openai", api_key="test-key", model="gpt-4o-test")
    )
    client.client = MagicMock()
    return client


def gemini_client() -> GeminiClient:
    """Build a GeminiClient whose SDK client is a mock."""
    client = GeminiClient(
        GeminiConfig(provider="gemini", api_key="test-key", model="gemini-test")
    )
    client.client = MagicMock()
    return client


def logged_call(logger: MagicMock) -> LLMCallLog:
    """Return the single LLMCallLog passed to logger.debug."""
    assert logger.debug.call_count == 1
    data: Any = logger.debug.call_args.kwargs["data"]
    assert isinstance(data, LLMCallLog)
    return data


# --- Anthropic ---------------------------------------------------------------


class TestAnthropicTokenSplit:
    """Anthropic fills input/output tokens from response.usage."""

    async def test_plain_call_splits_tokens(self):
        """A plain call reports usage.input_tokens and usage.output_tokens."""
        client = anthropic_client()
        client.client.messages.create = AsyncMock(
            return_value=anthropic_text_message(
                anthropic.types.Usage(
                    input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS
                )
            )
        )

        _, usage = await client.generate("hi")

        assert usage.input_tokens == INPUT_TOKENS
        assert usage.output_tokens == OUTPUT_TOKENS
        assert usage.token_count == TOTAL_TOKENS

    async def test_structured_call_splits_tokens(self):
        """A structured (tool_use) call reports input and output tokens."""
        client = anthropic_client()
        client.client.messages.create = AsyncMock(return_value=anthropic_tool_message())

        response, usage = await client.generate("hi", response_format=Answer)

        assert response == Answer(answer="42")
        assert usage.input_tokens == INPUT_TOKENS
        assert usage.output_tokens == OUTPUT_TOKENS
        assert usage.token_count == TOTAL_TOKENS

    async def test_plain_call_is_logged_with_split_tokens(self):
        """The LLMCallLog for a plain call carries input and output tokens."""
        client = anthropic_client()
        client.client.messages.create = AsyncMock(
            return_value=anthropic_text_message(
                anthropic.types.Usage(
                    input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS
                )
            )
        )
        logger = MagicMock()

        await client.generate("hi", logger=logger)

        log = logged_call(logger)
        assert log.success is True
        assert log.input_tokens == INPUT_TOKENS
        assert log.output_tokens == OUTPUT_TOKENS
        assert log.token_count == TOTAL_TOKENS


# --- OpenAI --------------------------------------------------------------------


class TestOpenAITokenSplit:
    """OpenAI fills input/output tokens from prompt_tokens/completion_tokens."""

    async def test_plain_call_splits_tokens(self):
        """A plain call reports prompt_tokens and completion_tokens."""
        client = openai_client()
        client.client.chat.completions.create = AsyncMock(
            return_value=openai_text_completion(openai_usage())
        )

        _, usage = await client.generate("hi")

        assert usage.input_tokens == INPUT_TOKENS
        assert usage.output_tokens == OUTPUT_TOKENS
        assert usage.token_count == TOTAL_TOKENS

    async def test_structured_call_splits_tokens(self):
        """A structured (parse) call reports prompt_tokens and completion_tokens."""
        client = openai_client()
        client.client.chat.completions.parse = AsyncMock(
            return_value=openai_parsed_completion()
        )

        response, usage = await client.generate("hi", response_format=Answer)

        assert response == Answer(answer="42")
        assert usage.input_tokens == INPUT_TOKENS
        assert usage.output_tokens == OUTPUT_TOKENS
        assert usage.token_count == TOTAL_TOKENS

    async def test_structured_call_is_logged_with_split_tokens(self):
        """The LLMCallLog for a structured call carries input and output tokens."""
        client = openai_client()
        client.client.chat.completions.parse = AsyncMock(
            return_value=openai_parsed_completion()
        )
        logger = MagicMock()

        await client.generate("hi", response_format=Answer, logger=logger)

        log = logged_call(logger)
        assert log.success is True
        assert log.input_tokens == INPUT_TOKENS
        assert log.output_tokens == OUTPUT_TOKENS
        assert log.token_count == TOTAL_TOKENS

    async def test_missing_usage_gives_zero(self):
        """A response without usage gives 0 input and 0 output tokens."""
        client = openai_client()
        client.client.chat.completions.create = AsyncMock(
            return_value=openai_text_completion(None)
        )

        _, usage = await client.generate("hi")

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.token_count == 0


# --- Gemini --------------------------------------------------------------------


class TestGeminiTokenSplit:
    """Gemini fills input/output tokens from prompt/candidates token counts."""

    async def test_plain_call_splits_tokens(self):
        """A plain call reports prompt_token_count and candidates_token_count."""
        client = gemini_client()
        client.client.aio.models.generate_content = AsyncMock(
            return_value=gemini_response("hello", gemini_usage())
        )

        _, usage = await client.generate("hi")

        assert usage.input_tokens == INPUT_TOKENS
        assert usage.output_tokens == OUTPUT_TOKENS
        assert usage.token_count == TOTAL_TOKENS

    async def test_structured_call_splits_tokens(self):
        """A structured call reports prompt_token_count and candidates_token_count."""
        client = gemini_client()
        client.client.aio.models.generate_content = AsyncMock(
            return_value=gemini_response('{"answer": "42"}', gemini_usage())
        )

        response, usage = await client.generate("hi", response_format=Answer)

        assert response == Answer(answer="42")
        assert usage.input_tokens == INPUT_TOKENS
        assert usage.output_tokens == OUTPUT_TOKENS
        assert usage.token_count == TOTAL_TOKENS

    async def test_plain_call_is_logged_with_split_tokens(self):
        """The LLMCallLog for a plain call carries input and output tokens."""
        client = gemini_client()
        client.client.aio.models.generate_content = AsyncMock(
            return_value=gemini_response("hello", gemini_usage())
        )
        logger = MagicMock()

        await client.generate("hi", logger=logger)

        log = logged_call(logger)
        assert log.success is True
        assert log.input_tokens == INPUT_TOKENS
        assert log.output_tokens == OUTPUT_TOKENS
        assert log.token_count == TOTAL_TOKENS

    async def test_missing_usage_gives_zero(self):
        """A response without usage_metadata gives 0 input and 0 output tokens."""
        client = gemini_client()
        client.client.aio.models.generate_content = AsyncMock(
            return_value=gemini_response("hello", None)
        )

        _, usage = await client.generate("hi")

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.token_count == 0


# --- Failed call ----------------------------------------------------------------


class TestFailedCallLog:
    """A failed LLM call logs zero input and output tokens."""

    @pytest.mark.parametrize(
        "make_client",
        [anthropic_client, openai_client, gemini_client],
        ids=["anthropic", "openai", "gemini"],
    )
    async def test_failed_call_logs_zero_tokens(self, make_client: Any):
        """The failure LLMCallLog has input_tokens == output_tokens == 0."""
        client = make_client()
        boom = AsyncMock(side_effect=RuntimeError("provider down"))
        client.client.messages.create = boom
        client.client.chat.completions.create = boom
        client.client.aio.models.generate_content = boom
        logger = MagicMock()

        with pytest.raises(RuntimeError):
            await client.generate("hi", logger=logger)

        log = logged_call(logger)
        assert log.success is False
        assert log.input_tokens == 0
        assert log.output_tokens == 0
        assert log.token_count == 0
