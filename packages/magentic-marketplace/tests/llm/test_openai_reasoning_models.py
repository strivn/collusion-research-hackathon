"""Which arguments the OpenAI client sends to reasoning and non-reasoning models.

gpt-6 models (GPT-6 Luna) are reasoning models: they get `max_completion_tokens`, no
`temperature`, and `reasoning_effort` as given, `none` included
(https://developers.openai.com/api/docs/models/gpt-6-luna lists none, low, medium,
high, xhigh, max). The SDK client is a mock; no API is called.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from magentic_marketplace.experiments.run_config import ModelConfig
from magentic_marketplace.marketplace.llm.clients.openai import (
    OpenAIClient,
    OpenAIConfig,
)
from magentic_marketplace.marketplace.llm.functional import ConcreteConfigAdapter


def text_completion() -> ChatCompletion:
    """Build a Chat Completions text response."""
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=0,
        model="test",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="hello"),
            )
        ],
    )


async def sent_arguments(model: str, **generate_kwargs: Any) -> dict[str, Any]:
    """Make one plain call to `model` and return the arguments sent to the SDK."""
    client = OpenAIClient(OpenAIConfig(provider="openai", api_key="test-key"))
    client.client = MagicMock()
    create = AsyncMock(return_value=text_completion())
    client.client.chat.completions.create = create

    await client.generate("hi", model=model, **generate_kwargs)

    return dict(create.call_args.kwargs)


async def test_gpt_6_luna_gets_the_reasoning_arguments():
    """gpt-6-luna: max_completion_tokens, no temperature, reasoning_effort none as given."""
    sent = await sent_arguments(
        "gpt-6-luna", reasoning_effort="none", max_tokens=2000, temperature=0.5
    )

    assert sent["reasoning_effort"] == "none"
    assert sent["max_completion_tokens"] == 2000
    assert "max_tokens" not in sent
    assert "temperature" not in sent


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
async def test_gpt_6_luna_effort_is_sent_as_given(effort):
    """gpt-6-luna: every effort it accepts is sent unchanged."""
    sent = await sent_arguments("gpt-6-luna", reasoning_effort=effort)

    assert sent["reasoning_effort"] == effort


async def test_gpt_5_mini_minimal_is_unchanged():
    """gpt-5-mini: minimal is still sent as minimal."""
    sent = await sent_arguments(
        "gpt-5-mini", reasoning_effort="minimal", max_tokens=2000
    )

    assert sent["reasoning_effort"] == "minimal"
    assert sent["max_completion_tokens"] == 2000


async def test_non_reasoning_model_gets_max_tokens_and_temperature():
    """gpt-4.1-mini: max_tokens and temperature, no reasoning_effort."""
    sent = await sent_arguments(
        "gpt-4.1-mini", reasoning_effort="none", max_tokens=2000, temperature=0.5
    )

    assert sent["max_tokens"] == 2000
    assert sent["temperature"] == 0.5
    assert "reasoning_effort" not in sent
    assert "max_completion_tokens" not in sent


def test_run_config_and_llm_config_accept_effort_none():
    """reasoning_effort none passes the run config's llm block and the client config."""
    model = ModelConfig(provider="openai", model="gpt-6-luna", reasoning_effort="none")
    config = ConcreteConfigAdapter.validate_python(
        {**model.llm_kwargs(), "api_key": "test-key"}
    )

    assert config.reasoning_effort == "none"
