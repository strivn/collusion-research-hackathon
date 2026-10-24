"""Per-role models and the request-rate cap from the run config's `llm` block."""

import asyncio
import time

import pytest

from magentic_marketplace.experiments.run_config import (
    LLMConfig,
    ModelConfig,
    RunConfig,
    role_kwargs,
)
from magentic_marketplace.marketplace.llm import base as llm_base


def test_unset_llm_block_keeps_env_llm():
    """A config without `llm` sets no role and no cap (stock)."""
    config = RunConfig(data="d", experiment_name="e")
    assert config.llm == LLMConfig()
    assert role_kwargs(config.llm.seller_strategy) == {}


def test_model_config_reads_key_from_named_env_var(monkeypatch):
    """The key comes from the variable `api_key_env` names, never from the config."""
    monkeypatch.setenv("TEST_ROLE_KEY", "secret")
    model = ModelConfig(
        provider="openai",
        model="google/gemma-4-26b-a4b-it:free",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="TEST_ROLE_KEY",
    )
    assert model.llm_kwargs() == {
        "provider": "openai",
        "model": "google/gemma-4-26b-a4b-it:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "secret",
    }


def test_custom_endpoint_without_key_name_is_rejected():
    """A base_url without api_key_env is refused, so the default key never leaves for it."""
    with pytest.raises(ValueError):
        ModelConfig(provider="openai", model="m", base_url="https://example.invalid")


def test_unknown_llm_key_is_rejected():
    """A misspelt role is refused, like every unknown setting."""
    with pytest.raises(ValueError):
        RunConfig.model_validate(
            {"data": "d", "experiment_name": "e", "llm": {"sellers": {}}}
        )


def test_request_rate_spaces_call_starts():
    """With 600 requests per minute, 4 calls start at least 0.3 s apart in total."""

    async def start_four() -> float:
        started = time.monotonic()
        await asyncio.gather(*(llm_base._request_rate.wait() for _ in range(4)))
        return time.monotonic() - started

    llm_base.set_requests_per_minute(600)
    try:
        assert asyncio.run(start_four()) >= 0.29
    finally:
        llm_base.set_requests_per_minute(None)


def test_no_cap_never_waits():
    """Without a cap a call starts at once."""
    llm_base.set_requests_per_minute(None)
    started = time.monotonic()
    asyncio.run(llm_base._request_rate.wait())
    assert time.monotonic() - started < 0.05
