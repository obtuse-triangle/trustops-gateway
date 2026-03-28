from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.settings import Settings
from app.langfuse_recorder import LangfuseRecorder


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        vllm_base_url="http://test-vllm:8000",
        gateway_api_key="test-gateway-key",
        langfuse_public_key="test-public-key",
        langfuse_secret_key="test-secret-key",
        langfuse_host="http://test-langfuse:3000",
        langfuse_enabled=False,
        request_timeout_seconds=30.0,
        max_response_bytes=20 * 1024 * 1024,
        log_level="DEBUG",
    )


@pytest.fixture
def mock_http_client() -> AsyncMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.fixture
def mock_langfuse(mock_settings: Settings) -> MagicMock:
    recorder = MagicMock(spec=LangfuseRecorder)
    recorder.settings = mock_settings
    recorder.client = None
    recorder.record = MagicMock()
    recorder.record_stream = MagicMock()
    return recorder


@pytest.fixture
def sample_request_payload() -> dict[str, Any]:
    return {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False,
    }
