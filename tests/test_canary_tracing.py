from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request

from app.langfuse_recorder import LangfuseRecorder
from app.proxy import forward_headers, proxy_request
from app.settings import Settings


def _make_settings() -> Settings:
    return Settings(
        vllm_base_url="http://test-vllm:8000",
        gateway_api_key="",
        langfuse_public_key="",
        langfuse_secret_key="",
        langfuse_host="",
        langfuse_enabled=False,
        request_timeout_seconds=30.0,
        max_response_bytes=20 * 1024 * 1024,
        log_level="DEBUG",
        prompt_config_path="/nonexistent/prompt-config.yaml",
    )


def _make_recorder() -> LangfuseRecorder:
    return LangfuseRecorder(_make_settings())


def _recorder_with_mock_client() -> LangfuseRecorder:
    recorder = _make_recorder()
    mock_obs = MagicMock()
    mock_obs.end = MagicMock()
    mock_obs.update = MagicMock()
    recorder.client = MagicMock()
    recorder.client.start_observation = MagicMock(return_value=mock_obs)
    recorder.client.flush = MagicMock()
    return recorder


def test_trace_version_non_streaming() -> None:
    recorder = _make_recorder()
    meta = recorder._trace_metadata(
        path="/v1/chat/completions",
        method="POST",
        request_payload={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        status_code=200,
        duration_ms=123.45,
        stream=False,
        prompt_version="v1",
    )
    assert meta["prompt_version"] == "v1"
    assert meta["stream"] == "false"
    assert meta["method"] == "POST"
    assert meta["path"] == "/v1/chat/completions"


def test_trace_version_streaming() -> None:
    recorder = _make_recorder()
    meta = recorder._trace_metadata(
        path="/v1/chat/completions",
        method="POST",
        request_payload={"model": "test-model", "messages": []},
        status_code=200,
        duration_ms=200.0,
        stream=True,
        prompt_version="v2",
    )
    assert meta["prompt_version"] == "v2"
    assert meta["stream"] == "true"


def test_backward_compat() -> None:
    recorder = _make_recorder()
    meta = recorder._trace_metadata(
        path="/v1/chat/completions",
        method="POST",
        request_payload={"model": "test-model", "messages": []},
        status_code=200,
        duration_ms=50.0,
        stream=False,
    )
    assert "prompt_version" not in meta
    assert meta["path"] == "/v1/chat/completions"
    assert meta["method"] == "POST"
    assert meta["stream"] == "false"


def test_version_in_tags() -> None:
    recorder = _recorder_with_mock_client()

    with patch("app.langfuse_recorder.propagate_attributes") as mock_propagate:
        recorder.record(
            path="/v1/chat/completions",
            method="POST",
            request_payload={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
            response_payload={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
            status_code=200,
            duration_ms=80.0,
            prompt_version="v1",
        )

    assert mock_propagate.called
    _, kwargs = mock_propagate.call_args
    tags: list[str] | None = kwargs.get("tags")
    assert tags is not None
    assert "prompt_version:v1" in tags


def test_no_tags_when_no_version() -> None:
    recorder = _recorder_with_mock_client()

    with patch("app.langfuse_recorder.propagate_attributes") as mock_propagate:
        recorder.record(
            path="/v1/chat/completions",
            method="POST",
            request_payload={"model": "test-model", "messages": []},
            response_payload={},
            status_code=200,
            duration_ms=40.0,
        )

    assert mock_propagate.called
    _, kwargs = mock_propagate.call_args
    assert kwargs.get("tags") is None


# ---------------------------------------------------------------------------
# X-Skip-Langfuse header suppression
# ---------------------------------------------------------------------------


def test_x_skip_langfuse_not_forwarded_upstream() -> None:
    """X-Skip-Langfuse header must be filtered by BLOCKED_HEADERS."""
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        "host": "example.com",
        "x-skip-langfuse": "true",
        "authorization": "Bearer test-key",
        "content-type": "application/json",
    }
    headers = forward_headers(mock_request)
    assert "x-skip-langfuse" not in headers
    assert "host" not in headers
    assert headers.get("authorization") == "Bearer test-key"
    assert headers.get("content-type") == "application/json"


async def test_active_fetch_skips_backend_recording(
    mock_settings: Settings,
    mock_langfuse: MagicMock,
) -> None:
    """Non-streaming request with X-Skip-Langfuse: true skips record()."""
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"x-skip-langfuse": "true", "content-type": "application/json"}
    mock_request.method = "POST"
    mock_request.query_params = {}
    mock_request.app.state.prompt_config_loader = None

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.aread = AsyncMock(return_value=b'{"choices": []}')

    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=mock_response)
    async_cm.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=async_cm)

    with patch("app.proxy.read_request_body", AsyncMock(return_value=b'{"model": "test", "messages": []}')):
        response = await proxy_request(
            path="/v1/chat/completions",
            request=mock_request,
            client=mock_client,
            settings=mock_settings,
            langfuse=mock_langfuse,
        )

    mock_langfuse.record.assert_not_called()
    assert response.status_code == 200


async def test_backend_records_normally_without_skip_header(
    mock_settings: Settings,
    mock_langfuse: MagicMock,
) -> None:
    """Non-streaming request without skip header calls record()."""
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"content-type": "application/json"}
    mock_request.method = "POST"
    mock_request.query_params = {}
    mock_request.app.state.prompt_config_loader = None

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.aread = AsyncMock(return_value=b'{"choices": []}')

    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=mock_response)
    async_cm.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=async_cm)

    with patch("app.proxy.read_request_body", AsyncMock(return_value=b'{"model": "test", "messages": []}')):
        response = await proxy_request(
            path="/v1/chat/completions",
            request=mock_request,
            client=mock_client,
            settings=mock_settings,
            langfuse=mock_langfuse,
        )

    mock_langfuse.record.assert_called_once()
    assert response.status_code == 200


@pytest.mark.parametrize("header_value", ["TRUE", "True", "true"])
def test_x_skip_langfuse_true_case_insensitive(header_value: str) -> None:
    """Header value matching is case-insensitive for 'true'."""
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        "x-skip-langfuse": header_value,
        "authorization": "Bearer test",
    }
    headers = forward_headers(mock_request)
    assert "x-skip-langfuse" not in headers

    computed = mock_request.headers.get("x-skip-langfuse", "").strip().lower() == "true"
    assert computed is True


async def test_x_skip_langfuse_streaming_suppresses_record_stream_and_usage_injection(
    mock_settings: Settings,
    mock_langfuse: MagicMock,
) -> None:
    """Streaming request with skip header skips record_stream and usage injection."""
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"x-skip-langfuse": "true", "content-type": "application/json"}
    mock_request.method = "POST"
    mock_request.query_params = {}
    mock_request.app.state.prompt_config_loader = None

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}

    async def _raw_iter():
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    mock_response.aiter_raw = _raw_iter

    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=mock_response)
    async_cm.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=async_cm)

    with patch("app.proxy.read_request_body", AsyncMock(return_value=b'{"model": "test", "messages": [], "stream": true}')):
        response = await proxy_request(
            path="/v1/chat/completions",
            request=mock_request,
            client=mock_client,
            settings=mock_settings,
            langfuse=mock_langfuse,
        )

    async for _ in response.body_iterator:
        pass

    mock_langfuse.record_stream.assert_not_called()
    call_kwargs = mock_client.stream.call_args[1]
    call_content = call_kwargs.get("content", b"")
    call_body: dict = json.loads(call_content.decode("utf-8")) if call_content else {}
    assert "stream_options" not in call_body or call_body.get("stream_options", {}).get("include_usage") is not True
