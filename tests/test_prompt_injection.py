from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx

from app.prompt_manager import PromptManager
from app.settings import Settings


PROMPT_TEXT = "You are a test assistant."


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
        prompts_dir="/nonexistent",
        canary_weight_env="CANARY_WEIGHT",
    )


def _make_prompt_manager(tmp_path: Path, text: str = PROMPT_TEXT) -> PromptManager:
    d = tmp_path / "prompts"
    d.mkdir(exist_ok=True)
    (d / "prompt_v1.txt").write_text(text, encoding="utf-8")
    return PromptManager(str(d))


def _make_upstream_client(captured: dict, status: int = 200, body: bytes = b'{"choices":[]}') -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.headers = httpx.Headers({"content-type": "application/json"})
    mock_resp.aread = AsyncMock(return_value=body)

    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    def _stream(method, url, *, content=None, **kwargs):
        if content is not None:
            try:
                captured["body"] = json.loads(content.decode("utf-8"))
            except Exception:
                captured["body"] = content
        return stream_cm

    client = MagicMock(spec=httpx.AsyncClient)
    client.stream = MagicMock(side_effect=_stream)
    return client


def _make_streaming_client(captured: dict, chunks: list[bytes]) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = httpx.Headers({"content-type": "text/event-stream"})

    async def _aiter_raw():
        for chunk in chunks:
            yield chunk

    def _stream(method, url, *, content=None, **kwargs):
        if content is not None:
            try:
                captured["body"] = json.loads(content.decode("utf-8"))
            except Exception:
                captured["body"] = content
        mock_resp.aiter_raw = MagicMock(return_value=_aiter_raw())
        return stream_cm

    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    client = MagicMock(spec=httpx.AsyncClient)
    client.stream = MagicMock(side_effect=_stream)
    return client


def _make_test_app(
    http_client: Any,
    settings: Settings | None = None,
    langfuse: Any = None,
    prompt_manager: PromptManager | None = None,
):
    from fastapi import FastAPI
    from app.routes import router

    app = FastAPI()
    app.state.settings = settings or _make_settings()
    app.state.http_client = http_client
    app.state.langfuse = langfuse
    app.state.prompt_manager = prompt_manager
    app.include_router(router)
    return app


async def test_prepend_system_prompt(tmp_path: Path) -> None:
    captured: dict = {}
    pm = _make_prompt_manager(tmp_path)
    try:
        app = _make_test_app(_make_upstream_client(captured), prompt_manager=pm)
        body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
        msgs = captured["body"]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == PROMPT_TEXT
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello"
    finally:
        pm.stop()


async def test_replace_system_prompt(tmp_path: Path) -> None:
    captured: dict = {}
    pm = _make_prompt_manager(tmp_path)
    try:
        app = _make_test_app(_make_upstream_client(captured), prompt_manager=pm)
        body = json.dumps({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "Old system message"},
                {"role": "user", "content": "Hello"},
            ],
        }).encode()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
        msgs = captured["body"]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == PROMPT_TEXT
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello"
    finally:
        pm.stop()


async def test_passthrough_non_chat(tmp_path: Path) -> None:
    captured: dict = {}
    pm = _make_prompt_manager(tmp_path)
    try:
        app = _make_test_app(_make_upstream_client(captured), prompt_manager=pm)
        body = json.dumps({
            "model": "test-model",
            "prompt": "Tell me a story",
        }).encode()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
        assert "messages" not in captured["body"]
        assert captured["body"]["prompt"] == "Tell me a story"
    finally:
        pm.stop()


async def test_langfuse_prompt_version_non_streaming(tmp_path: Path) -> None:
    captured: dict = {}
    pm = _make_prompt_manager(tmp_path)
    mock_langfuse = MagicMock()
    mock_langfuse.record = MagicMock()
    mock_langfuse.record_stream = MagicMock()
    try:
        app = _make_test_app(
            _make_upstream_client(captured),
            langfuse=mock_langfuse,
            prompt_manager=pm,
        )
        body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
        assert mock_langfuse.record.called
        _, kwargs = mock_langfuse.record.call_args
        assert kwargs.get("prompt_version") == "v1"
    finally:
        pm.stop()


async def test_langfuse_prompt_version_streaming(tmp_path: Path) -> None:
    captured: dict = {}
    pm = _make_prompt_manager(tmp_path)
    mock_langfuse = MagicMock()
    mock_langfuse.record = MagicMock()
    mock_langfuse.record_stream = MagicMock()
    sse_chunks = [
        b'data: {"id":"1","choices":[{"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}\n\n',
        b'data: {"id":"1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}\n\n',
        b"data: [DONE]\n\n",
    ]
    try:
        app = _make_test_app(
            _make_streaming_client(captured, sse_chunks),
            langfuse=mock_langfuse,
            prompt_manager=pm,
        )
        body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }).encode()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
            await response.aread()
        assert mock_langfuse.record_stream.called
        _, kwargs = mock_langfuse.record_stream.call_args
        assert kwargs.get("prompt_version") == "v1"
    finally:
        pm.stop()


async def test_backward_compat(tmp_path: Path) -> None:
    captured: dict = {}
    mock_langfuse = MagicMock()
    mock_langfuse.record = MagicMock()
    mock_langfuse.record_stream = MagicMock()
    app = _make_test_app(
        _make_upstream_client(captured),
        langfuse=mock_langfuse,
        prompt_manager=None,
    )
    body = json.dumps({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }).encode()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/json"},
        )
    assert mock_langfuse.record.called
    _, kwargs = mock_langfuse.record.call_args
    assert kwargs.get("prompt_version") is None
    msgs = captured["body"]["messages"]
    assert msgs[0]["role"] == "user"
