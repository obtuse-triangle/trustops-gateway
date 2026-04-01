from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx

from app.config_loader import PromptConfigLoader
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


def _make_config_loader(tmp_path: Path, *, system_prompt: str = "Config system prompt", temperature: float | None = 0.2, top_p: float | None = 0.9, top_k: int | None = 40) -> PromptConfigLoader:
    config_path = tmp_path / "prompt-config.yaml"
    lines = [f'system_prompt: "{system_prompt}"']
    if temperature is not None:
        lines.append(f"temperature: {temperature}")
    if top_p is not None:
        lines.append(f"top_p: {top_p}")
    if top_k is not None:
        lines.append(f"top_k: {top_k}")
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return PromptConfigLoader(str(config_path))


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
        captured["method"] = method
        captured["url"] = url
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

    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    def _stream(method, url, *, content=None, **kwargs):
        if content is not None:
            try:
                captured["body"] = json.loads(content.decode("utf-8"))
            except Exception:
                captured["body"] = content
        captured["method"] = method
        captured["url"] = url
        mock_resp.aiter_raw = MagicMock(return_value=_aiter_raw())
        return stream_cm

    client = MagicMock(spec=httpx.AsyncClient)
    client.stream = MagicMock(side_effect=_stream)
    return client


def _make_test_app(http_client: Any, *, settings: Settings | None = None, langfuse: Any = None, prompt_config_loader: PromptConfigLoader | None = None):
    from fastapi import FastAPI  # pyright: ignore[reportMissingImports]
    from app.routes import router

    app = FastAPI()
    app.state.settings = settings or _make_settings()
    app.state.http_client = http_client
    app.state.langfuse = langfuse
    app.state.prompt_config_loader = prompt_config_loader
    app.include_router(router)
    return app


def _make_loader_from_config(tmp_path: Path, *, system_prompt: str = "Config system prompt", temperature: float | None = 0.2, top_p: float | None = 0.9, top_k: int | None = 40) -> PromptConfigLoader:
    return _make_config_loader(tmp_path, system_prompt=system_prompt, temperature=temperature, top_p=top_p, top_k=top_k)


async def test_preview_uses_ui_overrides(tmp_path: Path) -> None:
    captured: dict = {}
    loader = _make_loader_from_config(tmp_path)
    try:
        app = _make_test_app(_make_upstream_client(captured), prompt_config_loader=loader)
        body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "system_prompt": "UI system prompt",
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 11,
        }).encode()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/preview", content=body, headers={"content-type": "application/json"})
        assert "system_prompt" not in captured["body"]
        assert captured["body"]["temperature"] == 0.7
        assert captured["body"]["top_p"] == 0.8
        assert captured["body"]["top_k"] == 11
        assert captured["body"]["messages"][0]["content"] == "Config system prompt"
    finally:
        loader.stop()


async def test_preview_falls_back_to_config(tmp_path: Path) -> None:
    captured: dict = {}
    loader = _make_loader_from_config(tmp_path)
    try:
        app = _make_test_app(_make_upstream_client(captured), prompt_config_loader=loader)
        body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/preview", content=body, headers={"content-type": "application/json"})
        assert captured["body"]["temperature"] == 0.2
        assert captured["body"]["top_p"] == 0.9
        assert captured["body"]["top_k"] == 40
        assert captured["body"]["messages"][0]["content"] == "Config system prompt"
    finally:
        loader.stop()


async def test_preview_streams_sse(tmp_path: Path) -> None:
    captured: dict = {}
    loader = _make_loader_from_config(tmp_path)
    chunks = [
        b'data: {"id":"1","choices":[{"delta":{"role":"assistant","content":"hi"},"finish_reason":null}] }\n\n',
        b'data: [DONE]\n\n',
    ]
    try:
        app = _make_test_app(_make_streaming_client(captured, chunks), prompt_config_loader=loader)
        body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }).encode()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/preview", content=body, headers={"content-type": "application/json"})
            text = await response.aread()
        assert response.status_code == 200
        assert b"data:" in text
        assert captured["url"].endswith("/v1/chat/completions")
    finally:
        loader.stop()


async def test_preview_langfuse_trace_name(tmp_path: Path) -> None:
    captured: dict = {}
    loader = _make_loader_from_config(tmp_path)
    mock_langfuse = MagicMock()
    mock_langfuse.record = MagicMock()
    mock_langfuse.record_stream = MagicMock()
    try:
        app = _make_test_app(_make_upstream_client(captured), langfuse=mock_langfuse, prompt_config_loader=loader)
        body = json.dumps({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/preview", content=body, headers={"content-type": "application/json"})
        _, kwargs = mock_langfuse.record.call_args
        assert kwargs.get("trace_name") == "playground-preview"
    finally:
        loader.stop()
