from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx

from app.config_loader import PromptConfigLoader, load_prompt_config
from app.settings import Settings


def _make_settings(prompt_config_path: str) -> Settings:
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
        prompt_config_path=prompt_config_path,
    )


def _make_upstream_client(captured: dict[str, Any]) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = httpx.Headers({"content-type": "application/json"})
    mock_resp.aread = AsyncMock(return_value=b'{"choices":[]}')

    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    def _stream(method, url, *, content=None, **kwargs):
        if content is not None:
            captured["body"] = json.loads(content.decode("utf-8"))
        return stream_cm

    client = MagicMock(spec=httpx.AsyncClient)
    client.stream = MagicMock(side_effect=_stream)
    return client


def _make_test_app(http_client: Any, settings: Settings, prompt_config_loader: PromptConfigLoader):
    from fastapi import FastAPI  # pyright: ignore[reportMissingImports]
    from app.routes import router

    app = FastAPI()
    app.state.settings = settings
    app.state.http_client = http_client
    app.state.langfuse = None
    app.state.prompt_config_loader = prompt_config_loader
    app.include_router(router)
    return app


def test_load_prompt_config_reads_configmap(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-config.yaml"
    config_path.write_text(
        """
apiVersion: v1
kind: ConfigMap
data:
  system_prompt: "System prompt"
  temperature: "0.25"
  top_p: "0.9"
  top_k: "40"
  prompt_version: "v1"
""".strip(),
        encoding="utf-8",
    )

    config = load_prompt_config(str(config_path))

    assert config.system_prompt == "System prompt"
    assert config.temperature == 0.25
    assert config.top_p == 0.9
    assert config.top_k == 40
    assert config.prompt_version == "v1"


def test_load_prompt_config_with_only_system_prompt(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-config.yaml"
    config_path.write_text(
        """
data:
  system_prompt: "Minimal config"
""".strip(),
        encoding="utf-8",
    )

    config = load_prompt_config(str(config_path))

    assert config.system_prompt == "Minimal config"
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.prompt_version == ""


def test_prompt_config_loader_keeps_startup_config(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-config.yaml"
    config_path.write_text(
        """
data:
  temperature: "0.1"
  top_p: "0.7"
  top_k: "8"
  prompt_version: "v1"
""".strip(),
        encoding="utf-8",
    )

    loader = PromptConfigLoader(str(config_path))
    try:
        assert loader.get_config().temperature == 0.1

        config_path.write_text(
            """
data:
  temperature: "0.8"
  top_p: "0.2"
  top_k: "12"
  prompt_version: "v2"
""".strip(),
            encoding="utf-8",
        )

        config = loader.get_config()
        assert config.temperature == 0.1
        assert config.top_p == 0.7
        assert config.top_k == 8
        assert config.prompt_version == "v1"
    finally:
        loader.stop()


async def test_chat_completions_overrides_client_generation_params(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-config.yaml"
    config_path.write_text(
        """
data:
  system_prompt: "Config system prompt"
  temperature: "0.33"
  top_p: "0.77"
  top_k: "11"
  prompt_version: "v1"
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}
    prompt_config_loader = PromptConfigLoader(str(config_path))
    try:
        app = _make_test_app(
            _make_upstream_client(captured),
            _make_settings(str(config_path)),
            prompt_config_loader,
        )
        body = json.dumps(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.99,
                "top_p": 0.01,
                "top_k": 999,
            }
        ).encode()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )

        assert captured["body"]["temperature"] == 0.33
        assert captured["body"]["top_p"] == 0.77
        assert captured["body"]["top_k"] == 11
        assert captured["body"]["messages"][0]["role"] == "system"
        assert captured["body"]["messages"][0]["content"] == "Config system prompt"
    finally:
        prompt_config_loader.stop()


def test_prompt_version_parsing(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-config.yaml"
    config_path.write_text(
        """
data:
  system_prompt: "Test prompt"
  prompt_version: "production-v2"
""".strip(),
        encoding="utf-8",
    )

    config = load_prompt_config(str(config_path))
    assert config.prompt_version == "production-v2"


def test_empty_config_returns_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt-config.yaml"
    config_path.write_text(
        """
data: {}
""".strip(),
        encoding="utf-8",
    )

    config = load_prompt_config(str(config_path))
    assert config.system_prompt == ""
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.prompt_version == ""
