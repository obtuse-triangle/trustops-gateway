from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.langfuse_recorder import LangfuseRecorder
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
