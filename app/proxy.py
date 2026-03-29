from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
import time
from typing import Any
from urllib.parse import urljoin

import httpx
from fastapi import HTTPException, Request, Response  # pyright: ignore[reportMissingImports]
from fastapi.responses import StreamingResponse  # pyright: ignore[reportMissingImports]

from app.config_loader import PromptConfig
from app.langfuse_recorder import LangfuseRecorder
from app.prompt_manager import PromptManager
from app.settings import Settings

logger = logging.getLogger("trustopsback")

BLOCKED_HEADERS = {"host", "content-length",
                   "connection", "accept-encoding", "x-gateway-api-key"}
NO_BUFFER_HEADERS = {"content-length",
                     "transfer-encoding", "connection", "content-encoding"}
USER_HEADER_CANDIDATES = ("x-user-id", "x-end-user-id", "x-gateway-user-id")
SESSION_HEADER_CANDIDATES = ("x-session-id", "x-conversation-id",
                             "x-thread-id", "x-chat-id", "x-gateway-session-id")


def build_upstream_url(base_url: str, path: str) -> str:
  return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def forward_headers(request: Request) -> dict[str, str]:
  headers: dict[str, str] = {}
  for key, value in request.headers.items():
    if key.lower() not in BLOCKED_HEADERS:
      headers[key] = value
  return headers


def extract_trace_identity(request: Request, request_json: Any) -> dict[str, str | None]:
  user_id = None
  session_id = None

  for header_name in USER_HEADER_CANDIDATES:
    header_value = request.headers.get(header_name)
    if header_value and header_value.strip():
      user_id = header_value.strip()
      break

  for header_name in SESSION_HEADER_CANDIDATES:
    header_value = request.headers.get(header_name)
    if header_value and header_value.strip():
      session_id = header_value.strip()
      break

  if isinstance(request_json, dict):
    if user_id is None:
      for key in ("user", "user_id", "userId", "langfuse_user_id"):
        value = request_json.get(key)
        if isinstance(value, str) and value.strip():
          user_id = value.strip()
          break
    if session_id is None:
      for key in ("session_id", "sessionId", "conversation_id", "conversationId", "thread_id", "threadId", "chat_id", "chatId", "langfuse_session_id"):
        value = request_json.get(key)
        if isinstance(value, str) and value.strip():
          session_id = value.strip()
          break

  return {"user_id": user_id, "session_id": session_id}


def create_http_client(settings: Settings) -> httpx.AsyncClient:
  return httpx.AsyncClient(
      base_url=settings.vllm_base_url,
      timeout=httpx.Timeout(settings.request_timeout_seconds),
  )


def _apply_generation_config(request_json: dict[str, Any], config: PromptConfig) -> dict[str, Any]:
  updated = dict(request_json)
  if config.temperature is not None:
    updated["temperature"] = config.temperature
  if config.top_p is not None:
    updated["top_p"] = config.top_p
  if config.top_k is not None:
    updated["top_k"] = config.top_k
  return updated


def _apply_system_prompt(request_json: dict[str, Any], system_prompt: str) -> dict[str, Any]:
  updated = dict(request_json)
  updated.pop("system_prompt", None)

  if not system_prompt.strip():
    return updated

  messages = list(updated.get("messages") or [])
  if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
    messages[0] = {**messages[0], "content": system_prompt}
  else:
    messages.insert(0, {"role": "system", "content": system_prompt})

  updated["messages"] = messages
  return updated


def apply_preview_config(request_json: dict[str, Any], config: PromptConfig | None) -> dict[str, Any]:
  updated = dict(request_json)
  request_system_prompt = updated.pop("system_prompt", None)
  system_prompt: str | None = request_system_prompt if isinstance(request_system_prompt, str) else None

  if config is not None:
    if not (isinstance(system_prompt, str) and system_prompt.strip()):
      system_prompt = config.system_prompt

    if updated.get("temperature") is None and config.temperature is not None:
      updated["temperature"] = config.temperature
    if updated.get("top_p") is None and config.top_p is not None:
      updated["top_p"] = config.top_p
    if updated.get("top_k") is None and config.top_k is not None:
      updated["top_k"] = config.top_k

  if isinstance(system_prompt, str) and system_prompt.strip():
    updated = _apply_system_prompt(updated, system_prompt)

  return updated


async def read_request_body(request: Request, settings: Settings) -> bytes:
  body = await request.body()
  if len(body) > settings.max_response_bytes:
    raise HTTPException(status_code=413, detail="Request body too large")
  return body


async def proxy_request(
    *,
    path: str,
    request: Request,
    client: httpx.AsyncClient,
    settings: Settings,
    langfuse: LangfuseRecorder | None,
    prompt_manager: PromptManager | None = None,
    upstream_path: str | None = None,
    request_json_override: dict[str, Any] | None = None,
    body_override: bytes | None = None,
    apply_generation_config: bool = True,
    trace_name: str | None = None,
) -> Response:
  upstream_url = build_upstream_url(settings.vllm_base_url, upstream_path or path)
  body = body_override
  if body is None:
    body = await read_request_body(request, settings)
  params = dict(request.query_params)
  headers = forward_headers(request)
  start = time.perf_counter()

  request_json: Any = request_json_override
  if request_json is None:
    try:
      request_json = json.loads(body.decode("utf-8")) if body else None
    except Exception:
      request_json = None
  trace_identity = extract_trace_identity(request, request_json)

  version_tag: str | None = None
  if (
      prompt_manager is not None
      and isinstance(request_json, dict)
      and isinstance(request_json.get("messages"), list)
  ):
    _roll = random.random()
    _prompt_text, version_tag = prompt_manager.get_prompt(_roll)
    if _prompt_text:
      _msgs = list(request_json["messages"])
      if _msgs and isinstance(_msgs[0], dict) and _msgs[0].get("role") == "system":
        _msgs[0] = {**_msgs[0], "content": _prompt_text}
      else:
        _msgs.insert(0, {"role": "system", "content": _prompt_text})
      request_json = {**request_json, "messages": _msgs}
      try:
        body = json.dumps(request_json).encode("utf-8")
        if "content-type" not in {k.lower() for k in headers}:
          headers["content-type"] = "application/json"
      except Exception:
        pass

  prompt_config_loader = getattr(request.app.state, "prompt_config_loader", None)
  prompt_config = prompt_config_loader.get_config() if prompt_config_loader is not None else None
  if apply_generation_config and path == "/v1/chat/completions" and isinstance(request_json, dict) and prompt_config is not None:
    request_json = _apply_generation_config(request_json, prompt_config)
    try:
      body = json.dumps(request_json).encode("utf-8")
      if "content-type" not in {k.lower() for k in headers}:
        headers["content-type"] = "application/json"
    except Exception:
      pass

  is_streaming_request = bool(isinstance(
    request_json, dict) and request_json.get("stream"))
  if is_streaming_request:
    headers["accept"] = "text/event-stream"

  # Inject stream_options.include_usage so vLLM appends a usage chunk at the end
  # of every streaming response. Without this, token counts are never available.
  upstream_body: bytes = body
  if is_streaming_request and isinstance(request_json, dict) and langfuse is not None:
    patched = dict(request_json)
    stream_options = dict(patched.get("stream_options") or {})
    stream_options["include_usage"] = True
    patched["stream_options"] = stream_options
    try:
      upstream_body = json.dumps(patched).encode("utf-8")
      if "content-type" not in {k.lower() for k in headers}:
        headers["content-type"] = "application/json"
    except Exception:
      upstream_body = body

  request_timeout = (
      httpx.Timeout(None, connect=settings.request_timeout_seconds,
                    write=settings.request_timeout_seconds, pool=settings.request_timeout_seconds)
      if is_streaming_request
      else httpx.Timeout(settings.request_timeout_seconds)
  )

  try:
    upstream_stream = client.stream(
        request.method,
        upstream_url,
        params=params,
        content=upstream_body if upstream_body else None,
        headers=headers,
        timeout=request_timeout,
    )
    upstream_response = await upstream_stream.__aenter__()
  except httpx.HTTPError as exc:
    logger.exception(
      "Upstream vLLM request failed before response headers: %s", exc)
    raise HTTPException(
        status_code=502,
        detail=f"Upstream vLLM request failed before response headers: {exc.__class__.__name__}: {exc}",
    ) from exc

  response_headers = {
      key: value
      for key, value in upstream_response.headers.items()
      if key.lower() not in NO_BUFFER_HEADERS
  }
  content_type = upstream_response.headers.get("content-type", "")
  is_streaming = "text/event-stream" in content_type.lower() or bool(
    request_json and request_json.get("stream"))

  if is_streaming:
    response_buffer = bytearray()

    async def stream_body():
      exc_info = (None, None, None)
      try:
        async for chunk in upstream_response.aiter_raw():
          response_buffer.extend(chunk)
          yield chunk
      except httpx.HTTPError as exc:
        logger.exception(
          "Upstream vLLM stream failed after response started: %s", exc)
        exc_info = sys.exc_info()
      except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected or generator was closed; capture for clean __aexit__
        exc_info = sys.exc_info()
        raise
      except BaseException:
        exc_info = sys.exc_info()
        raise
      finally:
        duration_ms = (time.perf_counter() - start) * 1000
        _langfuse = langfuse
        if path.startswith("/v1/") and isinstance(request_json, dict) and _langfuse is not None and response_buffer:
          response_text = response_buffer.decode("utf-8", errors="replace")
          _recorder = _langfuse
          loop = asyncio.get_event_loop()
          await loop.run_in_executor(
              None,
              lambda: _recorder.record_stream(
                  path=path,
                  method=request.method,
                  request_payload=request_json,
                  response_text=response_text,
                  status_code=upstream_response.status_code,
                  duration_ms=duration_ms,
                  start_time_perf=start,
                  user_id=trace_identity["user_id"],
                  session_id=trace_identity["session_id"],
                  prompt_version=version_tag,
                  trace_name=trace_name,
              ),
          )
        # Pass real exc_info so httpx can properly clean up the connection
        await upstream_stream.__aexit__(*exc_info)

    response_headers.update(
      {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return StreamingResponse(
        stream_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type="text/event-stream",
    )

  response_error: BaseException | None = None
  response_body = b""
  try:
    response_body = await upstream_response.aread()
  except BaseException as exc:
    response_error = exc
  finally:
    await upstream_stream.__aexit__(None, None, None)

  duration_ms = (time.perf_counter() - start) * 1000

  response_json: Any = None
  if response_body:
    try:
      response_json = json.loads(response_body.decode("utf-8"))
    except Exception:
      response_json = None

  if len(response_body) > settings.max_response_bytes:
    response_error = HTTPException(
      status_code=502, detail="Upstream response too large")

  if path.startswith("/v1/") and isinstance(request_json, dict) and langfuse is not None:
    langfuse.record(
        path=path,
        method=request.method,
        request_payload=request_json,
        response_payload=response_json,
        status_code=upstream_response.status_code,
        duration_ms=duration_ms,
        start_time_perf=start,
        user_id=trace_identity["user_id"],
        session_id=trace_identity["session_id"],
        prompt_version=version_tag,
        trace_name=trace_name,
    )

  if response_error is not None:
    raise response_error

  return Response(
      content=response_body,
      status_code=upstream_response.status_code,
      headers=response_headers,
      media_type=content_type or None,
  )
