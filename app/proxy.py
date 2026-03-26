from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.parse import urljoin

import httpx
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.langfuse_recorder import LangfuseRecorder
from app.settings import Settings

logger = logging.getLogger("trustopsback")

BLOCKED_HEADERS = {"host", "content-length", "connection", "accept-encoding", "x-gateway-api-key"}
NO_BUFFER_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding"}
USER_HEADER_CANDIDATES = ("x-user-id", "x-end-user-id", "x-gateway-user-id")
SESSION_HEADER_CANDIDATES = ("x-session-id", "x-conversation-id", "x-thread-id", "x-chat-id", "x-gateway-session-id")



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
) -> Response:
    upstream_url = build_upstream_url(settings.vllm_base_url, path)
    body = await read_request_body(request, settings)
    params = dict(request.query_params)
    headers = forward_headers(request)
    start = time.perf_counter()

    request_json: Any = None
    try:
        request_json = json.loads(body.decode("utf-8")) if body else None
    except Exception:
        request_json = None
    trace_identity = extract_trace_identity(request, request_json)

    is_streaming_request = bool(isinstance(request_json, dict) and request_json.get("stream"))
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
        httpx.Timeout(None, connect=settings.request_timeout_seconds, write=settings.request_timeout_seconds, pool=settings.request_timeout_seconds)
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
        logger.exception("Upstream vLLM request failed before response headers: %s", exc)
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
    is_streaming = "text/event-stream" in content_type.lower() or bool(request_json and request_json.get("stream"))

    if is_streaming:
        response_buffer = bytearray()

        async def stream_body():
            response_error: BaseException | None = None
            try:
                async for chunk in upstream_response.aiter_raw():
                    response_buffer.extend(chunk)
                    yield chunk
            except httpx.HTTPError as exc:
                logger.exception("Upstream vLLM stream failed after response started: %s", exc)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                response_text = response_buffer.decode("utf-8", errors="replace")
                if path.startswith("/v1/") and isinstance(request_json, dict) and langfuse is not None:
                    langfuse.record_stream(
                        path=path,
                        method=request.method,
                        request_payload=request_json,
                        response_text=response_text,
                        status_code=upstream_response.status_code,
                        duration_ms=duration_ms,
                        start_time_perf=start,
                        user_id=trace_identity["user_id"],
                        session_id=trace_identity["session_id"],
                    )
                await upstream_stream.__aexit__(None, None, None)

        response_headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
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
        response_error = HTTPException(status_code=502, detail="Upstream response too large")

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
        )

    if response_error is not None:
        raise response_error

    return Response(
        content=response_body,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=content_type or None,
    )
