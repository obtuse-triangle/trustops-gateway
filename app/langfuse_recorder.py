from __future__ import annotations

import json
import logging
import re
import time as _time
from collections.abc import Mapping
from typing import Any

from langfuse import Langfuse, propagate_attributes

from app.settings import Settings

logger = logging.getLogger("trustopsback")

DATA_URI_PATTERN = re.compile(r"data:[^\s\"']+;base64,[A-Za-z0-9+/=]+")
MAX_TRACE_TEXT_LENGTH = 12000
TRACE_REQUEST_USER_KEYS = (
  "user", "user_id", "userId", "langfuse_user_id", "langfuse_userId")
TRACE_REQUEST_SESSION_KEYS = (
    "session_id",
    "sessionId",
    "conversation_id",
    "conversationId",
    "thread_id",
    "threadId",
    "chat_id",
    "chatId",
    "langfuse_session_id",
    "langfuse_sessionId",
)


class LangfuseRecorder:
  def __init__(self, settings: Settings) -> None:
    self.settings = settings
    self.client = None
    if not settings.langfuse_enabled:
      logger.info("Langfuse is disabled")
      return
    if not (settings.langfuse_public_key and settings.langfuse_secret_key and settings.langfuse_host):
      logger.info("Langfuse env vars are incomplete; skipping tracing")
      return
    self.client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    logger.info("Langfuse initialized for host %s", settings.langfuse_host)

  def _sanitize_text(self, text: str) -> str:
    text = DATA_URI_PATTERN.sub("[redacted-data-uri]", text)
    if len(text) > MAX_TRACE_TEXT_LENGTH:
      text = text[:MAX_TRACE_TEXT_LENGTH] + "…[truncated]"
    return text

  def _sanitize_payload(self, payload: Any) -> Any:
    if payload is None or isinstance(payload, (bool, int, float)):
      return payload
    if isinstance(payload, str):
      return self._sanitize_text(payload)
    if isinstance(payload, bytes):
      return self._sanitize_text(payload.decode("utf-8", errors="replace"))
    if isinstance(payload, Mapping):
      return {str(key): self._sanitize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
      return [self._sanitize_payload(item) for item in payload]
    if isinstance(payload, tuple):
      return [self._sanitize_payload(item) for item in payload]
    return self._sanitize_text(str(payload))

  def _trace_metadata(
      self,
      *,
      path: str,
      method: str,
      request_payload: Any,
      status_code: int | None = None,
      duration_ms: float | None = None,
      stream: bool,
      prompt_version: str | None = None,
  ) -> dict[str, str]:
    metadata = {
        "path": path,
        "method": method.upper(),
        "stream": "true" if stream else "false",
    }
    if status_code is not None:
      metadata["status_code"] = str(status_code)
    if duration_ms is not None:
      metadata["duration_ms"] = f"{duration_ms:.2f}"
    if isinstance(request_payload, dict):
      model = request_payload.get("model")
      if isinstance(model, str) and model.strip():
        metadata["model"] = model.strip()
      messages = request_payload.get("messages")
      if isinstance(messages, list):
        metadata["message_count"] = str(len(messages))
    if prompt_version is not None:
      metadata["prompt_version"] = prompt_version
    return metadata

  def _extract_model_parameters(self, request_payload: Any) -> dict[str, Any]:
    if not isinstance(request_payload, Mapping):
      return {}

    model_parameters: dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_tokens",
        "min_tokens",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
        "chat_template_kwargs",
    ):
      value = request_payload.get(key)
      if value is not None:
        model_parameters[key] = self._sanitize_payload(value)
    return model_parameters

  def _usage_details_from_payload(self, payload: Any) -> dict[str, int]:
    if not isinstance(payload, Mapping):
      return {}

    usage = payload.get("usage") if "usage" in payload else payload
    if not isinstance(usage, Mapping):
      return {}

    usage_details: dict[str, int] = {}

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    # Langfuse v4 usage_details keys: "input" / "output" / "total"
    # (from official docs: usage_details={"input": N, "output": N})
    # OpenAI-style keys (prompt_tokens/completion_tokens) are also accepted and
    # auto-mapped by Langfuse server: prompt_tokens→input, completion_tokens→output.
    # We use the native keys directly for reliability.
    if isinstance(prompt_tokens, (int, float)) and not isinstance(prompt_tokens, bool):
      usage_details["input"] = int(prompt_tokens)
    if isinstance(completion_tokens, (int, float)) and not isinstance(completion_tokens, bool):
      usage_details["output"] = int(completion_tokens)
    if isinstance(total_tokens, (int, float)) and not isinstance(total_tokens, bool):
      usage_details["total"] = int(total_tokens)

    if "total" not in usage_details and "input" in usage_details and "output" in usage_details:
      usage_details["total"] = usage_details["input"] + usage_details["output"]

    # prompt_tokens_details keys get "input_" prefix (e.g. cached_tokens → input_cached_tokens)
    prompt_tokens_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_tokens_details, Mapping):
      for key, value in prompt_tokens_details.items():
        if isinstance(value, bool):
          continue
        if isinstance(value, (int, float)):
          usage_details[f"input_{key}"] = int(value)

    # completion_tokens_details keys get "output_" prefix (e.g. reasoning_tokens → output_reasoning_tokens)
    completion_tokens_details = usage.get("completion_tokens_details")
    if isinstance(completion_tokens_details, Mapping):
      for key, value in completion_tokens_details.items():
        if isinstance(value, bool):
          continue
        if isinstance(value, (int, float)):
          usage_details[f"output_{key}"] = int(value)

    return usage_details

  def _normalize_message_content(self, content: Any) -> str | None:
    if content is None:
      return None
    if isinstance(content, str):
      candidate = content.strip()
    elif isinstance(content, list):
      parts: list[str] = []
      for item in content:
        if isinstance(item, str):
          part = item.strip()
          if part:
            parts.append(part)
        elif isinstance(item, Mapping):
          text = item.get("text")
          if isinstance(text, str) and text.strip():
            parts.append(text.strip())
          elif item:
            parts.append(str(item))
        elif item is not None:
          parts.append(str(item))
      candidate = "\n".join(parts).strip()
    else:
      candidate = str(content).strip()
    if not candidate:
      return None
    return self._sanitize_text(candidate)

  def _summarize_messages_for_input(self, messages: list[Any]) -> dict[str, Any]:
    def summarize_message(message: Mapping[str, Any]) -> dict[str, Any]:
      summary: dict[str, Any] = {}
      role = message.get("role")
      if isinstance(role, str) and role.strip():
        summary["role"] = role.strip()

      content = self._normalize_message_content(message.get("content"))
      if content is not None:
        summary["content"] = content

      for key in ("name", "tool_call_id"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
          summary[key] = value.strip()

      for key in ("tool_calls", "function_call"):
        value = message.get(key)
        if value is not None:
          summary[key] = self._sanitize_payload(value)

      return summary

    history: list[dict[str, Any]] = []
    current_message: dict[str, Any] | None = None

    last_user_index = -1
    for index, message in enumerate(messages):
      if isinstance(message, Mapping):
        role = message.get("role")
        if role == "user":
          last_user_index = index

    for index, message in enumerate(messages):
      if not isinstance(message, Mapping):
        continue

      summarized_message = summarize_message(message)
      if not summarized_message:
        continue

      if index == last_user_index:
        current_message = summarized_message
      else:
        history.append(summarized_message)

    summary: dict[str, Any] = {
      "message_count": len(messages), "history": history}
    if current_message is not None:
      summary["current"] = current_message
    return summary

  def _summarize_request_for_input(self, request_payload: Any) -> Any:
    if not isinstance(request_payload, Mapping):
      return self._sanitize_payload(request_payload)

    summarized_request: dict[str, Any] = {
        key: self._sanitize_payload(value)
        for key, value in request_payload.items()
        if key != "messages"
    }

    messages = request_payload.get("messages")
    if isinstance(messages, list):
      summarized_request["messages"] = self._summarize_messages_for_input(
        messages)
    elif messages is not None:
      summarized_request["messages"] = self._sanitize_payload(messages)

    return summarized_request

  def _normalize_identifier(self, value: Any) -> str | None:
    if value is None:
      return None
    if isinstance(value, str):
      candidate = value.strip()
    else:
      candidate = str(value).strip()
    if not candidate:
      return None
    if len(candidate) > 200:
      candidate = candidate[:200]
    return candidate

  def _find_first_identifier(self, payload: Any, keys: tuple[str, ...]) -> str | None:
    if not isinstance(payload, Mapping):
      return None
    for key in keys:
      value = payload.get(key)
      normalized = self._normalize_identifier(value)
      if normalized:
        return normalized
    return None

  def _extract_trace_identity(self, request_payload: Any) -> dict[str, str | None]:
    user_id = self._find_first_identifier(
      request_payload, TRACE_REQUEST_USER_KEYS)
    session_id = self._find_first_identifier(
      request_payload, TRACE_REQUEST_SESSION_KEYS)

    if isinstance(request_payload, Mapping):
      metadata = request_payload.get("metadata")
      if isinstance(metadata, Mapping):
        if user_id is None:
          user_id = self._find_first_identifier(
            metadata, TRACE_REQUEST_USER_KEYS)
        if session_id is None:
          session_id = self._find_first_identifier(
            metadata, TRACE_REQUEST_SESSION_KEYS)

    return {"user_id": user_id, "session_id": session_id}

  def _merge_trace_identity(
      self,
      request_payload: Any,
      *,
      user_id: str | None = None,
      session_id: str | None = None,
  ) -> dict[str, str | None]:
    trace_identity = self._extract_trace_identity(request_payload)
    if user_id:
      trace_identity["user_id"] = user_id
    if session_id:
      trace_identity["session_id"] = session_id
    return trace_identity

  def _iter_sse_data_events(self, response_text: str) -> list[str]:
    events: list[str] = []
    current_lines: list[str] = []

    def flush_current_lines() -> bool:
      if not current_lines:
        return False
      data = "\n".join(current_lines).strip()
      current_lines.clear()
      if not data:
        return False
      if data == "[DONE]":
        return True
      events.append(data)
      return False

    for line in response_text.splitlines():
      stripped = line.strip()
      if not stripped:
        if flush_current_lines():
          return events
        continue
      if stripped.startswith(":"):
        continue
      if stripped.startswith("data:"):
        current_lines.append(stripped[5:].lstrip())

    flush_current_lines()
    return events

  def _summarize_stream_response(self, response_text: str) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[Any] = []
    usage: Any = None
    model: str | None = None
    finish_reason: str | None = None
    role: str | None = None
    chunk_count = 0
    parse_errors = 0

    for raw_event in self._iter_sse_data_events(response_text):
      try:
        event = json.loads(raw_event)
      except Exception:
        parse_errors += 1
        continue

      if not isinstance(event, dict):
        continue

      chunk_count += 1

      event_model = event.get("model")
      if model is None and isinstance(event_model, str) and event_model.strip():
        model = event_model.strip()

      if usage is None and event.get("usage") is not None:
        usage = event.get("usage")

      choices = event.get("choices")
      if not isinstance(choices, list):
        continue

      for choice in choices:
        if not isinstance(choice, dict):
          continue

        choice_finish_reason = choice.get("finish_reason")
        if choice_finish_reason is not None:
          finish_reason = str(choice_finish_reason)

        delta = choice.get("delta")
        if isinstance(delta, dict):
          delta_role = delta.get("role")
          if role is None and isinstance(delta_role, str) and delta_role.strip():
            role = delta_role.strip()

          delta_content = delta.get("content")
          if isinstance(delta_content, str) and delta_content:
            content_parts.append(delta_content)
          elif delta_content is not None and not isinstance(delta_content, dict):
            content_parts.append(str(delta_content))

          delta_reasoning_content = delta.get("reasoning_content")
          if isinstance(delta_reasoning_content, str) and delta_reasoning_content:
            reasoning_parts.append(delta_reasoning_content)
          elif delta_reasoning_content is not None and not isinstance(delta_reasoning_content, dict):
            reasoning_parts.append(str(delta_reasoning_content))

          delta_tool_calls = delta.get("tool_calls")
          if isinstance(delta_tool_calls, list) and delta_tool_calls:
            tool_calls.extend(delta_tool_calls)

          function_call = delta.get("function_call")
          if function_call is not None:
            tool_calls.append({"function_call": function_call})

        choice_text = choice.get("text")
        if isinstance(choice_text, str) and choice_text:
          content_parts.append(choice_text)

        message = choice.get("message")
        if isinstance(message, dict):
          message_content = message.get("content")
          if isinstance(message_content, str) and message_content:
            content_parts.append(message_content)

    summary: dict[str, Any] = {
        "stream": True,
        "chunk_count": chunk_count,
    }

    content = "".join(content_parts)
    if content:
      summary["content"] = content
    reasoning_content = "".join(reasoning_parts)
    if reasoning_content:
      summary["reasoning_content"] = reasoning_content
    if model:
      summary["model"] = model
    if role:
      summary["role"] = role
    if finish_reason:
      summary["finish_reason"] = finish_reason
    if usage is not None:
      summary["usage"] = usage
    if tool_calls:
      summary["tool_calls"] = tool_calls
    if parse_errors:
      summary["parse_errors"] = parse_errors
    return summary

  def record(
      self,
      *,
      path: str,
      method: str,
      request_payload: Any,
      response_payload: Any,
      status_code: int,
      duration_ms: float,
      start_time_perf: float | None = None,
      stream: bool = False,
      user_id: str | None = None,
      session_id: str | None = None,
      prompt_version: str | None = None,
      trace_name: str | None = None,
  ) -> None:
    if self.client is None:
      return
    trace_identity = self._merge_trace_identity(
        request_payload,
        user_id=user_id,
        session_id=session_id,
    )
    trace_kwargs: dict[str, Any] = {
        "name": trace_name or f"{method.upper()} {path}",
        "input": self._sanitize_payload(
            {
                "path": path,
                "method": method,
                "request": self._summarize_request_for_input(request_payload),
            }
        ),
        "metadata": self._trace_metadata(
            path=path,
            method=method,
            request_payload=request_payload,
            status_code=status_code,
            duration_ms=duration_ms,
            stream=stream,
            prompt_version=prompt_version,
        ),
    }
    model = trace_kwargs["metadata"].get("model")
    usage_details = self._usage_details_from_payload(response_payload)

    # Calculate start/end timestamps in nanoseconds for accurate latency.
    # perf_counter has no epoch; convert to wall-clock ns via the offset
    # between time.time_ns() and time.perf_counter().
    _now_ns = _time.time_ns()
    _now_perf = _time.perf_counter()
    _perf_to_ns_offset = _now_ns - int(_now_perf * 1e9)

    end_ns = _now_ns
    if start_time_perf is not None:
      start_ns = int(start_time_perf * 1e9) + _perf_to_ns_offset
    else:
      start_ns = end_ns - int(duration_ms * 1e6)

    _prop_tags: list[str] | None = None
    if prompt_version is not None:
      _prop_tags = [f"prompt_version:{prompt_version}"]

    with propagate_attributes(
        user_id=trace_identity.get("user_id"),
        session_id=trace_identity.get("session_id"),
        trace_name=trace_kwargs["name"],
        metadata={k: str(v) for k, v in trace_kwargs["metadata"].items()},
        tags=_prop_tags,
    ):
      obs = self.client.start_observation(
          name=trace_kwargs["name"],
          as_type="generation",
          input=trace_kwargs["input"],
          metadata=trace_kwargs["metadata"],
          model=model,
          model_parameters=self._extract_model_parameters(request_payload),
      )
      # Patch the OTel span's start time to reflect the real request start.
      if hasattr(obs, "_otel_span") and hasattr(obs._otel_span, "_start_time"):
        obs._otel_span._start_time = start_ns

      obs.update(
          output=self._sanitize_payload(response_payload),
          usage_details=usage_details if usage_details else None,
      )
      obs.end(end_time=end_ns)

    self.client.flush()

  def record_stream(
      self,
      *,
      path: str,
      method: str,
      request_payload: Any,
      response_text: str,
      status_code: int,
      duration_ms: float,
      start_time_perf: float | None = None,
      user_id: str | None = None,
      session_id: str | None = None,
      prompt_version: str | None = None,
      trace_name: str | None = None,
  ) -> None:
    response_payload = self._summarize_stream_response(response_text)
    self.record(
        path=path,
        method=method,
        request_payload=request_payload,
        response_payload=response_payload,
        status_code=status_code,
        duration_ms=duration_ms,
        start_time_perf=start_time_perf,
        stream=True,
        user_id=user_id,
        session_id=session_id,
        prompt_version=prompt_version,
        trace_name=trace_name,
    )
