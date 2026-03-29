from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request, Response

from app.dependencies import get_http_client, get_langfuse, get_prompt_manager, get_settings
from app.proxy import apply_preview_config, proxy_request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
  settings = get_settings(request)
  langfuse = get_langfuse(request)
  return {
      "status": "ok",
      "upstream": settings.vllm_base_url,
      "langfuse_enabled": bool(langfuse and langfuse.client),
  }


@router.get("/")
async def root(request: Request) -> dict[str, str]:
  settings = get_settings(request)
  return {"message": "trustOpsBack vLLM gateway", "upstream": settings.vllm_base_url}


@router.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_v1(path: str, request: Request) -> Response:
  settings = get_settings(request)
  return await proxy_request(
      path=f"/v1/{path}",
      request=request,
      client=get_http_client(request),
      settings=settings,
      langfuse=get_langfuse(request),
      prompt_manager=get_prompt_manager(request),
  )


@router.api_route("/openai/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_openai_compat(path: str, request: Request) -> Response:
  settings = get_settings(request)
  return await proxy_request(
      path=f"/v1/{path}",
      request=request,
      client=get_http_client(request),
      settings=settings,
      langfuse=get_langfuse(request),
      prompt_manager=get_prompt_manager(request),
  )


@router.post("/preview")
async def preview(request: Request) -> Response:
  settings = get_settings(request)
  body = await request.body()
  request_json: dict[str, Any] = {}
  if body:
    parsed = json.loads(body.decode("utf-8"))
    if isinstance(parsed, dict):
      request_json = parsed

  prompt_config_loader = getattr(request.app.state, "prompt_config_loader", None)
  prompt_config = prompt_config_loader.get_config() if prompt_config_loader is not None else None
  preview_json = apply_preview_config(request_json, prompt_config)

  return await proxy_request(
      path="/v1/chat/completions",
      upstream_path="/v1/chat/completions",
      request=request,
      client=get_http_client(request),
      settings=settings,
      langfuse=get_langfuse(request),
      prompt_manager=None,
      request_json_override=preview_json,
      body_override=json.dumps(preview_json).encode("utf-8"),
      apply_generation_config=False,
      trace_name="playground-preview",
  )
