from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response

from app.dependencies import get_http_client, get_langfuse, get_prompt_manager, get_settings
from app.proxy import proxy_request

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
