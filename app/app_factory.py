from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from app.langfuse_recorder import LangfuseRecorder
from app.prompt_manager import PromptManager
from app.proxy import create_http_client
from app.routes import router
from app.settings import get_settings


def create_app() -> FastAPI:
  settings = get_settings()
  logging.basicConfig(level=settings.log_level)
  logger = logging.getLogger("trustopsback")

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.http_client = create_http_client(settings)
    app.state.langfuse = LangfuseRecorder(settings)
    app.state.prompt_manager = PromptManager(
        settings.prompts_dir,
        canary_weight_env=settings.canary_weight_env,
    )
    logger.info("Gateway started for upstream %s", settings.vllm_base_url)
    try:
      yield
    finally:
      client = app.state.http_client
      await client.aclose()
      langfuse = getattr(app.state, "langfuse", None)
      if getattr(langfuse, "client", None) is not None:
        try:
          langfuse.client.flush()
        except Exception:
          logger.exception("Failed to flush Langfuse on shutdown")
      prompt_manager = getattr(app.state, "prompt_manager", None)
      if prompt_manager is not None:
        try:
          prompt_manager.stop()
        except Exception:
          logger.exception("Failed to stop PromptManager on shutdown")

  app = FastAPI(title="trustOpsBack vLLM Gateway",
                version="0.1.0", lifespan=lifespan)

  @app.middleware("http")
  async def api_key_guard(request: Request, call_next):
    if settings.gateway_api_key:
      provided_key = request.headers.get("x-gateway-api-key", "").strip()
      if provided_key != settings.gateway_api_key:
        return Response(content='{"detail":"Unauthorized"}', status_code=401, media_type="application/json")
    return await call_next(request)

  app.include_router(router)
  return app
