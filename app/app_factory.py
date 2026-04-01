from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response  # pyright: ignore[reportMissingImports]

from app.config_loader import PromptConfigLoader
from app.langfuse_recorder import LangfuseRecorder
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
    app.state.prompt_config_loader = PromptConfigLoader(settings.prompt_config_path)
    logger.info("Gateway started for upstream %s", settings.vllm_base_url)
    try:
      yield
    finally:
      client = app.state.http_client
      await client.aclose()
      langfuse = getattr(app.state, "langfuse", None)
      langfuse_client = getattr(langfuse, "client", None)
      if langfuse_client is not None:
        try:
          langfuse_client.flush()
        except Exception:
          logger.exception("Failed to flush Langfuse on shutdown")
      prompt_config_loader = getattr(app.state, "prompt_config_loader", None)
      if prompt_config_loader is not None:
        try:
          prompt_config_loader.stop()
        except Exception:
          logger.exception("Failed to stop PromptConfigLoader on shutdown")

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
