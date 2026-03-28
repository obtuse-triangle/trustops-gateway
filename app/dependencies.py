from __future__ import annotations

from fastapi import Request

from app.langfuse_recorder import LangfuseRecorder
from app.prompt_manager import PromptManager
from app.settings import Settings



def get_settings(request: Request) -> Settings:
    return request.app.state.settings



def get_http_client(request: Request):
    return request.app.state.http_client



def get_langfuse(request: Request) -> LangfuseRecorder | None:
    return request.app.state.langfuse



def get_prompt_manager(request: Request) -> PromptManager | None:
    return getattr(request.app.state, "prompt_manager", None)
