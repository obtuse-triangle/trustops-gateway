from __future__ import annotations

from fastapi import Request

from app.langfuse_recorder import LangfuseRecorder
from app.settings import Settings



def get_settings(request: Request) -> Settings:
    return request.app.state.settings



def get_http_client(request: Request):
    return request.app.state.http_client



def get_langfuse(request: Request) -> LangfuseRecorder | None:
    return request.app.state.langfuse
