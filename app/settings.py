from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    vllm_base_url: str
    gateway_api_key: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str
    langfuse_enabled: bool
    request_timeout_seconds: float
    max_response_bytes: int
    log_level: str
    prompts_dir: str = "/app/prompts"
    canary_weight_env: str = "CANARY_WEIGHT"



def _read_bool(value: str) -> bool:
    return value.lower() not in {"0", "false", "no", "off"}



def get_settings() -> Settings:
    return Settings(
        vllm_base_url=os.getenv("VLLM_BASE_URL", "").rstrip("/"),
        gateway_api_key=os.getenv("GATEWAY_API_KEY", "").strip(),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
        langfuse_host=os.getenv("LANGFUSE_HOST", "").strip(),
        langfuse_enabled=_read_bool(os.getenv("LANGFUSE_ENABLED", "true")),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
        max_response_bytes=int(os.getenv("MAX_RESPONSE_BYTES", str(20 * 1024 * 1024))),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        prompts_dir=os.getenv("PROMPTS_DIR", "/app/prompts"),
        canary_weight_env=os.getenv("CANARY_WEIGHT_ENV", "CANARY_WEIGHT"),
    )
