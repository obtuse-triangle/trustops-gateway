from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("trustopsback")


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str
    temperature: float | None
    top_p: float | None
    top_k: int | None
    prompt_version: str


def _as_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        data = raw.get("data")
        if isinstance(data, dict):
            return data
        return raw
    raise ValueError("prompt config must be a mapping")


def _text_value(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def _float_value(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise ValueError(f"{key} must be numeric")


def _int_value(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(float(value.strip()))
    raise ValueError(f"{key} must be numeric")


def load_prompt_config(path: str) -> PromptConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = _as_mapping(raw or {})

    system_prompt = _text_value(data, "system_prompt", "system_prompt.txt")
    prompt_version = _text_value(data, "prompt_version")

    return PromptConfig(
        system_prompt=system_prompt,
        temperature=_float_value(data, "temperature"),
        top_p=_float_value(data, "top_p"),
        top_k=_int_value(data, "top_k"),
        prompt_version=prompt_version,
    )


class PromptConfigLoader:
    def __init__(self, config_path: str) -> None:
        self._config_path = Path(config_path)
        self._lock = threading.RLock()
        self._config = PromptConfig("", None, None, None, "")

        self._reload()

    def _reload(self) -> None:
        try:
            new_config = load_prompt_config(str(self._config_path))
            with self._lock:
                self._config = new_config
            logger.info("PromptConfigLoader loaded %s", self._config_path)
        except Exception:
            logger.exception("PromptConfigLoader: error during reload — keeping previous state")

    def get_config(self) -> PromptConfig:
        with self._lock:
            return self._config

    def stop(self) -> None:
        logger.info("PromptConfigLoader stopped for %s", self._config_path)
