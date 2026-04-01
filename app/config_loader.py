from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from watchdog.events import FileSystemEventHandler  # pyright: ignore[reportMissingImports]
from watchdog.observers import Observer  # pyright: ignore[reportMissingImports]

logger = logging.getLogger("trustopsback")


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str
    temperature: float | None
    top_p: float | None
    top_k: int | None
    prompt_version: str


class _ConfigEventHandler(FileSystemEventHandler):
    def __init__(self, loader: "PromptConfigLoader", debounce_seconds: float = 0.3) -> None:
        super().__init__()
        self._loader = loader
        self._debounce_seconds = debounce_seconds
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def _schedule_reload(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._loader._reload)
            self._timer.daemon = True
            self._timer.start()

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self._schedule_reload()

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._schedule_reload()

    def on_deleted(self, event) -> None:
        if not event.is_directory:
            self._schedule_reload()


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
        self._observer: Observer | None = None

        self._reload()

        if self._config_path.parent.is_dir():
            self._event_handler = _ConfigEventHandler(self)
            observer = Observer()
            observer.schedule(self._event_handler, str(self._config_path.parent), recursive=False)
            observer.start()
            self._observer = observer
            logger.info("PromptConfigLoader watching %s", self._config_path)

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
        if self._observer is not None and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("PromptConfigLoader stopped watching %s", self._config_path)
