from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("trustopsback")

_VERSION_PATTERN = re.compile(r"^prompt_v(\d+)\.txt$")


class _PromptEventHandler(FileSystemEventHandler):
    def __init__(self, manager: "PromptManager", debounce_seconds: float = 0.3) -> None:
        super().__init__()
        self._manager = manager
        self._debounce_seconds = debounce_seconds
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def _schedule_reload(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._manager._reload)
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


class PromptManager:
    """
    Loads system prompts from ConfigMap-mounted files with watchdog hot-reload.

    File layout in prompts_dir:
      prompt_v<N>.txt   — versioned prompt files; the two highest-versioned
                          files become stable (older) and canary (newest).
      canary_weight.txt — optional integer 0-100; percentage of traffic
                          routed to canary. Falls back to canary_weight_env,
                          then defaults to 0 (all traffic to stable).

    If only one prompt file exists, stable == canary.
    """

    def __init__(
        self,
        prompts_dir: str,
        canary_weight_env: str = "CANARY_WEIGHT",
    ) -> None:
        self._prompts_dir = Path(prompts_dir)
        self._canary_weight_env = canary_weight_env
        self._lock = threading.RLock()
        self._data: dict = {"stable": ("", "v0"), "canary": ("", "v0"), "weight": 0.0, "versions": []}

        self._reload()

        self._observer: Optional[Observer] = None
        if self._prompts_dir.is_dir():
            self._event_handler = _PromptEventHandler(self)
            self._observer = Observer()
            self._observer.schedule(self._event_handler, str(self._prompts_dir), recursive=False)
            self._observer.start()
            logger.info("PromptManager watching %s", self._prompts_dir)

    def _reload(self) -> None:
        try:
            prompt_files: list[tuple[int, Path]] = []
            if self._prompts_dir.is_dir():
                for f in self._prompts_dir.iterdir():
                    m = _VERSION_PATTERN.match(f.name)
                    if m:
                        prompt_files.append((int(m.group(1)), f))

            prompt_files.sort(key=lambda x: x[0])

            if not prompt_files:
                logger.info("PromptManager: no prompt files found in %s", self._prompts_dir)
                return

            if len(prompt_files) == 1:
                v_num, f_path = prompt_files[0]
                text = f_path.read_text(encoding="utf-8")
                tag = f"v{v_num}"
                stable: tuple[str, str] = (text, tag)
                canary: tuple[str, str] = (text, tag)
            else:
                v_stable, f_stable = prompt_files[-2]
                v_canary, f_canary = prompt_files[-1]
                stable = (f_stable.read_text(encoding="utf-8"), f"v{v_stable}")
                canary = (f_canary.read_text(encoding="utf-8"), f"v{v_canary}")

            weight = self._read_weight()

            all_versions = [f"v{v}" for v, _ in prompt_files]
            new_data = {"stable": stable, "canary": canary, "weight": weight, "versions": all_versions}
            with self._lock:
                self._data = new_data

            logger.info(
                "PromptManager loaded: stable=%s canary=%s weight=%.1f versions=%d",
                stable[1], canary[1], weight, len(prompt_files),
            )
        except Exception:
            logger.exception("PromptManager: error during reload — keeping previous state")

    def _read_weight(self) -> float:
        weight_file = self._prompts_dir / "canary_weight.txt"
        if weight_file.is_file():
            try:
                return float(weight_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
        env_val = os.environ.get(self._canary_weight_env, "").strip()
        if env_val:
            try:
                return float(env_val)
            except ValueError:
                pass
        return 0.0

    def get_prompt(self, roll: float) -> tuple[str, str]:
        """
        Route based on roll (0.0–1.0). Returns (prompt_text, version_tag).
        Returns canary if roll < weight/100, else stable.
        """
        with self._lock:
            data = self._data
        if roll < data["weight"] / 100.0:
            return data["canary"]
        return data["stable"]

    def get_current_config(self) -> dict:
        with self._lock:
            data = self._data
        return {
            "stable_version": data["stable"][1],
            "canary_version": data["canary"][1],
            "canary_weight": data["weight"],
            "available_versions": list(data.get("versions", [])),
        }

    def is_healthy(self) -> bool:
        with self._lock:
            data = self._data
        return bool(data["stable"][0]) and bool(data["canary"][0])

    def get_stable_prompt(self) -> tuple[str, str]:
        with self._lock:
            return self._data["stable"]

    def get_canary_prompt(self) -> tuple[str, str]:
        with self._lock:
            return self._data["canary"]

    def start(self) -> None:
        if self._observer is not None and not self._observer.is_alive():
            self._observer.start()

    def stop(self) -> None:
        if self._observer is not None and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("PromptManager stopped watching %s", self._prompts_dir)
