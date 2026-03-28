from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import pytest

from app.prompt_manager import PromptManager


def _make_prompt_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    d = tmp_path / "prompts"
    d.mkdir()
    for name, content in files.items():
        (d / name).write_text(content, encoding="utf-8")
    return d


def test_read_prompt_versions(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "v1 content",
        "prompt_v2.txt": "v2 content",
    })
    pm = PromptManager(str(d))
    try:
        stable_text, stable_tag = pm.get_stable_prompt()
        canary_text, canary_tag = pm.get_canary_prompt()
        assert stable_text == "v1 content"
        assert stable_tag == "v1"
        assert canary_text == "v2 content"
        assert canary_tag == "v2"
    finally:
        pm.stop()


def test_canary_routing_weight(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable content",
        "prompt_v2.txt": "canary content",
        "canary_weight.txt": "30",
    })
    pm = PromptManager(str(d))
    try:
        rng = random.Random(42)
        canary_count = sum(
            1 for _ in range(1000)
            if pm.get_prompt(rng.random())[1] == "v2"
        )
        assert 250 <= canary_count <= 350, f"Expected ~300 canary requests, got {canary_count}"
    finally:
        pm.stop()


def test_hot_reload(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "old stable",
        "prompt_v2.txt": "old canary",
    })
    pm = PromptManager(str(d))
    try:
        text, _ = pm.get_canary_prompt()
        assert text == "old canary"

        (d / "prompt_v3.txt").write_text("new canary", encoding="utf-8")
        time.sleep(0.8)

        text, tag = pm.get_canary_prompt()
        assert text == "new canary", f"Expected 'new canary' after hot-reload, got '{text}'"
        assert tag == "v3"
    finally:
        pm.stop()


def test_single_prompt_no_canary(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {"prompt_v1.txt": "only prompt"})
    pm = PromptManager(str(d))
    try:
        stable_text, stable_tag = pm.get_stable_prompt()
        canary_text, canary_tag = pm.get_canary_prompt()
        assert stable_text == "only prompt"
        assert canary_text == "only prompt"
        assert stable_tag == canary_tag == "v1"
        for roll in [0.0, 0.01, 0.5, 0.99]:
            text, tag = pm.get_prompt(roll)
            assert text == "only prompt", f"roll={roll} should always return the single prompt"
    finally:
        pm.stop()


def test_missing_weight_file_falls_back_to_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable",
        "prompt_v2.txt": "canary",
    })
    monkeypatch.setenv("TEST_CANARY_WEIGHT", "50")
    pm = PromptManager(str(d), canary_weight_env="TEST_CANARY_WEIGHT")
    try:
        _, tag = pm.get_prompt(0.49)
        assert tag == "v2", f"roll=0.49 with weight=50 should route to canary, got {tag}"
        _, tag = pm.get_prompt(0.51)
        assert tag == "v1", f"roll=0.51 with weight=50 should route to stable, got {tag}"
    finally:
        pm.stop()


def test_default_weight_is_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CANARY_WEIGHT", raising=False)
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable",
        "prompt_v2.txt": "canary",
    })
    pm = PromptManager(str(d))
    try:
        for roll in [0.0, 0.01, 0.5, 0.99]:
            _, tag = pm.get_prompt(roll)
            assert tag == "v1", f"With weight=0, roll={roll} should always return stable (v1), got {tag}"
    finally:
        pm.stop()


def test_atomic_swap_thread_safety(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable",
        "prompt_v2.txt": "canary",
        "canary_weight.txt": "50",
    })
    pm = PromptManager(str(d))
    errors: list[str] = []

    def reader() -> None:
        for _ in range(500):
            try:
                text, tag = pm.get_prompt(0.5)
                assert isinstance(text, str) and isinstance(tag, str)
            except Exception as exc:
                errors.append(str(exc))

    def reloader() -> None:
        for _ in range(5):
            pm._reload()
            time.sleep(0.01)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads.append(threading.Thread(target=reloader))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    pm.stop()
    assert errors == [], f"Thread safety errors detected: {errors}"
