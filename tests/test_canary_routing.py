from __future__ import annotations

import random
import time
from pathlib import Path

from app.prompt_manager import PromptManager


def _make_prompt_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    d = tmp_path / "prompts"
    d.mkdir()
    for name, content in files.items():
        (d / name).write_text(content, encoding="utf-8")
    return d


def test_get_current_config_returns_correct_values(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable content",
        "prompt_v2.txt": "canary content",
        "canary_weight.txt": "10",
    })
    pm = PromptManager(str(d))
    try:
        config = pm.get_current_config()
        assert config["stable_version"] == "v1"
        assert config["canary_version"] == "v2"
        assert config["canary_weight"] == 10.0
        assert config["available_versions"] == ["v1", "v2"]
    finally:
        pm.stop()


def test_is_healthy_true_with_prompts(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable",
        "prompt_v2.txt": "canary",
    })
    pm = PromptManager(str(d))
    try:
        assert pm.is_healthy() is True
    finally:
        pm.stop()


def test_is_healthy_false_no_prompts(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_prompts"
    empty_dir.mkdir()
    pm = PromptManager(str(empty_dir))
    try:
        assert pm.is_healthy() is False
    finally:
        pm.stop()


def test_90_10_distribution_with_weight_10(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable",
        "prompt_v2.txt": "canary",
        "canary_weight.txt": "10",
    })
    pm = PromptManager(str(d))
    try:
        rng = random.Random(99)
        canary_count = sum(
            1 for _ in range(1000)
            if pm.get_prompt(rng.random())[1] == "v2"
        )
        assert 50 <= canary_count <= 150, f"Expected ~100 canary hits (±50), got {canary_count}"
    finally:
        pm.stop()


def test_weight_zero_routes_all_to_stable(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable",
        "prompt_v2.txt": "canary",
        "canary_weight.txt": "0",
    })
    pm = PromptManager(str(d))
    try:
        rng = random.Random(7)
        tags = {pm.get_prompt(rng.random())[1] for _ in range(1000)}
        assert tags == {"v1"}, f"All requests should route to stable with weight=0, got tags={tags}"
    finally:
        pm.stop()


def test_weight_100_routes_all_to_canary(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable",
        "prompt_v2.txt": "canary",
        "canary_weight.txt": "100",
    })
    pm = PromptManager(str(d))
    try:
        rng = random.Random(13)
        tags = {pm.get_prompt(rng.random())[1] for _ in range(1000)}
        assert tags == {"v2"}, f"All requests should route to canary with weight=100, got tags={tags}"
    finally:
        pm.stop()


def test_hot_reload_activates_canary(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {"prompt_v1.txt": "stable only"})
    pm = PromptManager(str(d))
    try:
        assert pm.is_healthy() is True
        _, tag = pm.get_prompt(0.05)
        assert tag == "v1", "Single-prompt setup should always route to v1"

        (d / "prompt_v2.txt").write_text("new canary", encoding="utf-8")
        (d / "canary_weight.txt").write_text("10", encoding="utf-8")
        time.sleep(0.8)

        _, tag = pm.get_prompt(0.05)
        assert tag == "v2", f"After hot-reload with weight=10, roll=0.05 should route to canary (v2), got {tag}"
    finally:
        pm.stop()


def test_changing_weight_triggers_rebalance(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable",
        "prompt_v2.txt": "canary",
        "canary_weight.txt": "0",
    })
    pm = PromptManager(str(d))
    try:
        _, tag = pm.get_prompt(0.3)
        assert tag == "v1", "Weight=0 should route all to stable"

        (d / "canary_weight.txt").write_text("50", encoding="utf-8")
        time.sleep(0.8)

        _, tag = pm.get_prompt(0.3)
        assert tag == "v2", f"After weight changed to 50, roll=0.3 should route to canary, got {tag}"
    finally:
        pm.stop()


def test_removing_canary_reverts_to_stable(tmp_path: Path) -> None:
    d = _make_prompt_dir(tmp_path, {
        "prompt_v1.txt": "stable",
        "prompt_v2.txt": "canary",
        "canary_weight.txt": "50",
    })
    pm = PromptManager(str(d))
    try:
        _, tag = pm.get_prompt(0.3)
        assert tag == "v2", "Weight=50 + roll=0.3 should route to canary initially"

        (d / "prompt_v2.txt").unlink()
        time.sleep(0.8)

        stable_text, stable_tag = pm.get_stable_prompt()
        canary_text, canary_tag = pm.get_canary_prompt()
        assert stable_tag == canary_tag == "v1", (
            f"After removing v2, both stable and canary should be v1, got stable={stable_tag} canary={canary_tag}"
        )
        rng = random.Random(42)
        tags = {pm.get_prompt(rng.random())[1] for _ in range(100)}
        assert tags == {"v1"}, f"After removing canary, all requests should route to v1, got {tags}"
    finally:
        pm.stop()
