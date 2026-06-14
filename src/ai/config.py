import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "ai-cli"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS = {
    "model": "deepseek/deepseek-v4-flash",
    "provider": None,  # let OpenRouter pick; or set e.g. {"order": ["DeepInfra"]}
    "theme": "auto",  # "auto", "dark", or "light"
    # ensemble mode: query these models in parallel, then consolidate
    "ensemble_models": [
        "deepseek/deepseek-v4-flash",
        "google/gemini-3.1-flash-lite-preview",
    ],
    "consensus_model": "deepseek/deepseek-v4-pro",
}


def is_dark_mode() -> bool:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return result.stdout.strip().lower() == "dark"
        except Exception:
            return True  # macOS defaults to light, but most dev terminals are dark
    # Linux / other: check common env hints
    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg:
        parts = colorfgbg.split(";")
        try:
            bg = int(parts[-1])
            return bg < 8  # low values = dark background
        except ValueError:
            pass
    return True  # default to dark


def _ensure_config() -> dict:
    if CONFIG_FILE.exists():
        cfg = json.loads(CONFIG_FILE.read_text())
        # backfill any keys added in newer versions
        merged = {**DEFAULTS, **cfg}
        if merged != cfg:
            save_config(merged)
        return merged
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(DEFAULTS, indent=2) + "\n")
    return dict(DEFAULTS)


def load_config() -> dict:
    return _ensure_config()


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n")
