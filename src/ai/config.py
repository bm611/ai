import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "ai-cli"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS = {
    "model": "openai/gpt-oss-120b",
    "provider": None,  # let OpenRouter pick; or set e.g. {"order": ["DeepInfra"]}
    "theme": "auto",  # "auto", "dark", "light", or any pygments theme name
}


def is_dark_mode() -> bool:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=2,
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
        return json.loads(CONFIG_FILE.read_text())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(DEFAULTS, indent=2) + "\n")
    return dict(DEFAULTS)


def load_config() -> dict:
    return _ensure_config()


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2) + "\n")
