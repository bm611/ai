import json
import os
import sys
import time

import httpx
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

DARK_THEME = Theme(
    {
        "markdown.h1": "bold bright_magenta",
        "markdown.h2": "bold bright_cyan",
        "markdown.h3": "bold bright_green",
        "markdown.h4": "bold bright_yellow",
        "markdown.link": "bright_blue underline",
        "markdown.item.bullet": "bright_cyan",
        "markdown.item.number": "bright_cyan",
        "markdown.block_quote": "italic magenta",
        "markdown.code": "bright_green on grey11",
        "markdown.hr": "bright_magenta",
    }
)

LIGHT_THEME = Theme(
    {
        "markdown.h1": "bold dark_magenta",
        "markdown.h2": "bold dark_cyan",
        "markdown.h3": "bold green",
        "markdown.h4": "bold dark_orange",
        "markdown.link": "blue underline",
        "markdown.item.bullet": "dark_cyan",
        "markdown.item.number": "dark_cyan",
        "markdown.block_quote": "italic dark_magenta",
        "markdown.code": "dark_green on grey93",
        "markdown.hr": "dark_magenta",
    }
)

CODE_THEMES = {
    "dark": "monokai",
    "light": "friendly",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        Console(stderr=True).print(
            "[red bold]Error:[/] OPENROUTER_API_KEY environment variable is not set.\n"
            "Export it or add it to your shell profile."
        )
        sys.exit(1)
    return key


def _resolve_theme(theme_setting: str) -> tuple[Theme, str]:
    from ai.config import is_dark_mode

    if theme_setting == "auto":
        mode = "dark" if is_dark_mode() else "light"
    elif theme_setting in ("dark", "light"):
        mode = theme_setting
    else:
        # treat as a raw pygments code theme name, detect markdown theme
        mode = "dark" if is_dark_mode() else "light"
        md_theme = DARK_THEME if mode == "dark" else LIGHT_THEME
        return md_theme, theme_setting

    md_theme = DARK_THEME if mode == "dark" else LIGHT_THEME
    return md_theme, CODE_THEMES[mode]


def stream_prompt(
    prompt: str,
    model: str,
    provider: dict | None = None,
    theme: str = "auto",
    chat: bool = False,
) -> None:
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-cli",
        "X-Title": "ai-cli",
    }

    messages: list[dict] = [{"role": "user", "content": prompt}]

    body: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if provider:
        body["provider"] = provider

    md_theme, code_theme = _resolve_theme(theme)

    def _md(text: str) -> Markdown:
        return Markdown(text, code_theme=code_theme)

    console = Console(theme=md_theme)

    # ── banner ──
    logo = [
        r"  __                       .__              .__           .__            __   ",
        r"_/  |_  ___________  _____ |__| ____ _____  |  |     ____ |  |__ _____ _/  |_ ",
        r"\   __\/ __ \_  __ \/     \|  |/    \\__  \ |  |   _/ ___\|  |  \\__  \\   __\\",
        r" |  | \  ___/|  | \/  Y Y  \  |   |  \/ __ \|  |__ \  \___|   Y  \/ __ \|  |  ",
        r" |__|  \___  >__|  |__|_|  /__|___|  (____  /____/  \___  >___|  (____  /__|  ",
        r"           \/            \/        \/     \/            \/     \/     \/      ",
    ]
    console.print()
    for line in logo:
        styled = Text(line, style="bold magenta")
        console.print(styled)
    console.print(Rule(style="dim"))

    collected = ""
    token_count = 0
    used_model = model
    provider_name = ""
    t_start = time.perf_counter()
    first_token_time: float | None = None

    with httpx.Client(timeout=120) as http:
        with http.stream("POST", OPENROUTER_URL, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                resp.read()
                console.print(f"[red bold]Error {resp.status_code}:[/] {resp.text}")
                sys.exit(1)

            usage = {}
            with Live(
                Padding(_md(""), (1, 2)),
                console=console,
                refresh_per_second=8,
            ) as live:
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line.removeprefix("data: ").strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    if "model" in chunk:
                        used_model = chunk["model"]
                    if "provider" in chunk:
                        provider_name = chunk["provider"]
                    if "usage" in chunk:
                        usage = chunk["usage"]
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
                        collected += text
                        live.update(Padding(_md(collected), (1, 2)))

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    gen_time = t_end - (first_token_time or t_start)

    # prefer real token count from usage data
    real_tokens = usage.get("completion_tokens", token_count)
    tok_per_sec = real_tokens / gen_time if gen_time > 0 else 0

    parts = [
        ("✦ ", "bold magenta"),
        (used_model, "bold cyan"),
    ]
    if provider_name:
        parts += [("  via ", "dim"), (provider_name, "bold yellow")]
    parts += [
        ("  │  ", "dim"),
        (f"{real_tokens}", "bold"),
        (" tokens", "dim"),
        ("  │  ", "dim"),
        (f"{tok_per_sec:.1f}", "bold"),
        (" tok/s", "dim"),
        ("  │  ", "dim"),
        (f"{elapsed:.1f}", "bold"),
        ("s", "dim"),
    ]
    console.print(Rule(style="dim"))
    console.print(Text.assemble(*parts))
    console.print()

    if not chat:
        return

    while True:
        try:
            followup = input(
                "\n\033[1;36mFollow-up\033[0m\033[2m (or \033[1mq\033[0m\033[2m to quit):\033[0m "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if followup.lower() in ("q", "quit", "exit"):
            return

        if not followup:
            continue

        messages.append({"role": "assistant", "content": collected})
        messages.append({"role": "user", "content": followup})
        body["messages"] = messages

        collected = ""
        token_count = 0
        used_model = model
        provider_name = ""
        t_start = time.perf_counter()
        first_token_time = None

        with httpx.Client(timeout=120) as http:
            with http.stream(
                "POST", OPENROUTER_URL, json=body, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    console.print(f"[red bold]Error {resp.status_code}:[/] {resp.text}")
                    sys.exit(1)

                usage = {}
                with Live(
                    Padding(_md(""), (1, 2)),
                    console=console,
                    refresh_per_second=8,
                ) as live:
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line.removeprefix("data: ").strip()
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        if "model" in chunk:
                            used_model = chunk["model"]
                        if "provider" in chunk:
                            provider_name = chunk["provider"]
                        if "usage" in chunk:
                            usage = chunk["usage"]
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                            collected += text
                            live.update(Padding(_md(collected), (1, 2)))

        t_end = time.perf_counter()
        elapsed = t_end - t_start
        gen_time = t_end - (first_token_time or t_start)

        real_tokens = usage.get("completion_tokens", token_count)
        tok_per_sec = real_tokens / gen_time if gen_time > 0 else 0

        parts = [
            ("✦ ", "bold magenta"),
            (used_model, "bold cyan"),
        ]
        if provider_name:
            parts += [("  via ", "dim"), (provider_name, "bold yellow")]
        parts += [
            ("  │  ", "dim"),
            (f"{real_tokens}", "bold"),
            (" tokens", "dim"),
            ("  │  ", "dim"),
            (f"{tok_per_sec:.1f}", "bold"),
            (" tok/s", "dim"),
            ("  │  ", "dim"),
            (f"{elapsed:.1f}", "bold"),
            ("s", "dim"),
        ]
        console.print(Rule(style="dim"))
        console.print(Text.assemble(*parts))
        console.print()
