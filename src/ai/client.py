import json
import os
import sys
import time

import httpx
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

from ai.tools import TOOLS, execute_tool

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


def _do_stream(
    body: dict,
    headers: dict,
    messages: list[dict],
    console: Console,
    _md,
) -> tuple[str, str, str, dict]:
    """Stream a single chat completion, handling tool calls automatically.

    Returns (collected_text, used_model, provider_name, usage).
    """
    collected = ""
    token_count = 0
    used_model = body["model"]
    provider_name = ""
    t_start = time.perf_counter()
    first_token_time: float | None = None

    # accumulators for tool calls streamed in deltas
    tool_calls_acc: dict[int, dict] = {}  # index -> {id, name, arguments}

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

                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})

                    # accumulate tool_calls deltas
                    for tc in delta.get("tool_calls", []):
                        idx = tc["index"]
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.get("id", ""),
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": "",
                            }
                        else:
                            if tc.get("id"):
                                tool_calls_acc[idx]["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                tool_calls_acc[idx]["name"] = tc["function"]["name"]
                        tool_calls_acc[idx]["arguments"] += tc.get("function", {}).get(
                            "arguments", ""
                        )

                    text = delta.get("content", "")
                    if text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
                        collected += text
                        live.update(Padding(_md(collected), (1, 2)))

    # ── handle tool calls ──
    if tool_calls_acc:
        # build the assistant message with tool_calls
        assistant_msg: dict = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in sorted(tool_calls_acc.values(), key=lambda x: x["id"])
            ],
        }
        if collected:
            assistant_msg["content"] = collected
        messages.append(assistant_msg)

        # execute each tool and append results
        for tc in sorted(tool_calls_acc.values(), key=lambda x: x["id"]):
            console.print(
                f"  [dim]⟳ calling[/] [bold cyan]{tc['name']}[/]"
                f"[dim]({json.loads(tc['arguments']).get('query', '')})[/]"
            )
            result = execute_tool(tc["name"], tc["arguments"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )

        # re-stream with tool results
        body["messages"] = messages
        return _do_stream(body, headers, messages, console, _md)

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

    return collected, used_model, provider_name, usage


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
        "tools": TOOLS,
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

    collected, _, _, _ = _do_stream(body, headers, messages, console, _md)

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

        collected, _, _, _ = _do_stream(body, headers, messages, console, _md)
