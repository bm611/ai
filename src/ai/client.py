import json
import os
import re
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
    else:
        mode = theme_setting

    md_theme = DARK_THEME if mode == "dark" else LIGHT_THEME
    return md_theme, CODE_THEMES[mode]


ToolCall = dict[int, dict]  # index -> {id, name, arguments}


def _accumulate_delta(delta: dict, acc: ToolCall) -> str:
    """Merge tool_call deltas and return text content."""
    for tc in delta.get("tool_calls", []):
        idx = tc["index"]
        acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        acc[idx]["id"] = tc.get("id") or acc[idx]["id"]
        acc[idx]["name"] = tc.get("function", {}).get("name") or acc[idx]["name"]
        acc[idx]["arguments"] += tc.get("function", {}).get("arguments", "")
    return delta.get("content", "")


def _handle_tool_calls(
    tc_acc: ToolCall, messages: list[dict], console: Console
) -> None:
    """Execute accumulated tool calls and append results to messages."""
    sorted_tcs = sorted(tc_acc.values(), key=lambda x: x["id"])
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in sorted_tcs
            ],
        }
    )
    for tc in sorted_tcs:
        query = json.loads(tc["arguments"]).get("query", "")
        console.print(
            f"  [dim]⟳ calling[/] [bold cyan]{tc['name']}[/][dim]({query})[/]"
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": execute_tool(tc["name"], tc["arguments"]),
            }
        )


def _print_stats(
    console: Console,
    used_model: str,
    provider_name: str,
    elapsed: float,
    gen_time: float,
    real_tokens: int,
) -> None:
    tok_per_sec = real_tokens / gen_time if gen_time > 0 else 0
    parts = [("✦ ", "bold magenta"), (used_model, "bold cyan")]
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


def _do_stream(
    body: dict,
    headers: dict,
    messages: list[dict],
    console: Console,
    _md,
) -> tuple[str, str, str, dict]:
    """Stream a single chat completion, handling tool calls automatically."""
    collected, used_model, first_token_time, token_count = "", body["model"], None, 0
    provider_name, tool_calls_acc = "", {}

    with httpx.Client(timeout=120) as http:
        with http.stream("POST", OPENROUTER_URL, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                resp.read()
                console.print(f"[red bold]Error {resp.status_code}:[/] {resp.text}")
                sys.exit(1)

            usage: dict = {}
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
                    used_model = chunk.get("model", used_model)
                    provider_name = chunk.get("provider", provider_name)
                    if "usage" in chunk:
                        usage = chunk["usage"]

                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    text = _accumulate_delta(delta, tool_calls_acc)
                    if text:
                        first_token_time = first_token_time or time.perf_counter()
                        token_count += 1
                        collected += text
                        live.update(Padding(_md(collected), (1, 2)))

    t_start = first_token_time or time.perf_counter()
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    gen_time = t_end - (first_token_time or t_end)

    if tool_calls_acc:
        _handle_tool_calls(tool_calls_acc, messages, console)
        body["messages"] = messages
        return _do_stream(body, headers, messages, console, _md)

    real_tokens = usage.get("completion_tokens", token_count)
    _print_stats(console, used_model, provider_name, elapsed, gen_time, real_tokens)
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

    from datetime import date

    messages: list[dict] = [
        {
            "role": "system",
            "content": f"Today's date is {date.today().isoformat()}.",
        },
        {"role": "user", "content": prompt},
    ]

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

    def _prep_math(text: str) -> str:
        # Wrap LaTeX display math ([ ... ]) and ($ ... $) in code blocks
        text = re.sub(
            r"\\\[(.+?)\\\]",
            lambda m: f"```\n{m.group(1).strip()}\n```",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"\$\$(.+?)\$\$",
            lambda m: f"```\n{m.group(1).strip()}\n```",
            text,
            flags=re.DOTALL,
        )
        # Wrap inline math ($...$) in inline code
        text = re.sub(
            r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
            lambda m: f"`{m.group(1).strip()}`",
            text,
        )
        return text

    def _md(text: str) -> Markdown:
        return Markdown(_prep_math(text), code_theme=code_theme)

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
