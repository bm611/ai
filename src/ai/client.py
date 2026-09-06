import asyncio
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
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from ai.tools import TOOLS, execute_tool

# When True (non-TTY stdout or --plain), suppress the banner, spinner,
# colored panels and stats bar so output is pipe-friendly raw markdown.
_PLAIN = False
_SHOW_BANNER = True


def set_plain(plain: bool) -> None:
    global _PLAIN
    _PLAIN = plain


def set_show_banner(show: bool) -> None:
    global _SHOW_BANNER
    _SHOW_BANNER = show

# Per-mode markdown styling, code-highlight theme, and banner colour.
MODES = {
    "dark": {
        "code_theme": "monokai",
        "banner": "bold magenta",
        "colors": {
            "h1": "bright_magenta",
            "h2": "bright_cyan",
            "h3": "bright_green",
            "h4": "bright_yellow",
            "link": "bright_blue",
            "item": "bright_cyan",
            "quote": "magenta",
            "code": "bright_green on grey11",
            "hr": "bright_magenta",
        },
    },
    "light": {
        "code_theme": "friendly",
        "banner": "bold dark_magenta",
        "colors": {
            "h1": "dark_magenta",
            "h2": "dark_cyan",
            "h3": "green",
            "h4": "dark_orange",
            "link": "blue",
            "item": "dark_cyan",
            "quote": "dark_magenta",
            "code": "dark_green on grey93",
            "hr": "dark_magenta",
        },
    },
    "retro": {
        "code_theme": "rrt",
        "banner": "bold green1",
        "colors": {
            "h1": "green1",
            "h2": "chartreuse1",
            "h3": "green_yellow",
            "h4": "yellow",
            "link": "bright_yellow",
            "item": "green1",
            "quote": "green3",
            "code": "bold green1 on grey0",
            "hr": "green1",
        },
    },
}


def _markdown_theme(colors: dict) -> Theme:
    return Theme(
        {
            "markdown.h1": f"bold {colors['h1']}",
            "markdown.h2": f"bold {colors['h2']}",
            "markdown.h3": f"bold {colors['h3']}",
            "markdown.h4": f"bold {colors['h4']}",
            "markdown.link": f"{colors['link']} underline",
            "markdown.item.bullet": colors["item"],
            "markdown.item.number": colors["item"],
            "markdown.block_quote": f"italic {colors['quote']}",
            "markdown.code": colors["code"],
            "markdown.hr": colors["hr"],
        }
    )


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MIMO_URL = "https://api.xiaomimimo.com/v1/chat/completions"

# Human-readable hints for common HTTP failures.
_ERROR_HINTS = {
    400: "Request rejected — check the model ID and parameters.",
    401: "Authentication failed — check your API key environment variable.",
    402: "Out of credits — top up your account and retry.",
    403: "Access denied for this key/model.",
    404: "Model not found — run `ai config models` for popular IDs.",
    408: "Upstream timeout — try again.",
    429: "Rate limited — wait a moment or route to another provider.",
    502: "Bad gateway — the upstream provider may be down.",
    503: "Service unavailable — try again shortly.",
}


def _http_error_detail(status_code: int, body: str) -> tuple[str, str]:
    """Return (detail, hint) extracted from a failed HTTP response."""
    detail = ""
    try:
        payload = json.loads(body)
        err = payload.get("error")
        detail = (err.get("message") if isinstance(err, dict) else "") \
            or payload.get("message") or ""
    except json.JSONDecodeError:
        pass
    if not detail:
        detail = (body or "").strip()
    return detail[:300], _ERROR_HINTS.get(status_code, "")


def _print_http_error(console: Console, status_code: int, body: str) -> None:
    """Render a friendly error panel and exit."""
    detail, hint = _http_error_detail(status_code, body)
    content = Text()
    content.append(f"HTTP {status_code}", style="bold red")
    if detail:
        content.append("\n")
        content.append(detail)
    if hint:
        content.append("\n\n")
        content.append(hint, style="yellow")
    console.print(
        Panel(content, title="request failed", title_align="left", border_style="red")
    )
    sys.exit(1)


def _is_mimo_model(model: str) -> bool:
    return model.startswith("mimo/") or model.startswith("mimo-")


def _api_model_name(model: str) -> str:
    if model.startswith("mimo/"):
        return model.removeprefix("mimo/")
    return model


def _get_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        Console(stderr=True).print(
            f"[red bold]Error:[/] {env_var} environment variable is not set.\n"
            "Export it or add it to your shell profile."
        )
        sys.exit(1)
    return key


def _build_request(
    model: str,
    messages: list[dict],
    provider: dict | None,
    stream: bool,
    tools: bool = False,
) -> tuple[str, dict, dict]:
    """Build (url, headers, body) for a chat completion request."""
    if _is_mimo_model(model):
        url = MIMO_URL
        headers = {
            "api-key": _get_key("MIMO_API_KEY"),
            "Content-Type": "application/json",
        }
    else:
        url = OPENROUTER_URL
        headers = {
            "Authorization": f"Bearer {_get_key('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-cli",
            "X-Title": "ai-cli",
        }

    body: dict = {
        "model": _api_model_name(model),
        "messages": messages,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if tools and not _is_mimo_model(model):
        body["tools"] = TOOLS
    if provider:
        body["provider"] = provider
    return url, headers, body


def _resolve_theme(theme_setting: str) -> tuple[Theme, str, str]:
    from ai.config import is_dark_mode

    if theme_setting == "auto":
        mode = "dark" if is_dark_mode() else "light"
    elif theme_setting in MODES:
        mode = theme_setting
    else:
        mode = "dark"

    spec = MODES[mode]
    return _markdown_theme(spec["colors"]), spec["code_theme"], mode


ToolCall = dict[int, dict]  # index -> {id, name, arguments}


def _accumulate_delta(delta: dict, acc: ToolCall) -> str:
    """Merge tool_call deltas and return text content."""
    for tc in delta.get("tool_calls") or []:
        idx = tc["index"]
        acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        acc[idx]["id"] = tc.get("id") or acc[idx]["id"]
        acc[idx]["name"] = tc.get("function", {}).get("name") or acc[idx]["name"]
        acc[idx]["arguments"] += tc.get("function", {}).get("arguments", "")
    return delta.get("content") or ""


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
    ttft: float | None = None,
) -> None:
    if _PLAIN:
        return
    tok_per_sec = real_tokens / gen_time if gen_time > 0 else 0
    parts = [("✦ ", "bold magenta"), (used_model, "bold cyan")]
    if provider_name:
        parts += [("  via ", "dim"), (provider_name, "bold yellow")]
    parts += [
        ("  │  ", "dim"),
        (f"{real_tokens}", "bold"),
        (" tokens", "dim"),
    ]
    if ttft is not None:
        parts += [
            ("  │  ", "dim"),
            (f"{ttft:.2f}", "bold"),
            ("s to first token", "dim"),
        ]
    parts += [
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


# Re-parsing the full markdown on every chunk gets expensive on long answers;
# throttle re-renders once the response grows past this many characters.
_STREAM_RENDER_THRESHOLD = 800


def _do_stream(
    url: str,
    body: dict,
    headers: dict,
    messages: list[dict],
    console: Console,
    _md,
) -> tuple[str, str, str, dict]:
    """Stream a single chat completion, handling tool calls automatically."""
    collected, used_model, first_token_time, token_count = "", body["model"], None, 0
    provider_name, tool_calls_acc = "", {}
    request_start = time.perf_counter()

    with httpx.Client(timeout=120) as http:
        with http.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                resp.read()
                _print_http_error(console, resp.status_code, resp.text)

            usage: dict = {}
            if _PLAIN:
                # Raw output: stream text verbatim as it arrives.
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line.removeprefix("data: ").strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    used_model = chunk.get("model", used_model)
                    if "provider" in chunk:
                        provider_name = chunk["provider"]
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    text = _accumulate_delta(delta, tool_calls_acc)
                    if text:
                        collected += text
                        print(text, end="", flush=True)
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                if collected:
                    print()
            else:
                # Transient status while waiting for the first token, so the
                # UI isn't just a blank box during model warm-up.
                live_ctx = Live(
                    Text("  thinking…", style="dim italic"),
                    console=console,
                    refresh_per_second=8,
                )
                with live_ctx as live:
                    last_len = 0
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line.removeprefix("data: ").strip()
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        used_model = chunk.get("model", used_model)
                        provider_name = chunk.get("provider", provider_name)
                        if chunk.get("usage"):
                            usage = chunk["usage"]

                        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                        text = _accumulate_delta(delta, tool_calls_acc)
                        if text and first_token_time is None:
                            first_token_time = time.perf_counter()
                        if text:
                            token_count += 1
                            collected += text
                            grown = len(collected) - last_len
                            if (
                                len(collected) < _STREAM_RENDER_THRESHOLD
                                or grown >= _STREAM_RENDER_THRESHOLD // 4
                            ):
                                live.update(Padding(_md(collected), (1, 2)))
                                last_len = len(collected)

    t_start = first_token_time or time.perf_counter()
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    gen_time = t_end - (first_token_time or t_end)
    ttft = (first_token_time - request_start) if first_token_time else None

    if tool_calls_acc:
        _handle_tool_calls(tool_calls_acc, messages, console)
        body["messages"] = messages
        return _do_stream(url, body, headers, messages, console, _md)

    real_tokens = usage.get("completion_tokens", token_count)
    _print_stats(console, used_model, provider_name, elapsed, gen_time, real_tokens, ttft)
    return collected, used_model, provider_name, usage


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


def _make_renderer(theme: str) -> tuple[Console, callable, str]:
    """Return a themed Console, an _md(text) -> Markdown renderer, and banner style."""
    md_theme, code_theme, mode = _resolve_theme(theme)

    def _md(text: str) -> Markdown:
        return Markdown(_prep_math(text), code_theme=code_theme)

    return Console(theme=md_theme), _md, MODES[mode]["banner"]


def _print_banner(console: Console, style: str = "bold magenta") -> None:
    if not _SHOW_BANNER or _PLAIN:
        return
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
        console.print(Text(line, style=style))
    console.print(Rule(style="dim"))


def stream_prompt(
    prompt: str,
    model: str,
    provider: dict | None = None,
    theme: str = "auto",
    chat: bool = False,
) -> None:
    from datetime import date

    messages: list[dict] = [
        {
            "role": "system",
            "content": f"Today's date is {date.today().isoformat()}.",
        },
        {"role": "user", "content": prompt},
    ]

    url, headers, body = _build_request(model, messages, provider, stream=True, tools=True)

    console, _md, banner_style = _make_renderer(theme)
    _print_banner(console, banner_style)

    collected, _, _, _ = _do_stream(url, body, headers, messages, console, _md)

    if not chat:
        return

    while True:
        try:
            followup = Prompt.ask(
                "\n[bold cyan]Follow-up[/] [dim](or [bold]q[/] to quit)[/]"
            )
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

        collected, _, _, _ = _do_stream(url, body, headers, messages, console, _md)


# ── ensemble mode ────────────────────────────────────────────────────


def _short(model: str) -> str:
    """Short display name for a model id."""
    return model.split("/", 1)[-1]


async def _query_model(model: str, messages: list[dict], provider: dict | None, state: dict) -> None:
    """Run one non-streaming completion, recording progress in `state`."""
    url, headers, body = _build_request(model, messages, provider, stream=False)
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=180) as http:
            resp = await http.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                state["status"] = "error"
                detail, hint = _http_error_detail(resp.status_code, resp.text)
                state["error"] = (
                    f"HTTP {resp.status_code}"
                    + (f": {detail}" if detail else "")
                    + (f" — {hint}" if hint and not detail else "")
                )
                return
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            state["text"] = (choice.get("message") or {}).get("content") or ""
            state["usage"] = data.get("usage") or {}
            state["resolved"] = data.get("model", model)
            state["status"] = "done"
    except Exception as exc:  # noqa: BLE001 - surface any network/parse error in UI
        state["status"] = "error"
        state["error"] = str(exc)
    finally:
        state["elapsed"] = time.perf_counter() - start


def _ensemble_status_panel(
    states: list[dict], title: str, started: float | None = None
) -> Panel:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=2)
    grid.add_column(width=10)
    grid.add_column()
    for i, st in enumerate(states, 1):
        status = st["status"]
        name = _short(st["model"])
        if status == "responding":
            icon = Spinner("dots", style="cyan")
            suffix = (
                f" · {time.perf_counter() - started:.0f}s"
                if started is not None
                else ""
            )
            detail = Text(f"responding{suffix}…", style="dim")
        elif status == "done":
            tokens = st.get("usage", {}).get("completion_tokens", 0)
            icon = Text("✓", style="bold green")
            detail = Text(
                f"done · {tokens} tokens · {st.get('elapsed', 0):.1f}s", style="green"
            )
        else:
            icon = Text("✗", style="bold red")
            detail = Text(st.get("error", "error"), style="red")
        grid.add_row(
            icon,
            Text(name, style="bold cyan"),
            detail,
        )
    return Panel(grid, title=title, title_align="left", border_style="dim", padding=(1, 2))


async def _run_ensemble(
    messages: list[dict],
    models: list[str],
    provider: dict | None,
    console: Console,
) -> list[dict]:
    states = [{"model": m, "status": "responding", "text": ""} for m in models]
    tasks = [
        asyncio.create_task(_query_model(m, messages, provider, states[i]))
        for i, m in enumerate(models)
    ]

    if _PLAIN:
        # No live panel — just wait quietly for all models to finish.
        await asyncio.gather(*tasks)
        return states

    title = f"⚡ Querying {len(models)} models in parallel"
    started = time.perf_counter()
    with Live(
        _ensemble_status_panel(states, title), console=console, refresh_per_second=12
    ) as live:
        while not all(t.done() for t in tasks):
            live.update(_ensemble_status_panel(states, title, started))
            await asyncio.sleep(0.08)
        live.update(_ensemble_status_panel(states, title, started))
    await asyncio.gather(*tasks)
    return states


def ensemble_prompt(
    prompt: str,
    models: list[str],
    consensus_model: str,
    provider: dict | None = None,
    theme: str = "auto",
) -> None:
    from datetime import date

    if len(models) < 2:
        Console(stderr=True).print(
            "[red bold]Error:[/] ensemble mode needs at least 2 models "
            "(set with [bold]ai config set ensemble_models[/])."
        )
        sys.exit(1)

    base_messages = [
        {"role": "system", "content": f"Today's date is {date.today().isoformat()}."},
        {"role": "user", "content": prompt},
    ]

    console, _md, banner_style = _make_renderer(theme)
    _print_banner(console, banner_style)

    states = asyncio.run(_run_ensemble(base_messages, models, provider, console))

    succeeded = [s for s in states if s["status"] == "done" and s["text"].strip()]
    if not succeeded:
        console.print("[red bold]All models failed to respond.[/]")
        sys.exit(1)

    # ── per-model summaries ──
    if not _PLAIN:
        console.print()
    for i, st in enumerate(states, 1):
        if st["status"] != "done" or not st["text"].strip():
            continue
        header = f"Model {i} · {_short(st['model'])}"
        if _PLAIN:
            console.print(f"\n## {header}\n")
            console.print(st["text"].strip())
        else:
            usage = st.get("usage") or {}
            tokens = usage.get("completion_tokens", 0)
            elapsed = st.get("elapsed", 0)
            subtitle = (
                f"{tokens} tokens · {elapsed:.1f}s"
                if tokens or elapsed
                else None
            )
            console.print(
                Panel(
                    _md(st["text"].strip()),
                    title=header,
                    subtitle=subtitle,
                    subtitle_align="right",
                    title_align="left",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

    # ── consolidate ──
    if not _PLAIN:
        console.print()
        console.print(
            Padding(
                Text.assemble(
                    ("✦ ", "bold magenta"),
                    ("Consolidating with ", "dim"),
                    (_short(consensus_model), "bold cyan"),
                    ("…", "dim"),
                ),
                (0, 2),
            )
        )

    answers = "\n\n".join(
        f'<answer model="Model {i}">\n{s["text"].strip()}\n</answer>'
        for i, s in enumerate(states, 1)
        if s["status"] == "done" and s["text"].strip()
    )
    consensus_messages = [
        {
            "role": "system",
            "content": (
                "You are an expert editor. You are given a user's prompt and several "
                "independent answers from different AI models. Synthesize a single, "
                "best possible response: combine the strongest points, resolve any "
                "contradictions in favor of correctness, drop errors, and fill gaps. "
                "Do not mention the individual models, that multiple answers exist, "
                "or that you are consolidating. Just give the final answer."
            ),
        },
        {
            "role": "user",
            "content": f"User prompt:\n{prompt}\n\nCandidate answers:\n{answers}",
        },
    ]

    url, headers, body = _build_request(
        consensus_model, consensus_messages, provider, stream=True
    )
    _do_stream(url, body, headers, consensus_messages, console, _md)
