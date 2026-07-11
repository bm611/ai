import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ai.client import ensemble_prompt, stream_prompt
from ai.config import CONFIG_FILE, load_config, save_config

POPULAR_MODELS = [
    ("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash"),
    ("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro"),
    ("google/gemma-4-31b-it", "Gemma 4 31B Instruct"),
    ("google/gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash Lite Preview"),
    ("mimo/mimo-v2.5", "MiMo V2.5 (https://api.xiaomimimo.com/v1/chat/completions)"),
    (
        "mimo/mimo-v2.5-pro",
        "MiMo V2.5 Pro (https://api.xiaomimimo.com/v1/chat/completions)",
    ),
]

THEME_OPTIONS = [
    ("auto", "Detect system light/dark mode (default)"),
    ("dark", "Force dark theme"),
    ("light", "Force light theme"),
    ("retro", "Green-phosphor CRT terminal theme"),
]

CONFIG_KEYS = [
    ("model", "OpenRouter model ID", "deepseek/deepseek-v4-flash"),
    ("theme", "Color theme for output", "auto"),
    ("provider", "OpenRouter provider routing (JSON)", '{"order": ["DeepInfra"]}'),
    (
        "ensemble_models",
        "Models queried in parallel for -e (JSON list or comma-separated)",
        '["deepseek/deepseek-v4-flash", "google/gemini-3.1-flash-lite-preview"]',
    ),
    ("consensus_model", "Model that consolidates ensemble answers", "deepseek/deepseek-v4-pro"),
]

VALID_THEMES = {theme for theme, _ in THEME_OPTIONS}


def _read_file(path_str: str) -> str:
    p = Path(path_str).resolve()
    if not p.is_file():
        Console(stderr=True).print(f"[red bold]Error:[/] file not found: {p}")
        sys.exit(1)
    return f'<file path="{p.name}">\n{p.read_text()}\n</file>'


def _ensemble_models(cfg: dict) -> list[str]:
    """Return the configured ensemble models as a list of model ids."""
    raw = cfg.get("ensemble_models") or []
    if isinstance(raw, str):
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(raw)


def _marked_table(rows: list[tuple[str, str]], current: str) -> Table:
    """Build a two-column (id, description) table with an 'active' marker."""
    table = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    table.add_column("Name", style="bold")
    table.add_column("Description", style="dim")
    table.add_column("", justify="right")
    for item_id, desc in rows:
        marker = "[green]● active[/]" if item_id == current else ""
        table.add_row(item_id, desc, marker)
    return table


class PromptGroup(click.Group):
    """Custom group that treats unrecognised commands as prompt text.

    When the first positional arg isn't a known subcommand (e.g. a quoted
    prompt string), push everything into ctx.args so the group callback
    receives it, instead of letting click error with "No such command".
    """

    def parse_args(self, ctx, args):
        # Peek at the first non-option arg. If it's not a registered
        # subcommand, skip subcommand resolution entirely.
        positional = [a for a in args if not a.startswith("-")]
        first = positional[0] if positional else None
        if first is not None and first not in self.commands:
            # Let click parse only the group's own options (-m, -f, etc.)
            super(click.Group, self).parse_args(ctx, args)
            # Stash all positional args (the prompt) into ctx.args
            ctx.args = positional
            ctx.protected_args = []
            ctx.invoked_subcommand = None
            return positional
        return super().parse_args(ctx, args)


@click.group(
    cls=PromptGroup,
    invoke_without_command=True,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option(
    "-m", "--model", default=None, help="Override the model for this request."
)
@click.option(
    "-f",
    "--file",
    "files",
    multiple=True,
    type=click.Path(),
    help="Attach file(s) as context. Repeatable: -f a.py -f b.js",
)
@click.option(
    "-c",
    "--chat",
    "chat",
    is_flag=True,
    default=False,
    help="Enter chat mode to ask follow-up questions after the initial response.",
)
@click.option(
    "-e",
    "--ensemble",
    "ensemble",
    is_flag=True,
    default=False,
    help="Query multiple models in parallel, then consolidate into one answer.",
)
@click.pass_context
def cli(ctx, model, files, chat, ensemble):
    """AI CLI — query LLMs via OpenRouter from your terminal.

    \b
    Examples:
        ai "explain quicksort in python"
        ai -f main.py "explain this code"
        ai -f src/a.py -f src/b.py "find the bug"
        ai -m anthropic/claude-sonnet-4 "write a haiku"
        ai -c "explain quicksort"          # chat mode: ask follow-ups
        ai -e "what is the best sorting algorithm?"  # ensemble: query 2 models + consolidate
        pbpaste | ai "review this code"
        cat log.txt | ai "summarize errors"

    \b
    Configuration:
        ai config show              Show current config
        ai config set key value     Update a setting
        ai config models            List popular models
        ai config themes            List available themes
    """
    if ctx.invoked_subcommand is not None:
        return

    prompt = ctx.args

    # build prompt parts
    parts: list[str] = []

    # read from stdin if piped
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            parts.append(f"<stdin>\n{stdin_text}\n</stdin>")

    # attach files
    for f in files:
        parts.append(_read_file(f))

    # the actual prompt text
    prompt_text = " ".join(prompt).strip()
    if prompt_text:
        parts.append(prompt_text)

    if not parts:
        click.echo(ctx.get_help())
        return

    cfg = load_config()
    provider = cfg.get("provider")
    theme = cfg.get("theme", "auto")
    full_prompt = "\n\n".join(parts)

    if ensemble:
        models = _ensemble_models(cfg)
        consensus_model = model or cfg.get("consensus_model", "deepseek/deepseek-v4-pro")
        ensemble_prompt(full_prompt, models, consensus_model, provider, theme)
        return

    use_model = model or cfg["model"]
    stream_prompt(full_prompt, use_model, provider, theme, chat=chat)


# ── config subcommand group ──────────────────────────────────────────


@cli.group()
def config():
    """View or update configuration."""


@config.command("show")
def config_show():
    """Show current config values and file location."""
    cfg = load_config()
    console = Console()
    console.print(f"\n[dim]Config file:[/] {CONFIG_FILE}\n")

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")

    descriptions = {k: desc for k, desc, _ in CONFIG_KEYS}
    for key, value in cfg.items():
        display = str(value) if value is not None else "[dim]not set[/]"
        table.add_row(key, display, descriptions.get(key, ""))

    console.print(table)
    console.print()


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a config value.

    \b
    Keys:
        model            OpenRouter model ID
        theme            auto | dark | light | retro
        provider         Provider routing as JSON
        ensemble_models  Models for -e (JSON list or comma-separated)
        consensus_model  Model that consolidates ensemble answers

    \b
    Examples:
        ai config set model anthropic/claude-sonnet-4
        ai config set theme dark
        ai config set provider '{"order": ["DeepInfra"]}'
        ai config set ensemble_models 'deepseek/deepseek-v4-flash,x-ai/grok-4'
        ai config set consensus_model deepseek/deepseek-v4-pro
    """
    valid_keys = {k for k, _, _ in CONFIG_KEYS}
    if key not in valid_keys:
        Console(stderr=True).print(
            f"[red bold]Error:[/] unknown key [bold]{key}[/]. "
            f"Valid keys: {', '.join(sorted(valid_keys))}"
        )
        sys.exit(1)

    cfg = load_config()
    if key == "theme" and value not in VALID_THEMES:
        Console(stderr=True).print(
            f"[red bold]Error:[/] unknown theme [bold]{value}[/]. "
            f"Valid themes: {', '.join(sorted(VALID_THEMES))}"
        )
        sys.exit(1)

    if key == "provider":
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {"order": [value]}

    if key == "ensemble_models":
        try:
            parsed = json.loads(value)
            value = parsed if isinstance(parsed, list) else [str(parsed)]
        except json.JSONDecodeError:
            value = [m.strip() for m in value.split(",") if m.strip()]

    cfg[key] = value
    save_config(cfg)
    Console().print(f"[green]✓[/] [bold]{key}[/] → {value}")


@config.command("models")
def config_models():
    """List popular OpenRouter models."""
    console = Console()
    cfg = load_config()
    current = cfg.get("model", "")

    console.print(
        "\n[bold]Popular models[/] [dim](use OpenRouter IDs, or MiMo V2.5 with MIMO_API_KEY)[/]\n"
    )
    console.print(_marked_table(POPULAR_MODELS, current))
    console.print(f"\n[dim]Set with:[/]  ai config set model <model-id>")
    console.print(f"[dim]Browse all:[/] https://openrouter.ai/models\n")
    console.print(f"[dim]MiMo API:[/] https://api.xiaomimimo.com/v1/chat/completions\n")


@config.command("themes")
def config_themes():
    """List available color themes."""
    console = Console()
    cfg = load_config()
    current = cfg.get("theme", "auto")

    console.print("\n[bold]Available themes[/]\n")
    console.print(_marked_table(THEME_OPTIONS, current))
    console.print(f"\n[dim]Set with:[/]  ai config set theme <name>\n")
