import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ai.config import load_config, save_config, CONFIG_FILE
from ai.client import stream_prompt

POPULAR_MODELS = [
    ("x-ai/grok-4.1-fast", "Grok 4.1 Fast"),
    ("google/gemma-4-31b-it", "Gemma 4 31B Instruct"),
    ("minimax/minimax-m2.7", "MiniMax M2.7"),
    ("google/gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash Lite Preview"),
]

THEME_OPTIONS = [
    ("auto", "Detect system light/dark mode (default)"),
    ("dark", "Force dark theme (monokai code highlighting)"),
    ("light", "Force light theme (friendly code highlighting)"),
    ("monokai", "Pygments: monokai (dark)"),
    ("dracula", "Pygments: dracula (dark)"),
    ("one-dark", "Pygments: one-dark (dark)"),
    ("nord", "Pygments: nord (dark)"),
    ("github-dark", "Pygments: github-dark (dark)"),
    ("friendly", "Pygments: friendly (light)"),
    ("tango", "Pygments: tango (light)"),
    ("paraiso-light", "Pygments: paraiso-light (light)"),
    ("solarized-light", "Pygments: solarized-light (light)"),
]

CONFIG_KEYS = [
    ("model", "OpenRouter model ID", "openai/gpt-4o-mini"),
    ("theme", "Color theme for output", "auto"),
    ("provider", "OpenRouter provider routing (JSON)", '{"order": ["DeepInfra"]}'),
]


def _read_file(path_str: str) -> str:
    p = Path(path_str).resolve()
    if not p.is_file():
        Console(stderr=True).print(f"[red bold]Error:[/] file not found: {p}")
        sys.exit(1)
    return f'<file path="{p.name}">\n{p.read_text()}\n</file>'


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
@click.pass_context
def cli(ctx, model, files, chat):
    """AI CLI — query LLMs via OpenRouter from your terminal.

    \b
    Examples:
        ai "explain quicksort in python"
        ai -f main.py "explain this code"
        ai -f src/a.py -f src/b.py "find the bug"
        ai -m anthropic/claude-sonnet-4 "write a haiku"
        ai -c "explain quicksort"          # chat mode: ask follow-ups
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
    use_model = model or cfg["model"]
    provider = cfg.get("provider")

    stream_prompt(
        "\n\n".join(parts), use_model, provider, cfg.get("theme", "auto"), chat=chat
    )


# ── config subcommand group ──────────────────────────────────────────


@cli.group()
def config():
    """View or update configuration."""


@config.command("show")
def config_show():
    """Show current config values and file location."""
    import json

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
        model       OpenRouter model ID
        theme       auto | dark | light | <pygments-theme>
        provider    Provider routing as JSON

    \b
    Examples:
        ai config set model anthropic/claude-sonnet-4
        ai config set theme dracula
        ai config set provider '{"order": ["DeepInfra"]}'
    """
    import json

    valid_keys = {k for k, _, _ in CONFIG_KEYS}
    if key not in valid_keys:
        Console(stderr=True).print(
            f"[red bold]Error:[/] unknown key [bold]{key}[/]. "
            f"Valid keys: {', '.join(sorted(valid_keys))}"
        )
        sys.exit(1)

    cfg = load_config()
    if key == "provider":
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {"order": [value]}

    cfg[key] = value
    save_config(cfg)
    Console().print(f"[green]✓[/] [bold]{key}[/] → {value}")


@config.command("models")
def config_models():
    """List popular OpenRouter models."""
    console = Console()
    cfg = load_config()
    current = cfg.get("model", "")

    console.print("\n[bold]Popular models[/] [dim](use any OpenRouter model ID)[/]\n")

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    table.add_column("Model", style="bold")
    table.add_column("Description", style="dim")
    table.add_column("", justify="right")

    for model_id, desc in POPULAR_MODELS:
        marker = "[green]● active[/]" if model_id == current else ""
        table.add_row(model_id, desc, marker)

    console.print(table)
    console.print(f"\n[dim]Set with:[/]  ai config set model <model-id>")
    console.print(f"[dim]Browse all:[/] https://openrouter.ai/models\n")


@config.command("themes")
def config_themes():
    """List available color themes."""
    console = Console()
    cfg = load_config()
    current = cfg.get("theme", "auto")

    console.print("\n[bold]Available themes[/]\n")

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    table.add_column("Theme", style="bold")
    table.add_column("Description", style="dim")
    table.add_column("", justify="right")

    for theme_id, desc in THEME_OPTIONS:
        marker = "[green]● active[/]" if theme_id == current else ""
        table.add_row(theme_id, desc, marker)

    console.print(table)
    console.print(f"\n[dim]Set with:[/]  ai config set theme <name>")
    console.print(f"[dim]Any Pygments theme works:[/] https://pygments.org/styles/\n")
