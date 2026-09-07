# ai

Query LLMs via [OpenRouter](https://openrouter.ai) from your terminal.

## Install

```bash
pip install -e .
```

## Setup

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-..."
```

## Usage

```bash
# Open full-screen chat
ai

# Start with a prompt (/quit to exit)
ai "explain quicksort in python"

# Get a single answer
ai --no-chat "explain quicksort in python"

# Attach files as context
ai -f main.py "explain this code"
ai -f src/a.py -f src/b.py "find the bug"

# Override the model
ai -m anthropic/claude-sonnet-4 "write a haiku"

# Pipe from stdin
pbpaste | ai "review this code"
cat log.txt | ai "summarize errors"

# Ensemble mode: query multiple models in parallel, then consolidate
ai -e "what is the best approach to rate limiting?"

# Quiet / script-friendly output
ai --no-banner "explain quicksort"
ai --plain "..." > response.md     # force raw markdown output
```

## Output modes

Interactive `ai` opens a full-screen chat with a compact header, scrollable
conversation, persistent input, and a muted model/status line. Type a message
and press Enter to send. `/help` shows shortcuts, `/clear` starts a new
conversation, `/cancel` stops the current request, and `/quit` (or Ctrl+C /
Ctrl+D) returns to your shell. The status line shows the current request stage
and elapsed time.
Answers stream in place, appending block by block so the screen stays steady
instead of redrawing the whole reply on every chunk. Code blocks render in a
framed panel labelled with their language.
Conversation history is kept for this session only.

- `--no-chat` returns a single answer in the regular terminal.
- Piped input, redirected output, and `--plain` run once without the TUI.
- Redirected output and `--plain` produce raw Markdown.
- `--no-banner` hides the heading.
- Ensemble mode returns one consolidated answer in the regular terminal.

The full-screen layout follows the configured theme (`dark`, `light`, or
`retro`), which also drives syntax highlighting inside code blocks.

## Ensemble mode

`-e/--ensemble` sends your prompt to several models **in parallel**, shows a live
status panel as each one responds, prints each model's answer, then uses a
"consensus" model to synthesize a single best answer.

```bash
ai -e "design a caching strategy for a read-heavy API"
ai -e -m anthropic/claude-sonnet-4 "..."   # -m overrides the consensus model
```

Configure which models are used:

```bash
ai config set ensemble_models 'deepseek/deepseek-v4-flash,google/gemini-3.1-flash-lite-preview'
ai config set consensus_model deepseek/deepseek-v4-pro
```

## Configuration

```bash
ai config show              # Show current config and file location
ai config set model z-ai/glm-5.3-flash
ai config set theme dracula
ai config set provider '{"order": ["DeepInfra"]}'
ai config models            # List popular models
ai config themes             # List available themes
```

Config is stored at `~/.config/ai-cli/config.json`.

| Key | Description | Default |
|---|---|---|
| `model` | OpenRouter model ID | `z-ai/glm-5.3-flash` |
| `theme` | `auto`, `dark`, `light`, or any Pygments style | `auto` |
| `provider` | OpenRouter provider routing (JSON) | not set |
| `ensemble_models` | Models queried in parallel for `-e` (JSON list or comma-separated) | `deepseek/deepseek-v4-flash`, `google/gemini-3.1-flash-lite-preview` |
| `consensus_model` | Model that consolidates ensemble answers | `deepseek/deepseek-v4-pro` |

## Flags

| Flag | Description |
|---|---|
| `-m, --model` | Override the model for this request |
| `-f, --file` | Attach file(s) as context (repeatable) |
| `-c, --chat / --no-chat` | Follow-up chat in the terminal (enabled by default) |
| `-e, --ensemble` | Query multiple models in parallel, then consolidate |
| `--no-banner` | Skip the compact heading |
| `--plain` | Raw markdown output, no styling (implied when stdout is piped) |

## Themes

Theme detection follows the system light/dark mode on macOS. On Linux or when set to `dark`/`light`, the matching built-in theme is used. Any [Pygments style](https://pygments.org/styles/) name also works for code highlighting.

## Dependencies

- [click](https://click.palletsprojects.com/) — CLI framework
- [httpx](https://www.python-httpx.org/) — HTTP client with streaming
- [rich](https://rich.readthedocs.io/) — terminal formatting and markdown rendering
- [Textual](https://textual.textualize.io/) — full-screen chat interface

## License

MIT
