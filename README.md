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
ai -f app.py "explain this code"
ai -f src/a.py -f src/b.py "find the bug"

# Override the model
ai -m anthropic/claude-sonnet-4 "write a haiku"

# Pipe from stdin
pbpaste | ai "review this code"
cat log.txt | ai "summarize errors"

# Quiet / script-friendly output
ai --no-banner "explain quicksort"
ai --plain "..." > response.md     # force raw markdown output
```

## Output modes

Interactive `ai` opens a full-screen chat with a compact header, scrollable
conversation, persistent input, and a muted model/status line. Type a message
and press Enter to send. The status line shows the current request stage and
elapsed time.

Typing `/` opens a command palette above the input. Arrow keys move through the
matching commands, Tab (or →) completes the highlighted one, Enter runs it, and
Escape dismisses the menu. A dimmed inline completion previews the rest of a
command as you type.

| Command | Description |
|---|---|
| `/help` | List commands and keyboard shortcuts |
| `/new` | Start a new conversation (resets context) |
| `/clear` | Clear the screen, keep conversation context |
| `/cancel` | Stop the current request |
| `/quit` | Exit and return to the shell (also `/exit`, `q`, Ctrl+C, Ctrl+D) |
Answers stream in place, appending block by block so the screen stays steady
instead of redrawing the whole reply on every chunk. Code blocks render in a
framed panel labelled with their language.
Conversation history is kept for this session only.

- `--no-chat` returns a single answer in the regular terminal.
- Piped input, redirected output, and `--plain` run once without the TUI.
- Redirected output and `--plain` produce raw Markdown.
- `--no-banner` hides the heading.

The full-screen layout follows the configured theme (`dark`, `light`, or
`retro`), which also drives syntax highlighting inside code blocks.

## Configuration

```bash
ai config show              # Show current config and file location
ai config set model inception/mercury-3
ai config set theme dracula
ai config set provider '{"order": ["DeepInfra"]}'
ai config models            # List popular models
ai config themes             # List available themes
```

Config is stored at `~/.config/ai-cli/config.json`.

| Key | Description | Default |
|---|---|---|
| `model` | OpenRouter model ID | `inception/mercury-3` |
| `theme` | `auto`, `dark`, `light`, or any Pygments style | `auto` |
| `provider` | OpenRouter provider routing (JSON) | not set |

## Flags

| Flag | Description |
|---|---|
| `-m, --model` | Override the model for this request |
| `-f, --file` | Attach file(s) as context (repeatable) |
| `-c, --chat / --no-chat` | Follow-up chat in the terminal (enabled by default) |
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
