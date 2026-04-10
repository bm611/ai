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
# One-shot prompt
ai "explain quicksort in python"

# Attach files as context
ai -f main.py "explain this code"
ai -f src/a.py -f src/b.py "find the bug"

# Override the model
ai -m anthropic/claude-sonnet-4 "write a haiku"

# Pipe from stdin
pbpaste | ai "review this code"
cat log.txt | ai "summarize errors"
```

## Configuration

```bash
ai config show              # Show current config and file location
ai config set model x-ai/glm-5.1
ai config set theme dracula
ai config set provider '{"order": ["DeepInfra"]}'
ai config models            # List popular models
ai config themes             # List available themes
```

Config is stored at `~/.config/ai-cli/config.json`.

| Key | Description | Default |
|---|---|---|
| `model` | OpenRouter model ID | `x-ai/grok-4.1-fast` |
| `theme` | `auto`, `dark`, `light`, or any Pygments style | `auto` |
| `provider` | OpenRouter provider routing (JSON) | not set |

## Themes

Theme detection follows the system light/dark mode on macOS. On Linux or when set to `dark`/`light`, the matching built-in theme is used. Any [Pygments style](https://pygments.org/styles/) name also works for code highlighting.

## Dependencies

- [click](https://click.palletsprojects.com/) — CLI framework
- [httpx](https://www.python-httpx.org/) — HTTP client with streaming
- [rich](https://rich.readthedocs.io/) — terminal formatting and markdown rendering

## License

MIT