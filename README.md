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

# Ensemble mode: query multiple models in parallel, then consolidate
ai -e "what is the best approach to rate limiting?"
```

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
ai config set model x-ai/glm-5.1
ai config set theme dracula
ai config set provider '{"order": ["DeepInfra"]}'
ai config models            # List popular models
ai config themes             # List available themes
```

Config is stored at `~/.config/ai-cli/config.json`.

| Key | Description | Default |
|---|---|---|
| `model` | OpenRouter model ID | `deepseek/deepseek-v4-flash` |
| `theme` | `auto`, `dark`, `light`, or any Pygments style | `auto` |
| `provider` | OpenRouter provider routing (JSON) | not set |
| `ensemble_models` | Models queried in parallel for `-e` (JSON list or comma-separated) | `deepseek/deepseek-v4-flash`, `google/gemini-3.1-flash-lite-preview` |
| `consensus_model` | Model that consolidates ensemble answers | `deepseek/deepseek-v4-pro` |

## Themes

Theme detection follows the system light/dark mode on macOS. On Linux or when set to `dark`/`light`, the matching built-in theme is used. Any [Pygments style](https://pygments.org/styles/) name also works for code highlighting.

## Dependencies

- [click](https://click.palletsprojects.com/) — CLI framework
- [httpx](https://www.python-httpx.org/) — HTTP client with streaming
- [rich](https://rich.readthedocs.io/) — terminal formatting and markdown rendering

## License

MIT
