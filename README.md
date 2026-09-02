# ai

Query LLMs via [OpenRouter](https://openrouter.ai) or the direct MiMo API from your terminal.

## Install

```bash
go install github.com/bm611/ai@latest
```

Or build the current checkout:

```bash
go build -trimpath -o ai .
```

## Setup

Set the key for the API you use:

```bash
export OPENROUTER_API_KEY="..."
export MIMO_API_KEY="..." # only needed for MiMo models
export EXA_API_KEY="..."  # only needed when a model uses web search
```

## Usage

```bash
# One-shot prompt
ai "explain quicksort in Go"

# Attach files as context
ai -f main.go "explain this code"
ai -f cli.go -f client.go "find the bug"

# Override the model
ai -m anthropic/claude-sonnet-4 "write a haiku"

# Pipe from stdin
pbpaste | ai "review this code"
cat log.txt | ai "summarize errors"

# Chat mode
ai -c "explain quicksort"

# Query multiple models in parallel, then consolidate
ai -e "what is the best approach to rate limiting?"

# Script-friendly output
ai --no-banner "explain quicksort"
ai --plain "..." > response.md
```

## Output modes

Interactive output includes the themed ASCII banner, live token streaming, a waiting indicator, and a stats footer with tokens, time to first token, throughput, and elapsed time.

- Piped or redirected stdout automatically emits raw markdown without banner, styling, or stats.
- `--plain` forces raw output.
- `--no-banner` keeps interactive streaming and stats but skips the logo.

## Ensemble mode

`-e/--ensemble` queries the configured models concurrently, prints each successful answer, then asks the consensus model to synthesize a final answer.

```bash
ai config set ensemble_models 'deepseek/deepseek-v4-flash,google/gemini-3.1-flash-lite-preview'
ai config set consensus_model deepseek/deepseek-v4-pro
```

## Configuration

```bash
ai config show
ai config set model z-ai/glm-5.3-flash
ai config set theme retro
ai config set provider '{"order": ["DeepInfra"]}'
ai config models
ai config themes
```

Configuration remains compatible with the Python version and is stored at `~/.config/ai-cli/config.json`.

| Key | Description | Default |
|---|---|---|
| `model` | OpenRouter or MiMo model ID | `z-ai/glm-5.3-flash` |
| `theme` | `auto`, `dark`, `light`, or `retro` | `auto` |
| `provider` | OpenRouter provider routing JSON | not set |
| `ensemble_models` | Models queried in parallel for `-e` | DeepSeek V4 Flash, Gemini 3.1 Flash Lite |
| `consensus_model` | Model that consolidates ensemble answers | `deepseek/deepseek-v4-pro` |

## Flags

| Flag | Description |
|---|---|
| `-m, --model` | Override the model for this request |
| `-f, --file` | Attach files as context; repeatable |
| `-c, --chat` | Ask follow-up questions after the first response |
| `-e, --ensemble` | Query multiple models concurrently, then consolidate |
| `--no-banner` | Skip the ASCII logo |
| `--plain` | Raw markdown output; automatic when stdout is piped |

## Development

```bash
go test ./...
go vet ./...
go build -trimpath -ldflags="-s -w" -o bin/ai .
go test -run='^$' -bench='^BenchmarkCLIStartup$' -benchtime=30x
```

The implementation uses the Go standard library plus `golang.org/x/term` for reliable terminal detection.

## License

MIT
