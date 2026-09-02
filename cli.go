package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"

	"golang.org/x/term"
)

type modelOption struct {
	ID          string
	Description string
}

var popularModels = []modelOption{
	{"z-ai/glm-5.3-flash", "GLM 5.3 Flash"},
	{"deepseek/deepseek-v4-flash", "DeepSeek V4 Flash"},
	{"deepseek/deepseek-v4-pro", "DeepSeek V4 Pro"},
	{"google/gemma-4-31b-it", "Gemma 4 31B Instruct"},
	{"google/gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash Lite Preview"},
	{"mimo/mimo-v2.5", "MiMo V2.5 (direct API, not OpenRouter)"},
	{"mimo/mimo-v2.5-pro", "MiMo V2.5 Pro (direct API, not OpenRouter)"},
}

var themeOptions = []modelOption{
	{"auto", "Detect system light/dark mode (default)"},
	{"dark", "Force dark theme"},
	{"light", "Force light theme"},
	{"retro", "Green-phosphor CRT terminal theme"},
}

var configDescriptions = map[string]string{
	"model":           "OpenRouter model ID",
	"theme":           "Color theme for output",
	"provider":        "OpenRouter provider routing (JSON)",
	"ensemble_models": "Models queried in parallel for -e",
	"consensus_model": "Model that consolidates ensemble answers",
}

var configOrder = []string{
	"model",
	"provider",
	"theme",
	"ensemble_models",
	"consensus_model",
}

type cliOptions struct {
	Model      string
	Files      []string
	Chat       bool
	Ensemble   bool
	NoBanner   bool
	Plain      bool
	Help       bool
	PromptArgs []string
}

func run(args []string) int {
	if configArgs, ok := findConfigCommand(args); ok {
		return runConfig(configArgs)
	}

	opts, err := parseCLI(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 2
	}
	if opts.Help {
		printHelp(os.Stdout)
		return 0
	}

	parts := make([]string, 0, len(opts.Files)+2)
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: read stdin: %v\n", err)
			return 1
		}
		if text := strings.TrimSpace(string(data)); text != "" {
			parts = append(parts, "<stdin>\n"+text+"\n</stdin>")
		}
	}

	for _, name := range opts.Files {
		part, err := readPromptFile(name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			return 1
		}
		parts = append(parts, part)
	}
	if prompt := strings.TrimSpace(strings.Join(opts.PromptArgs, " ")); prompt != "" {
		parts = append(parts, prompt)
	}
	if len(parts) == 0 {
		printHelp(os.Stdout)
		return 0
	}

	cfg, err := loadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}
	ui := outputMode{
		Plain:      opts.Plain || !term.IsTerminal(int(os.Stdout.Fd())),
		ShowBanner: !opts.NoBanner,
		Theme:      configString(cfg, "theme", defaultTheme),
	}
	prompt := strings.Join(parts, "\n\n")
	provider := cfg["provider"]

	if opts.Ensemble {
		model := opts.Model
		if model == "" {
			model = configString(cfg, "consensus_model", defaultConsensusModel)
		}
		err = ensemblePrompt(prompt, configStrings(cfg, "ensemble_models"), model, provider, ui)
	} else {
		model := opts.Model
		if model == "" {
			model = configString(cfg, "model", defaultModel)
		}
		err = streamPrompt(prompt, model, provider, opts.Chat, ui)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}
	return 0
}

func findConfigCommand(args []string) ([]string, bool) {
	for i := 0; i < len(args); i++ {
		switch arg := args[i]; {
		case arg == "config":
			return args[i+1:], true
		case arg == "--":
			return nil, false
		case arg == "-m" || arg == "--model" || arg == "-f" || arg == "--file":
			i++
		case strings.HasPrefix(arg, "--model=") || strings.HasPrefix(arg, "--file=") ||
			(strings.HasPrefix(arg, "-m") && len(arg) > 2) ||
			(strings.HasPrefix(arg, "-f") && len(arg) > 2) ||
			arg == "-c" || arg == "--chat" || arg == "-e" || arg == "--ensemble" ||
			isBooleanFlagCluster(arg) || arg == "--no-banner" || arg == "--plain":
			continue
		default:
			return nil, false
		}
	}
	return nil, false
}

func parseCLI(args []string) (cliOptions, error) {
	var opts cliOptions
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case (arg == "-h" || arg == "--help") && len(opts.PromptArgs) == 0:
			opts.Help = true
		case arg == "-m" || arg == "--model":
			if i+1 >= len(args) {
				return opts, fmt.Errorf("%s requires a value", arg)
			}
			i++
			opts.Model = args[i]
		case strings.HasPrefix(arg, "--model="):
			opts.Model = strings.TrimPrefix(arg, "--model=")
		case strings.HasPrefix(arg, "-m") && len(arg) > 2:
			opts.Model = strings.TrimPrefix(strings.TrimPrefix(arg, "-m"), "=")
		case arg == "-f" || arg == "--file":
			if i+1 >= len(args) {
				return opts, fmt.Errorf("%s requires a value", arg)
			}
			i++
			opts.Files = append(opts.Files, args[i])
		case strings.HasPrefix(arg, "--file="):
			opts.Files = append(opts.Files, strings.TrimPrefix(arg, "--file="))
		case strings.HasPrefix(arg, "-f") && len(arg) > 2:
			opts.Files = append(opts.Files, strings.TrimPrefix(strings.TrimPrefix(arg, "-f"), "="))
		case arg == "-c" || arg == "--chat":
			opts.Chat = true
		case arg == "-e" || arg == "--ensemble":
			opts.Ensemble = true
		case isBooleanFlagCluster(arg):
			opts.Chat = opts.Chat || strings.ContainsRune(arg, 'c')
			opts.Ensemble = opts.Ensemble || strings.ContainsRune(arg, 'e')
		case arg == "--no-banner":
			opts.NoBanner = true
		case arg == "--plain":
			opts.Plain = true
		case arg == "--":
			opts.PromptArgs = append(opts.PromptArgs, args[i+1:]...)
			return opts, nil
		default:
			opts.PromptArgs = append(opts.PromptArgs, arg)
		}
	}
	return opts, nil
}

func isBooleanFlagCluster(arg string) bool {
	if len(arg) < 3 || arg[0] != '-' || arg[1] == '-' {
		return false
	}
	for _, flag := range arg[1:] {
		if flag != 'c' && flag != 'e' {
			return false
		}
	}
	return true
}

func readPromptFile(name string) (string, error) {
	path, err := filepath.Abs(name)
	if err != nil {
		return "", fmt.Errorf("resolve file %s: %w", name, err)
	}
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return "", fmt.Errorf("file not found: %s", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read file %s: %w", path, err)
	}
	return fmt.Sprintf("<file path=\"%s\">\n%s\n</file>", filepath.Base(path), data), nil
}

func printHelp(w io.Writer) {
	fmt.Fprint(w, `AI CLI — query LLMs via OpenRouter from your terminal.

Usage:
  ai [options] "prompt"
  ai config <show|set|models|themes>

Options:
  -m, --model ID    Override the model for this request
  -f, --file PATH   Attach a file as context (repeatable)
  -c, --chat        Ask follow-up questions after the first response
  -e, --ensemble    Query configured models in parallel, then consolidate
      --no-banner   Skip the ASCII logo
      --plain       Raw markdown output (automatic when stdout is piped)
  -h, --help        Show this help

Examples:
  ai "explain quicksort in Go"
  ai -f main.go "explain this code"
  pbpaste | ai "review this code"
  ai -e "compare these approaches"
`)
}

func runConfig(args []string) int {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprint(os.Stdout, `Usage: ai config <command>

Commands:
  show                 Show current config and file location
  set <key> <value>    Update a setting
  models               List popular models
  themes               List available themes
`)
		return 0
	}

	var err error
	switch args[0] {
	case "show":
		err = configShow()
	case "set":
		if len(args) != 3 {
			fmt.Fprintln(os.Stderr, "Error: usage: ai config set <key> <value>")
			return 2
		}
		err = configSet(args[1], args[2])
	case "models":
		err = configModels()
	case "themes":
		err = configThemes()
	default:
		fmt.Fprintf(os.Stderr, "Error: unknown config command %q\n", args[0])
		return 2
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return 1
	}
	return 0
}

func configShow() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	path, err := configPath()
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stdout, "Config file: %s\n\n", path)
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "KEY\tVALUE\tDESCRIPTION")
	for _, key := range configOrder {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", key, displayValue(cfg[key]), configDescriptions[key])
	}
	return tw.Flush()
}

func configSet(key, raw string) error {
	if _, ok := configDescriptions[key]; !ok {
		keys := make([]string, 0, len(configDescriptions))
		for item := range configDescriptions {
			keys = append(keys, item)
		}
		sort.Strings(keys)
		return fmt.Errorf("unknown key %q; valid keys: %s", key, strings.Join(keys, ", "))
	}
	if key == "theme" && !validTheme(raw) {
		return fmt.Errorf("unknown theme %q; valid themes: auto, dark, light, retro", raw)
	}

	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	var value any = raw
	switch key {
	case "provider":
		if err := json.Unmarshal([]byte(raw), &value); err != nil {
			value = map[string]any{"order": []string{raw}}
		}
	case "ensemble_models":
		var parsed any
		if err := json.Unmarshal([]byte(raw), &parsed); err == nil {
			if list, ok := parsed.([]any); ok {
				value = list
			} else {
				value = []string{fmt.Sprint(parsed)}
			}
		} else {
			value = splitList(raw)
		}
	}
	cfg[key] = value
	if err := saveConfig(cfg); err != nil {
		return err
	}
	fmt.Fprintf(os.Stdout, "✓ %s → %s\n", key, displayValue(value))
	return nil
}

func validTheme(theme string) bool {
	for _, option := range themeOptions {
		if option.ID == theme {
			return true
		}
	}
	return false
}

func configModels() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	fmt.Fprint(os.Stdout, "Popular models (use OpenRouter IDs, or MiMo V2.5 with MIMO_API_KEY)\n\n")
	printOptions(popularModels, configString(cfg, "model", defaultModel))
	fmt.Fprintln(os.Stdout, "\nSet with:  ai config set model <model-id>")
	fmt.Fprintln(os.Stdout, "Browse all: https://openrouter.ai/models")
	fmt.Fprintln(os.Stdout, "MiMo API:   https://api.xiaomimimo.com/v1/chat/completions")
	return nil
}

func configThemes() error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	fmt.Fprint(os.Stdout, "Available themes\n\n")
	printOptions(themeOptions, configString(cfg, "theme", defaultTheme))
	fmt.Fprintln(os.Stdout, "\nSet with:  ai config set theme <name>")
	return nil
}

func printOptions(options []modelOption, current string) {
	tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tDESCRIPTION\t")
	for _, option := range options {
		active := ""
		if option.ID == current {
			active = "● active"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", option.ID, option.Description, active)
	}
	_ = tw.Flush()
}
