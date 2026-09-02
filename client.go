package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"time"
)

const (
	openRouterURL = "https://openrouter.ai/api/v1/chat/completions"
	mimoURL       = "https://api.xiaomimimo.com/v1/chat/completions"
)

type outputMode struct {
	Plain      bool
	ShowBanner bool
	Theme      string
}

type chatMessage map[string]any

type apiRequest struct {
	Model         string          `json:"model"`
	Messages      []chatMessage   `json:"messages"`
	Stream        bool            `json:"stream"`
	StreamOptions map[string]bool `json:"stream_options,omitempty"`
	Tools         []any           `json:"tools,omitempty"`
	Provider      any             `json:"provider,omitempty"`
}

type usage struct {
	CompletionTokens int `json:"completion_tokens"`
}

type toolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type streamDelta struct {
	Content   string          `json:"content"`
	ToolCalls []toolCallDelta `json:"tool_calls"`
}

type streamChunk struct {
	Model    string `json:"model"`
	Provider string `json:"provider"`
	Usage    *usage `json:"usage"`
	Choices  []struct {
		Delta streamDelta `json:"delta"`
	} `json:"choices"`
}

type accumulatedToolCall struct {
	ID        string
	Name      string
	Arguments string
}

type streamResult struct {
	Text     string
	Model    string
	Provider string
	Usage    usage
	Messages []chatMessage
}

var streamIdleTimeout = 120 * time.Second

var streamHTTPClient = func() *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = streamIdleTimeout
	return &http.Client{Transport: transport}
}()

type scanEvent struct {
	line string
	err  error
}

var toolDefinitions = []any{
	map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "web_search",
			"description": "Search the web for real-time or up-to-date information. Use this for current events, recent news, live data, or anything requiring fresh information.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "The search query to look up on the web.",
					},
				},
				"required": []string{"query"},
			},
		},
	},
}

func streamPrompt(prompt, model string, provider any, chat bool, ui outputMode) error {
	messages := []chatMessage{
		{"role": "system", "content": "Today's date is " + time.Now().Format("2006-01-02") + "."},
		{"role": "user", "content": prompt},
	}
	url, headers, payload, err := buildRequest(model, messages, provider, true, true)
	if err != nil {
		return err
	}
	printBanner(ui)

	result, err := doStream(url, headers, payload, ui, 0)
	if err != nil {
		return err
	}
	if !chat {
		return nil
	}

	reader := bufio.NewReader(os.Stdin)
	messages = result.Messages
	for {
		fmt.Fprint(os.Stdout, "\nFollow-up (or q to quit): ")
		followup, readErr := reader.ReadString('\n')
		followup = strings.TrimSpace(followup)
		if readErr != nil && followup == "" {
			if readErr == io.EOF {
				fmt.Fprintln(os.Stdout)
				return nil
			}
			return fmt.Errorf("read follow-up: %w", readErr)
		}
		if followup == "" {
			continue
		}
		switch strings.ToLower(followup) {
		case "q", "quit", "exit":
			return nil
		}

		messages = append(messages,
			chatMessage{"role": "assistant", "content": result.Text},
			chatMessage{"role": "user", "content": followup},
		)
		payload.Messages = messages
		result, err = doStream(url, headers, payload, ui, 0)
		if err != nil {
			return err
		}
		messages = result.Messages
	}
}

func buildRequest(model string, messages []chatMessage, provider any, stream, withTools bool) (string, http.Header, apiRequest, error) {
	url := openRouterURL
	headers := make(http.Header)
	headers.Set("Content-Type", "application/json")

	if isMimoModel(model) {
		key, err := environmentKey("MIMO_API_KEY")
		if err != nil {
			return "", nil, apiRequest{}, err
		}
		url = mimoURL
		headers.Set("api-key", key)
	} else {
		key, err := environmentKey("OPENROUTER_API_KEY")
		if err != nil {
			return "", nil, apiRequest{}, err
		}
		headers.Set("Authorization", "Bearer "+key)
		headers.Set("HTTP-Referer", "https://github.com/ai-cli")
		headers.Set("X-Title", "ai-cli")
	}

	payload := apiRequest{
		Model:    apiModelName(model),
		Messages: messages,
		Stream:   stream,
	}
	if stream {
		payload.StreamOptions = map[string]bool{"include_usage": true}
	}
	if withTools && !isMimoModel(model) {
		payload.Tools = toolDefinitions
	}
	if providerEnabled(provider) {
		payload.Provider = provider
	}
	return url, headers, payload, nil
}

func providerEnabled(provider any) bool {
	switch value := provider.(type) {
	case nil:
		return false
	case bool:
		return value
	case string:
		return value != ""
	case float64:
		return value != 0
	case []any:
		return len(value) > 0
	case map[string]any:
		return len(value) > 0
	default:
		return true
	}
}

func environmentKey(name string) (string, error) {
	key := os.Getenv(name)
	if key == "" {
		return "", fmt.Errorf("%s environment variable is not set; export it or add it to your shell profile", name)
	}
	return key, nil
}

func isMimoModel(model string) bool {
	return strings.HasPrefix(model, "mimo/") || strings.HasPrefix(model, "mimo-")
}

func apiModelName(model string) string {
	return strings.TrimPrefix(model, "mimo/")
}

func doStream(url string, headers http.Header, payload apiRequest, ui outputMode, toolRound int) (streamResult, error) {
	if toolRound > 8 {
		return streamResult{}, fmt.Errorf("stopped after 8 consecutive tool-call rounds")
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return streamResult{}, fmt.Errorf("encode request: %w", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return streamResult{}, fmt.Errorf("create request: %w", err)
	}
	req.Header = headers.Clone()

	waiting := !ui.Plain
	if waiting {
		fmt.Fprint(os.Stdout, ansi(true, "2", "  thinking…"))
	}
	requestStart := time.Now()
	resp, err := streamHTTPClient.Do(req)
	if err != nil {
		clearWaiting(waiting)
		return streamResult{}, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		clearWaiting(waiting)
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return streamResult{}, httpStatusError(resp.StatusCode, data)
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 2*1024*1024)
	events := make(chan scanEvent)
	go scanStream(ctx, scanner, events)
	idleTimer := time.NewTimer(streamIdleTimeout)
	defer idleTimer.Stop()

	var collected strings.Builder
	usedModel := payload.Model
	providerName := ""
	toolCalls := make(map[int]*accumulatedToolCall)
	var finalUsage usage
	var firstToken time.Time
	tokenChunks := 0
	streamDone := false

	for !streamDone {
		select {
		case <-idleTimer.C:
			clearWaiting(waiting)
			cancel()
			_ = resp.Body.Close()
			return streamResult{}, fmt.Errorf("response stream idle for %s", streamIdleTimeout)
		case event, ok := <-events:
			if !ok {
				streamDone = true
				continue
			}
			if event.err != nil {
				clearWaiting(waiting)
				return streamResult{}, fmt.Errorf("read response stream: %w", event.err)
			}
			resetTimer(idleTimer, streamIdleTimeout)
			line := event.line
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
			if data == "[DONE]" {
				streamDone = true
				cancel()
				continue
			}
			var chunk streamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				clearWaiting(waiting)
				return streamResult{}, fmt.Errorf("parse stream event: %w", err)
			}
			if chunk.Model != "" {
				usedModel = chunk.Model
			}
			if chunk.Provider != "" {
				providerName = chunk.Provider
			}
			if chunk.Usage != nil {
				finalUsage = *chunk.Usage
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			delta := chunk.Choices[0].Delta
			for _, incoming := range delta.ToolCalls {
				call := toolCalls[incoming.Index]
				if call == nil {
					call = &accumulatedToolCall{}
					toolCalls[incoming.Index] = call
				}
				if incoming.ID != "" {
					call.ID = incoming.ID
				}
				if incoming.Function.Name != "" {
					call.Name = incoming.Function.Name
				}
				call.Arguments += incoming.Function.Arguments
			}
			if delta.Content == "" {
				continue
			}
			if firstToken.IsZero() {
				firstToken = time.Now()
				clearWaiting(waiting)
				waiting = false
			}
			tokenChunks++
			collected.WriteString(delta.Content)
			fmt.Fprint(os.Stdout, delta.Content)
		}
	}
	clearWaiting(waiting)
	if collected.Len() > 0 {
		fmt.Fprintln(os.Stdout)
	}

	messages := payload.Messages
	if len(toolCalls) > 0 {
		messages = appendToolResults(messages, toolCalls, ui)
		payload.Messages = messages
		return doStream(url, headers, payload, ui, toolRound+1)
	}

	if finalUsage.CompletionTokens == 0 {
		finalUsage.CompletionTokens = tokenChunks
	}
	elapsed := time.Since(requestStart)
	generation := time.Duration(0)
	var ttft time.Duration
	if !firstToken.IsZero() {
		generation = time.Since(firstToken)
		ttft = firstToken.Sub(requestStart)
	}
	printStats(ui, usedModel, providerName, elapsed, generation, finalUsage.CompletionTokens, ttft)
	return streamResult{
		Text:     collected.String(),
		Model:    usedModel,
		Provider: providerName,
		Usage:    finalUsage,
		Messages: messages,
	}, nil
}

func scanStream(ctx context.Context, scanner *bufio.Scanner, events chan<- scanEvent) {
	defer close(events)
	for scanner.Scan() {
		select {
		case events <- scanEvent{line: scanner.Text()}:
		case <-ctx.Done():
			return
		}
	}
	if err := scanner.Err(); err != nil {
		select {
		case events <- scanEvent{err: err}:
		case <-ctx.Done():
		}
	}
}

func resetTimer(timer *time.Timer, duration time.Duration) {
	if !timer.Stop() {
		select {
		case <-timer.C:
		default:
		}
	}
	timer.Reset(duration)
}

func clearWaiting(waiting bool) {
	if waiting {
		fmt.Fprint(os.Stdout, "\r\033[2K")
	}
}

func appendToolResults(messages []chatMessage, calls map[int]*accumulatedToolCall, ui outputMode) []chatMessage {
	indices := make([]int, 0, len(calls))
	for index := range calls {
		indices = append(indices, index)
	}
	sort.Ints(indices)

	serialized := make([]any, 0, len(indices))
	for _, index := range indices {
		call := calls[index]
		serialized = append(serialized, map[string]any{
			"id":   call.ID,
			"type": "function",
			"function": map[string]any{
				"name":      call.Name,
				"arguments": call.Arguments,
			},
		})
	}
	messages = append(messages, chatMessage{"role": "assistant", "tool_calls": serialized})

	for _, index := range indices {
		call := calls[index]
		query := ""
		var args struct {
			Query string `json:"query"`
		}
		_ = json.Unmarshal([]byte(call.Arguments), &args)
		query = args.Query
		if !ui.Plain {
			fmt.Fprintf(os.Stdout, "  ⟳ calling %s(%s)\n", call.Name, query)
		}
		messages = append(messages, chatMessage{
			"role":         "tool",
			"tool_call_id": call.ID,
			"content":      executeTool(call.Name, call.Arguments),
		})
	}
	return messages
}

func httpStatusError(status int, body []byte) error {
	detail := ""
	var payload struct {
		Error   any    `json:"error"`
		Message string `json:"message"`
	}
	if json.Unmarshal(body, &payload) == nil {
		switch value := payload.Error.(type) {
		case map[string]any:
			detail, _ = value["message"].(string)
		case string:
			detail = value
		}
		if detail == "" {
			detail = payload.Message
		}
	}
	if detail == "" {
		detail = strings.TrimSpace(string(body))
	}
	if len(detail) > 300 {
		detail = detail[:300]
	}
	hint := map[int]string{
		400: "check the model ID and parameters",
		401: "check your API key environment variable",
		402: "top up your account and retry",
		403: "access denied for this key/model",
		404: "model not found; run `ai config models` for popular IDs",
		408: "upstream timeout; try again",
		429: "rate limited; wait a moment or route to another provider",
		502: "the upstream provider may be down",
		503: "service unavailable; try again shortly",
	}[status]
	message := fmt.Sprintf("HTTP %d", status)
	if detail != "" {
		message += ": " + detail
	}
	if hint != "" {
		message += " — " + hint
	}
	return fmt.Errorf("%s", message)
}

func printBanner(ui outputMode) {
	if ui.Plain || !ui.ShowBanner {
		return
	}
	color := themeColor(ui.Theme)
	logo := []string{
		`  __                       .__              .__           .__            __   `,
		`_/  |_  ___________  _____ |__| ____ _____  |  |     ____ |  |__ _____ _/  |_ `,
		`\   __\/ __ \_  __ \/     \|  |/    \\__  \ |  |   _/ ___\|  |  \\__  \\   __\\`,
		` |  | \  ___/|  | \/  Y Y  \  |   |  \/ __ \|  |__ \  \___|   Y  \/ __ \|  |  `,
		` |__|  \___  >__|  |__|_|  /__|___|  (____  /____/  \___  >___|  (____  /__|  `,
		`           \/            \/        \/     \/            \/     \/     \/      `,
	}
	fmt.Fprintln(os.Stdout)
	for _, line := range logo {
		fmt.Fprintln(os.Stdout, ansi(true, color, line))
	}
	fmt.Fprintln(os.Stdout, ansi(true, "2", strings.Repeat("─", 78)))
}

func printStats(ui outputMode, model, provider string, elapsed, generation time.Duration, tokens int, ttft time.Duration) {
	if ui.Plain {
		return
	}
	rate := 0.0
	if generation > 0 {
		rate = float64(tokens) / generation.Seconds()
	}
	parts := []string{"✦ " + model}
	if provider != "" {
		parts = append(parts, "via "+provider)
	}
	parts = append(parts, fmt.Sprintf("%d tokens", tokens))
	if ttft > 0 {
		parts = append(parts, fmt.Sprintf("%.2fs to first token", ttft.Seconds()))
	}
	parts = append(parts, fmt.Sprintf("%.1f tok/s", rate), fmt.Sprintf("%.1fs", elapsed.Seconds()))
	fmt.Fprintln(os.Stdout, ansi(true, "2", strings.Repeat("─", 78)))
	fmt.Fprintln(os.Stdout, strings.Join(parts, "  │  "))
	fmt.Fprintln(os.Stdout)
}

func ansi(enabled bool, code, text string) string {
	if !enabled {
		return text
	}
	return "\033[" + code + "m" + text + "\033[0m"
}

func themeColor(setting string) string {
	mode := setting
	if mode == "auto" {
		if isDarkMode() {
			mode = "dark"
		} else {
			mode = "light"
		}
	}
	switch mode {
	case "light":
		return "35"
	case "retro":
		return "92"
	default:
		return "95"
	}
}

func isDarkMode() bool {
	if runtime.GOOS == "darwin" {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		output, err := exec.CommandContext(ctx, "defaults", "read", "-g", "AppleInterfaceStyle").Output()
		return err == nil && strings.EqualFold(strings.TrimSpace(string(output)), "dark")
	}
	parts := strings.Split(os.Getenv("COLORFGBG"), ";")
	if len(parts) > 0 {
		var background int
		if _, err := fmt.Sscanf(parts[len(parts)-1], "%d", &background); err == nil {
			return background < 8
		}
	}
	return true
}

type ensembleState struct {
	Model    string
	Text     string
	Resolved string
	Usage    usage
	Elapsed  time.Duration
	Err      error
}

type indexedState struct {
	Index int
	State ensembleState
}

func ensemblePrompt(prompt string, models []string, consensusModel string, provider any, ui outputMode) error {
	if len(models) < 2 {
		return fmt.Errorf("ensemble mode needs at least 2 models (set with `ai config set ensemble_models`)")
	}
	messages := []chatMessage{
		{"role": "system", "content": "Today's date is " + time.Now().Format("2006-01-02") + "."},
		{"role": "user", "content": prompt},
	}
	printBanner(ui)
	if !ui.Plain {
		fmt.Fprintf(os.Stdout, "⚡ Querying %d models in parallel…\n", len(models))
	}

	states := make([]ensembleState, len(models))
	results := make(chan indexedState, len(models))
	for index, model := range models {
		go func(index int, model string) {
			results <- indexedState{Index: index, State: queryModel(model, messages, provider)}
		}(index, model)
	}
	for range models {
		result := <-results
		states[result.Index] = result.State
		if !ui.Plain {
			if result.State.Err != nil {
				fmt.Fprintf(os.Stdout, "  ✗ %-36s %v\n", shortModel(result.State.Model), result.State.Err)
			} else {
				fmt.Fprintf(os.Stdout, "  ✓ %-36s %d tokens · %.1fs\n", shortModel(result.State.Model), result.State.Usage.CompletionTokens, result.State.Elapsed.Seconds())
			}
		}
	}

	succeeded := 0
	for _, state := range states {
		if state.Err == nil && strings.TrimSpace(state.Text) != "" {
			succeeded++
		}
	}
	if succeeded == 0 {
		return fmt.Errorf("all models failed to respond")
	}

	for index, state := range states {
		if state.Err != nil || strings.TrimSpace(state.Text) == "" {
			continue
		}
		header := fmt.Sprintf("Model %d · %s", index+1, shortModel(state.Model))
		if ui.Plain {
			fmt.Fprintf(os.Stdout, "\n## %s\n\n%s\n", header, strings.TrimSpace(state.Text))
		} else {
			fmt.Fprintf(os.Stdout, "\n%s\n%s\n%s\n", header, strings.Repeat("─", len(header)), strings.TrimSpace(state.Text))
			fmt.Fprintf(os.Stdout, "%d tokens · %.1fs\n", state.Usage.CompletionTokens, state.Elapsed.Seconds())
		}
	}
	if !ui.Plain {
		fmt.Fprintf(os.Stdout, "\n✦ Consolidating with %s…\n", shortModel(consensusModel))
	}

	var answers strings.Builder
	for index, state := range states {
		if state.Err == nil && strings.TrimSpace(state.Text) != "" {
			fmt.Fprintf(&answers, "<answer model=\"Model %d\">\n%s\n</answer>\n\n", index+1, strings.TrimSpace(state.Text))
		}
	}
	consensusMessages := []chatMessage{
		{
			"role":    "system",
			"content": "You are an expert editor. You are given a user's prompt and several independent answers from different AI models. Synthesize a single, best possible response: combine the strongest points, resolve contradictions in favor of correctness, drop errors, and fill gaps. Do not mention the individual models, that multiple answers exist, or that you are consolidating. Just give the final answer.",
		},
		{
			"role":    "user",
			"content": fmt.Sprintf("User prompt:\n%s\n\nCandidate answers:\n%s", prompt, answers.String()),
		},
	}
	url, headers, payload, err := buildRequest(consensusModel, consensusMessages, provider, true, false)
	if err != nil {
		return err
	}
	_, err = doStream(url, headers, payload, ui, 0)
	return err
}

func queryModel(model string, messages []chatMessage, provider any) ensembleState {
	state := ensembleState{Model: model}
	url, headers, payload, err := buildRequest(model, messages, provider, false, false)
	if err != nil {
		state.Err = err
		return state
	}
	body, err := json.Marshal(payload)
	if err != nil {
		state.Err = fmt.Errorf("encode request: %w", err)
		return state
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		state.Err = fmt.Errorf("create request: %w", err)
		return state
	}
	req.Header = headers.Clone()
	started := time.Now()
	resp, err := http.DefaultClient.Do(req)
	state.Elapsed = time.Since(started)
	if err != nil {
		state.Err = fmt.Errorf("request failed: %w", err)
		return state
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	state.Elapsed = time.Since(started)
	if err != nil {
		state.Err = fmt.Errorf("read response: %w", err)
		return state
	}
	if resp.StatusCode != http.StatusOK {
		state.Err = httpStatusError(resp.StatusCode, data)
		return state
	}
	var decoded struct {
		Model   string `json:"model"`
		Usage   usage  `json:"usage"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		state.Err = fmt.Errorf("parse response: %w", err)
		return state
	}
	if len(decoded.Choices) == 0 {
		state.Err = fmt.Errorf("response contained no choices")
		return state
	}
	state.Text = decoded.Choices[0].Message.Content
	state.Resolved = decoded.Model
	state.Usage = decoded.Usage
	return state
}

func shortModel(model string) string {
	if _, suffix, ok := strings.Cut(model, "/"); ok {
		return suffix
	}
	return model
}
