package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestFindConfigCommand(t *testing.T) {
	args, ok := findConfigCommand([]string{"--plain", "-m", "ignored", "config", "show"})
	if !ok || strings.Join(args, " ") != "show" {
		t.Fatalf("config command not found: %v %v", args, ok)
	}
	if _, ok := findConfigCommand([]string{"explain", "config"}); ok {
		t.Fatal("prompt text was mistaken for the config command")
	}
}

func TestParseCLI(t *testing.T) {
	opts, err := parseCLI([]string{"-f", "a.go", "--file=b.go", "-m", "test/model", "-c", "-e", "explain", "this"})
	if err != nil {
		t.Fatal(err)
	}
	if opts.Model != "test/model" {
		t.Fatalf("model = %q", opts.Model)
	}
	if got := strings.Join(opts.Files, ","); got != "a.go,b.go" {
		t.Fatalf("files = %q", got)
	}
	if got := strings.Join(opts.PromptArgs, " "); got != "explain this" {
		t.Fatalf("prompt = %q", got)
	}
	if !opts.Chat || !opts.Ensemble {
		t.Fatalf("flags not parsed: %#v", opts)
	}

	opts, err = parseCLI([]string{"-ce", "-mtest/attached", "-fREADME.md", "explain", "--help"})
	if err != nil {
		t.Fatal(err)
	}
	if opts.Model != "test/attached" || strings.Join(opts.Files, ",") != "README.md" || !opts.Chat || !opts.Ensemble {
		t.Fatalf("attached short options not parsed: %#v", opts)
	}
	if opts.Help || strings.Join(opts.PromptArgs, " ") != "explain --help" {
		t.Fatalf("trailing help was not preserved as prompt text: %#v", opts)
	}

	if _, err := parseCLI([]string{"--model"}); err == nil {
		t.Fatal("missing model value did not fail")
	}
}

func TestBuildRequestRoutesProviders(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "openrouter-test")
	t.Setenv("MIMO_API_KEY", "mimo-test")
	messages := []chatMessage{{"role": "user", "content": "hello"}}

	url, headers, payload, err := buildRequest("openai/test", messages, nil, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if url != openRouterURL || headers.Get("Authorization") != "Bearer openrouter-test" {
		t.Fatalf("unexpected OpenRouter request: %s %#v", url, headers)
	}
	if len(payload.Tools) != 1 || !payload.StreamOptions["include_usage"] {
		t.Fatalf("stream request missing tools or usage: %#v", payload)
	}
	for _, falsey := range []any{"", false, float64(0), []any{}, map[string]any{}} {
		_, _, payload, err = buildRequest("openai/test", messages, falsey, true, true)
		if err != nil {
			t.Fatal(err)
		}
		if payload.Provider != nil {
			t.Fatalf("falsey provider was included: %#v", falsey)
		}
	}

	url, headers, payload, err = buildRequest("mimo/mimo-v2.5", messages, nil, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if url != mimoURL || headers.Get("api-key") != "mimo-test" || payload.Model != "mimo-v2.5" {
		t.Fatalf("unexpected MiMo request: %s %#v %#v", url, headers, payload)
	}
	if len(payload.Tools) != 0 {
		t.Fatal("MiMo request should not include OpenRouter tools")
	}
}

func TestDoStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"model\":\"resolved/model\",\"provider\":\"TestProvider\",\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"usage\":{\"completion_tokens\":2},\"choices\":[]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	result, err := doStream(server.URL, make(http.Header), apiRequest{
		Model:    "requested/model",
		Messages: []chatMessage{{"role": "user", "content": "hello"}},
		Stream:   true,
	}, outputMode{Plain: true}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "hello world" || result.Model != "resolved/model" || result.Provider != "TestProvider" {
		t.Fatalf("unexpected result: %#v", result)
	}
	if result.Usage.CompletionTokens != 2 {
		t.Fatalf("completion tokens = %d", result.Usage.CompletionTokens)
	}
}

func TestPlainToolCallDoesNotWriteStatus(t *testing.T) {
	originalStdout := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = writer
	messages := appendToolResults(nil, map[int]*accumulatedToolCall{
		0: {ID: "call-1", Name: "unknown", Arguments: `{}`},
	}, outputMode{Plain: true})
	_ = writer.Close()
	os.Stdout = originalStdout
	output, err := io.ReadAll(reader)
	_ = reader.Close()
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != 0 || len(messages) != 2 {
		t.Fatalf("plain tool call leaked output %q or messages were lost: %#v", output, messages)
	}
}

func TestDoStreamIdleTimeout(t *testing.T) {
	previousTimeout := streamIdleTimeout
	streamIdleTimeout = 20 * time.Millisecond
	defer func() { streamIdleTimeout = previousTimeout }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()

	_, err := doStream(server.URL, make(http.Header), apiRequest{Model: "test", Stream: true}, outputMode{Plain: true}, 0)
	if err == nil || !strings.Contains(err.Error(), "stream idle") {
		t.Fatalf("expected stream idle timeout, got %v", err)
	}
}

func TestHTTPStatusError(t *testing.T) {
	err := httpStatusError(401, []byte(`{"error":{"message":"bad key"}}`))
	message := err.Error()
	if !strings.Contains(message, "bad key") || !strings.Contains(message, "check your API key") {
		t.Fatalf("unexpected message: %s", message)
	}
}
