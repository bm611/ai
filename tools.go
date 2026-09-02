package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const exaSearchURL = "https://api.exa.ai/search"

func executeTool(name, arguments string) string {
	var result string
	var err error
	switch name {
	case "web_search":
		var args struct {
			Query string `json:"query"`
		}
		if err = json.Unmarshal([]byte(arguments), &args); err == nil {
			if args.Query == "" {
				err = fmt.Errorf("query is required")
			} else {
				result, err = webSearch(args.Query)
			}
		}
	default:
		err = fmt.Errorf("unknown tool: %s", name)
	}
	if err == nil {
		return result
	}
	data, _ := json.Marshal(map[string]string{"error": fmt.Sprintf("%T: %v", err, err)})
	return string(data)
}

func webSearch(query string) (string, error) {
	key := os.Getenv("EXA_API_KEY")
	if key == "" {
		return "", fmt.Errorf("EXA_API_KEY environment variable is not set")
	}
	payload := map[string]any{
		"query":      query,
		"numResults": 5,
		"contents": map[string]any{
			"highlights": map[string]any{"maxCharacters": 4000},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("encode Exa request: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, exaSearchURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create Exa request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", key)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("Exa request failed: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return "", fmt.Errorf("read Exa response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Exa HTTP %d: %s", resp.StatusCode, string(data))
	}

	var decoded struct {
		Results []struct {
			Title      string   `json:"title"`
			URL        string   `json:"url"`
			Highlights []string `json:"highlights"`
		} `json:"results"`
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		return "", fmt.Errorf("parse Exa response: %w", err)
	}
	results := make([]map[string]any, 0, len(decoded.Results))
	for _, item := range decoded.Results {
		entry := map[string]any{"title": item.Title, "url": item.URL}
		if len(item.Highlights) > 0 {
			entry["highlights"] = item.Highlights
		}
		results = append(results, entry)
	}
	output, err := json.Marshal(results)
	if err != nil {
		return "", fmt.Errorf("encode Exa results: %w", err)
	}
	return string(output), nil
}
