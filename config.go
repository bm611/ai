package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

const (
	defaultModel          = "z-ai/glm-5.3-flash"
	defaultTheme          = "auto"
	defaultConsensusModel = "deepseek/deepseek-v4-pro"
)

var defaultEnsembleModels = []string{
	"deepseek/deepseek-v4-flash",
	"google/gemini-3.1-flash-lite-preview",
}

func defaultConfig() map[string]any {
	return map[string]any{
		"model":           defaultModel,
		"provider":        nil,
		"theme":           defaultTheme,
		"ensemble_models": append([]string(nil), defaultEnsembleModels...),
		"consensus_model": defaultConsensusModel,
	}
}

func configPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("find home directory: %w", err)
	}
	return filepath.Join(home, ".config", "ai-cli", "config.json"), nil
}

func loadConfig() (map[string]any, error) {
	path, err := configPath()
	if err != nil {
		return nil, err
	}

	cfg := defaultConfig()
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		if err := saveConfig(cfg); err != nil {
			return nil, err
		}
		return cfg, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var stored map[string]any
	if err := json.Unmarshal(data, &stored); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}
	for key, value := range stored {
		cfg[key] = value
	}
	if !reflect.DeepEqual(cfg, stored) {
		if err := saveConfig(cfg); err != nil {
			return nil, err
		}
	}
	return cfg, nil
}

func saveConfig(cfg map[string]any) error {
	path, err := configPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create config directory: %w", err)
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("encode config: %w", err)
	}
	data = append(data, '\n')

	tmp, err := os.CreateTemp(filepath.Dir(path), ".config-*.json")
	if err != nil {
		return fmt.Errorf("create temporary config: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o644); err != nil {
		tmp.Close()
		return fmt.Errorf("set config permissions: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write config: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close config: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("replace config: %w", err)
	}
	return nil
}

func configString(cfg map[string]any, key, fallback string) string {
	value, ok := cfg[key].(string)
	if !ok || value == "" {
		return fallback
	}
	return value
}

func configStrings(cfg map[string]any, key string) []string {
	switch value := cfg[key].(type) {
	case []any:
		items := make([]string, 0, len(value))
		for _, item := range value {
			if text, ok := item.(string); ok && strings.TrimSpace(text) != "" {
				items = append(items, strings.TrimSpace(text))
			}
		}
		return items
	case []string:
		return append([]string(nil), value...)
	case string:
		return splitList(value)
	default:
		return nil
	}
}

func splitList(value string) []string {
	parts := strings.Split(value, ",")
	items := make([]string, 0, len(parts))
	for _, part := range parts {
		if part = strings.TrimSpace(part); part != "" {
			items = append(items, part)
		}
	}
	return items
}

func displayValue(value any) string {
	if value == nil {
		return "not set"
	}
	if data, err := json.Marshal(value); err == nil {
		return string(data)
	}
	return fmt.Sprint(value)
}
